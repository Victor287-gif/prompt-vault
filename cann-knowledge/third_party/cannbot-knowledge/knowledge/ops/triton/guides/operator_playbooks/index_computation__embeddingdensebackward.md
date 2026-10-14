---
type: Guide
title: EmbeddingDenseBackward 算子（index-accumulate） 实现手册
description: '算子类别: indexaccumulate（按索引将梯度累加到 embedding 权重。'
tags: [task.implementation, topic.implementation.masking, topic.alignment, topic.implementation.atomic, topic.compute.indexing, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# EmbeddingDenseBackward 算子（index-accumulate） 实现手册

> 类别约束见 [index_computation_constraints.md](../constraints/index_computation_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §4 EmbeddingDenseBackward 算子（index-accumulate）

**算子类别**: `index-accumulate`（按索引将梯度累加到 embedding 权重；`embedding_dense_backward`）
**典型特征**: 尾维度 `E`/`D` 通常较宽，索引重复概率高，fp16/bf16 累加精度敏感
**性能基准**: Level1 索引计算类算子旧版本均能达到 ≥0.6x 几何平均加速比 vs torch

### §4.0 首次生成必读：为什么必须把主要框架写对

EmbeddingDenseBackward 是**三阶段结构必须一次写对**的算子：频数统计、连续维向量 scatter_add、padding 清零。若试图用单个 kernel 内双重循环实现，原子数会随 `N×E` 爆炸，性能极差且易触发 Vector Core timeout。**首次生成如果把框架写偏（例如未拆分为 freq/scatter-add/padding-zero、未用 fp32 workspace、未对连续 E 维做向量 atomic），后续迭代很难救回**。

### §4.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须拆分为三阶段 kernel

- **必须**拆分为 `freq_kernel`、`scatter_add_kernel`、`padding_zero_kernel` 三个 kernel。
- **禁止**把所有可选逻辑塞进同一张 JIT 签名，或用单个 kernel 内双重标量循环实现。
- **Why:** 单个 kernel 内对 `N` 和 `E` 做双重标量循环，每个 `(pos, e)` 单独 `tl.atomic_add`，原子数随 `N×E` 爆炸，性能极差且易触发 Vector Core timeout。
- **How to apply:** 见 §4.2 L2.1 三阶段调用骨架。

#### L1.2 必须对连续 `E`/`D` 维做向量 `tl.atomic_add`

- **必须**按 embedding_dim `E` 分核，每个 program 负责一段连续 `E`，对 `grad_output` 做向量加载后执行向量 `tl.atomic_add`。
- **禁止**用 `tl.get_element(..., [j])` 拆成标量后再 atomic_add。
- **Why:** 向量 atomic 原子数下降 1~2 个数量级；标量提取破坏 SIMD。
- **How to apply:** 见 §4.3 L3.2 `embedding_grad_kernel`。

#### L1.3 fp16/bf16 累加必须使用 fp32 workspace

- **必须**分配 `torch.float32` 中间 buffer 进行 `atomic_add`，最后 cast 回原始 dtype。
- **Why:** fp16/bf16 `atomic_add` 直接累加会导致精度下降。

#### L1.4 `scale_grad_by_freq` 必须查表，禁止二次扫描

- **必须**先用 `freq_kernel` 统计每个 index 出现次数，主 kernel 查 `1/freq` 缩放。
- **禁止**在 `scatter_add_kernel` 内二次扫描 indices 统计频数。
- **Why:** 频数统计与主累加分拆避免热循环内分支和二次遍历。

#### L1.5 `padding_idx` 清零必须使用独立轻量 kernel

- **必须**用 `padding_zero_kernel` 独立轻量 kernel 清零 `padding_idx` 行。
- **禁止**在热 scatter 循环内判断 `if idx != padding_idx`。
- **Why:** 热循环内分支破坏向量化。

### §4.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 三阶段调用骨架

推荐拆分为三个 kernel，避免把所有可选逻辑塞进同一张 JIT 签名：

1. **`freq_kernel`**：N-并行 `tl.atomic_add(freq_ptr + idx, 1.0)`，统计每个输出位置出现次数。
2. **`scatter_add_kernel`**：N 分块，连续 D 维向量 `tl.atomic_add`，按 `1/freq` 缩放，fp32 累加。
3. **`padding_zero_kernel`**：清零 `padding_idx` 对应的行。

### §4.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `freq_kernel`

```python
@triton.jit
def freq_kernel(indices_ptr, freq_ptr, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    idx = tl.load(indices_ptr + offs, mask=mask).to(tl.int32)
    tl.atomic_add(freq_ptr + idx, 1.0, mask=mask)
```

#### L3.2 `embedding_grad_kernel`

```python
@triton.jit
def embedding_grad_kernel(
    grad_ptr, indices_ptr, freq_ptr, out_ptr,
    N, D, scale_grad_by_freq: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    idx = tl.load(indices_ptr + n_offs, mask=n_mask).to(tl.int32)
    scale = 1.0
    if scale_grad_by_freq:
        freq = tl.load(freq_ptr + idx, mask=n_mask)
        scale = 1.0 / freq

    for n in range(BLOCK_N):
        if n_start + n >= N:
            break
        row_idx = idx[n]
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            val = tl.load(grad_ptr + (n_start + n) * D + d_offs, mask=d_mask) * scale
            tl.atomic_add(out_ptr + row_idx * D + d_offs, val, mask=d_mask)
```

#### L3.3 `padding_zero_kernel`

```python
@triton.jit
def padding_zero_kernel(out_ptr, padding_idx, D, BLOCK_D: tl.constexpr):
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    tl.store(out_ptr + padding_idx * D + d_offs, 0.0, mask=d_mask)
```

### §4.4 EmbeddingDenseBackward 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 参考 Level1 索引计算类算子 | 24_EmbeddingDenseBackward | ≥0.6x | 旧版本几何平均加速比 vs torch；新版本若未拆分三阶段/未用 fp32 workspace 普遍跌至 0.05x~0.4x |

**关键结论**:
1. 必须拆分为 freq / scatter-add / padding-zero 三个 kernel。
2. 必须对连续 `E`/`D` 维做向量 `tl.atomic_add`。
3. fp16/bf16 累加必须使用 fp32 workspace。

---
