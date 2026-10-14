---
type: Guide
title: IndexPut / Scatter 算子（index-write） 实现手册
description: '算子类别: indexwrite（按索引向输出写入或原子累加。'
tags: [task.implementation, topic.compute.indexing, topic.implementation.atomic, topic.implementation.masking, topic.compilation, topic.alignment]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# IndexPut / Scatter 算子（index-write） 实现手册

> 类别约束见 [index_computation_constraints.md](../constraints/index_computation_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 IndexPut / Scatter 算子（index-write）

**算子类别**: `index-write`（按索引向输出写入或原子累加；`index_put_` / `scatter` / `scatter_add` 等）
**典型特征**: 输出张量形状固定，写入位置由索引映射；存在写冲突、原子操作粒度、坐标解码开销
**性能基准**: Level1 索引计算类算子旧版本均能达到 ≥0.6x 几何平均加速比 vs torch

### §3.0 首次生成必读：为什么必须把主要框架写对

IndexPut / Scatter 是**并行策略和原子粒度强相关**的算子：连续宽度 `E/D` 较大时必须用向量 `tl.atomic_add`，`ndim ≤ 4` 时必须编译期维度特化，重复索引时需要冲突处理。**首次生成如果把框架写偏（例如用标量 atomic 内层循环、通用 ND kernel 运行时解码、未处理 accumulate=False 的重复索引），后续迭代很难把性能救回来**——典型加速比会跌至 0.05x~0.15x。

### §3.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 连续宽度 `E/D ≥ 64` 时必须使用向量 `tl.atomic_add`

- **必须**在连续输出段 `E ≥ 64/128` 时使用向量 `tl.atomic_add(ptr + arange(0, BLOCK_E), vals, mask=...)`。
- **禁止**先对索引/值做向量 `tl.load` 得到 `idx_tile`/`val_tile`，再用 `tl.get_element(..., [j])` 拆成标量，然后对每个元素单独调用 `tl.atomic_add(output_ptr + idx * E + e, val)`。
- **Why:** 向量 atomic 一次覆盖多个连续元素，原子数下降 1~2 个数量级；拆成标量会破坏 SIMD 并爆炸原子数，典型加速比跌至 0.05x~0.15x。
- **How to apply:** 见 §3.3 L3.1 `index_put_1d_kernel`。

#### L1.2 根据 `E/D` 宽度选择并行轴

- **必须**根据 shape 选择并行轴：
  - `E/D` 较宽（≥128/256）：**index-并行 + E 维向量 atomic_add**。
  - `E/D` 较小或冲突密集：**E-并行 + 向量 atomic_add**，每个 core 串行扫描 indices。
  - 索引基本唯一且 `accumulate=False`：可走 **输出位置并行 + 向量 store**。
- **Why:** 并行轴若不与连续访问对齐，会引入标量原子竞争和离散写入。

#### L1.3 `ndim ≤ 4` 必须编译期维度特化

- **必须**在 host 侧 `dim` 归一化并 `tl.constexpr` 特化。
- **禁止**在 kernel 内运行时从 global buffer 读取 shape/stride 并做 `%`/`//` 解码。
- **Why:** 运行时坐标解码引入标量开销，且无法合并访存。
- **How to apply:** 见 §3.3 L3.2 `scatter_add_nd_kernel`。

#### L1.4 `accumulate=False` 且存在重复索引时必须实现 last-write-wins

- **必须**在 `accumulate=False` 且存在重复索引时，实现 last-write-wins（单 block 串行或两阶段）。
- **Why:** 缺少写冲突处理会导致结果不稳定。

#### L1.5 边界检查与 padding 清零必须外置

- **必须**覆盖 `index < 0`、`index >= dim_size` 等边界检查，越界写加 mask。
- **必须**将 `padding_idx` 等排除位置清零作为独立轻量 kernel，不在热 scatter 循环内分支。
- **Why:** 热循环内分支破坏向量化；越界写触发 ACL 错误。

### §3.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分派决策树

1. Host 侧 `dim` 归一化并 `tl.constexpr` 特化。
2. 若 `accumulate=True` 且输出连续宽度 `E` 很大（`≥ 128/256`）：
   - 索引重复概率低或冲突可接受：走 **index-并行 + E 维向量 atomic_add**。
   - 冲突不可接受：走 **E-并行 + 串行扫描 indices + E 维向量 atomic_add**。
3. 若 `accumulate=True` 但 `E` 较小或通用 ND：走 **E-并行 + 标量 atomic**。
4. 若 `accumulate=False` 且索引唯一：可走输出位置并行 + 向量 store。
5. 若 `accumulate=False` 且存在重复索引：实现 last-write-wins（单 block 串行或两阶段）。

### §3.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `index_put_1d_kernel`（1D Output-Position-Parallel IndexPut）

适用条件：索引写入 1D 输出，`accumulate` 开启时冲突可控或索引基本唯一。

```python
@triton.jit
def index_put_1d_kernel(
    index_ptr, values_ptr, out_ptr,
    M, N,
    accumulate: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pos_start = pid * BLOCK_M
    pos_end = (pid + 1) * BLOCK_M
    offs = tl.arange(0, BLOCK_M)
    mask = offs < M

    # 每个 program 负责一段连续输出位置
    for idx_pos in range(0, M, BLOCK_M):
        target_idx = tl.load(index_ptr + idx_pos + offs, mask=mask).to(tl.int32)
        val = tl.load(values_ptr + idx_pos + offs, mask=mask)
        in_range = (target_idx >= pos_start) & (target_idx < pos_end) & mask
        if accumulate:
            tl.atomic_add(out_ptr + target_idx, val, mask=in_range)
        else:
            tl.store(out_ptr + target_idx, val, mask=in_range)
```

#### L3.2 `scatter_add_nd_kernel`（ND Scatter-Add with Coordinate Decomposition）

适用条件：通用 ND scatter_add，`ndim` 在编译期确定。

```python
@triton.jit
def scatter_add_nd_kernel(
    src_ptr, index_ptr, out_ptr,
    shape_ptr, flat_stride_ptr, out_stride_ptr, src_stride_ptr, index_stride_ptr,
    M, E,
    DIM: tl.constexpr, NDIM: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_e = tl.program_id(0)
    e_start = pid_e * BLOCK_E
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    # 将扁平 e 解码为 ND 坐标（NDIM 为 constexpr，循环会被展开）
    e = e_start + offs_e
    out_base = out_ptr
    src_base = src_ptr
    idx_base = index_ptr
    for j in range(NDIM):
        size = tl.load(shape_ptr + j)
        flat_stride = tl.load(flat_stride_ptr + j)
        coord = (e // flat_stride) % size
        out_base = out_base + coord * tl.load(out_stride_ptr + j)
        src_base = src_base + coord * tl.load(src_stride_ptr + j)
        if j != DIM:
            idx_base = idx_base + coord * tl.load(index_stride_ptr + j)

    # 沿索引维度 m 串行累加
    for m in range(0, M):
        idx = tl.load(idx_base + m * tl.load(index_stride_ptr + DIM)).to(tl.int32)
        val = tl.load(src_base + m * tl.load(src_stride_ptr + DIM)).to(tl.float32)
        tl.atomic_add(out_base + idx * tl.load(out_stride_ptr + DIM), val, mask=mask_e)
```

> 注意：此为通用 fallback。`ndim ≤ 4` 时，应改用 `DIM=0/1/2/3` 的特化版本，避免运行时循环解码。

### §3.4 IndexPut / Scatter 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 参考 Level1 索引计算类算子 | 19_IndexPut / 21_Scatter | ≥0.6x | 旧版本几何平均加速比 vs torch；新版本若丢失向量 atomic/维度特化普遍跌至 0.05x~0.4x |

**关键结论**:
1. 连续宽度 `E/D ≥ 64` 时必须使用向量 `tl.atomic_add`。
2. `ndim ≤ 4` 时必须编译期维度特化。
3. `accumulate=False` 且存在重复索引时必须实现 last-write-wins。

---
