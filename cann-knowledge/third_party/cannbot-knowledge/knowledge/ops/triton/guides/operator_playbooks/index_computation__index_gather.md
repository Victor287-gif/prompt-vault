---
type: Guide
title: Index / Gather 算子（index-read） 实现手册
description: '算子类别: indexread（按索引从输入中收集元素。'
tags: [task.implementation, topic.compute.indexing, topic.data_movement.layout_transform, topic.implementation.memory_management, topic.implementation.masking, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Index / Gather 算子（index-read） 实现手册

> 类别约束见 [index_computation_constraints.md](../constraints/index_computation_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 Index / Gather 算子（index-read）

**算子类别**: `index-read`（按索引从输入中收集元素；`index_select` / `gather` / `take` / `embedding` 等）
**典型特征**: 索引张量决定数据读取位置，多为 `int32`/`int64`；数据张量多为 `fp16`/`bf16`/`fp32`，维度 1D~4D；核心风险是离散全局内存访问与非连续维度访问
**性能基准**: Level1 索引计算类算子旧版本均能达到 ≥0.6x 几何平均加速比 vs torch

### §2.0 首次生成必读：为什么必须把主要框架写对

Index / Gather 是**访存模式高度依赖 dim 位置**的算子：last-dim 可走 UB 预载 + `tl.gather`，非 last-dim 需评估转置收益，通用 ND kernel 性能极差。**首次生成如果把框架写偏（例如用通用 ND kernel 统所有路径、未将 gather dim 转置到 last-dim、循环边界用输入 shape），后续迭代很难通过局部修 bug 把性能救回来**——典型加速比会从 0.6x 跌至 0.05x~0.2x。

本章按 Layer 1→3 组织：
- **L1 是硬性约束**，首次生成必须全部满足；
- **L2 是 host 分派骨架 + 辅助函数**，必须一次写对；
- **L3 是关键 kernel 实现**，贴出优化的重点代码和那些不容易首次生成就生成出来的代码。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 Host 侧必须按 `dim=0/1/2/3` 启动 `tl.constexpr` 特化 kernel

- **必须**在 host 侧将 `dim` 归一化，并按 `DIM=0/1/2/3` 启动独立 kernel 实例，将 `DIM`/`NDIM` 作为 `tl.constexpr` 传入。
- **禁止**在 kernel 内用运行时 `for j in range(NDIM)` 解码坐标，或用运行时 `if dim == X` 分支。
- **Why:** 运行时坐标解码会引入大量标量除法/取模、非合并访存，且无法使用 UB gather，典型加速比仅 0.05x~0.2x。
- **How to apply:** 见 §2.2 L2.1 分派决策树。

#### L1.2 Gather 必须优先走 UB 预载 + `tl.gather`

- **必须**在 `dim` 对应 last-dim 且 `dim_size ≤ UB_LIMIT` 时，走 **UB 预载整行 + `tl.gather(row, idx, axis=0)`**。
- **必须**在非 last-dim gather 且 `dim_size ≤ UB_LIMIT` 且转置收益 > 2× 拷贝成本时，host 侧 `movedim(dim, -1).contiguous()` 将 gather 维转置到最后，再走 UB 预载 + `tl.gather`，最后转置回。
- **禁止**将所有 gather 都实现为 `x_ptr + idx * stride` 的 direct-indirect 路径。
- **Why:** UB 预载 + `tl.gather` 是 Ascend 上的硬件 gather 路径；direct-indirect fallback 离散全局读取，典型加速比仅 0.05x~0.2x。
- **How to apply:** 见 §2.3 L3.1 `gather_ub_kernel`。

#### L1.3 循环边界必须基于输出/索引张量的 shape

- **必须**使用 `M = index.shape[:dim].prod()` 等基于输出/索引张量的 shape 计算循环边界。
- **禁止**使用 `M = x.shape[:dim].prod()` 等基于输入 shape 的边界。
- **Why:** gather 的输出位置由索引决定，用输入 shape 会导致越界或漏算。

#### L1.4 UB 预载尺寸必须校验 UB 容量上限

- **必须**保证 `tl.gather` 的 `ROW_SIZE` 小于 UB 容量上限（fp32 约 32768 元素）。
- **必须**在 `dim_size > UB_LIMIT` 时分块为 `BLOCK_*` + `BLOCK_*_SUB`。
- **Why:** 一次性预载整行但 `dim_size > UB_LIMIT` 会导致 spilling，反而更慢。

#### L1.5 转置收益必须大于 2× 拷贝成本

- **必须**先估算转置收益，仅当 `gather_gain > 2 × transpose_cost` 时才 host 侧转置。
- **Why:** 无条件 `.contiguous()` 或 `.movedim(...).contiguous()` 会引入额外 memcpy，可能抵消 gather 收益。
- **How to apply:** 见 §2.2 L2.2 转置收益估算。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分派决策树

1. Host 侧将 `dim` 归一化，并按 `DIM=0/1/2/3` 启动 `tl.constexpr` 特化 kernel。
2. 若 `dim` 为 last-dim 且 `dim_size ≤ UB_LIMIT`：走 **UB 预载 + `tl.gather`**。
3. 若 `dim != last` 且 `dim_size ≤ UB_LIMIT` 且转置收益 > 拷贝成本：先转置到 last-dim，再走 UB 预载 + `tl.gather`，最后转置回。
4. 仅当 UB 路径不可行且 shape 较小时，使用 **direct-indirect fallback**。

#### L2.2 转置收益估算

对非 last-dim gather，转置收益估算：

```text
transpose_cost ≈ 2 * tensor.numel() * elem_bytes / copy_bw
gather_gain    ≈ saved_strided_loads * memory_latency
仅当 gather_gain > 2 * transpose_cost 时转置
```

### §2.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `gather_ub_kernel`（UB 预载 + `tl.gather`）

适用条件：`dim` 对应 last-dim，且 `dim_size ≤ UB_LIMIT`。

```python
@triton.jit
def gather_ub_kernel(
    x_ptr, idx_ptr, out_ptr,
    M, K, T,
    UB_LIMIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_m >= M or pid_k >= K:
        return

    # UB 预载一行输入（last-dim）
    row = tl.load(x_ptr + pid_m * T + tl.arange(0, UB_LIMIT), mask=tl.arange(0, UB_LIMIT) < T)
    idx = tl.load(idx_ptr + pid_m * K + pid_k).to(tl.int32)

    # 沿 trailing 维度分块 gather
    for t_start in range(0, T, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        mask = t_offs < T
        gathered = tl.gather(row, idx * T + t_offs, mask=mask)
        tl.store(out_ptr + (pid_m * K + pid_k) * T + t_offs, gathered, mask=mask)
```

> 非 last-dim 场景：先评估转置收益，必要时 host 侧转置到 last-dim 后再调用本模板。

### §2.4 Index / Gather 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 参考 Level1 索引计算类算子 | 18_Index / 20_Gather | ≥0.6x | 旧版本几何平均加速比 vs torch；新版本若丢失 UB gather/维度特化普遍跌至 0.05x~0.4x |

**关键结论**:
1. last-dim gather 必须走 UB 预载 + `tl.gather`，direct-indirect 仅作为 fallback。
2. `ndim ≤ 4` 时必须编译期维度特化，禁止运行时 ND 坐标解码。
3. 循环边界必须基于输出/索引张量 shape，而非输入 shape。

---
