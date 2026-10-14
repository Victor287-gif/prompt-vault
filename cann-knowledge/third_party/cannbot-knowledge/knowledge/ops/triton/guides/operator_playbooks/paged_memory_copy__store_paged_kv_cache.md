---
type: Guide
title: StorePagedKVCache 算子（Scatter-Copy） 实现手册
description: '典型特征: 按预计算元数据 scatter-copy K/V token 到 paged cache，离散写入 memory-bound'
tags: [task.implementation, topic.data_movement.data_copy, topic.data_movement.memory_access, topic.compute.indexing]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:28:46Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---

# StorePagedKVCache 算子（Scatter-Copy） 实现手册

> 类别约束见 [../paged_memory_copy_constraints.md](../constraints/paged_memory_copy_constraints.md)（L1.1~L1.4 硬约束在此），
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §1 StorePagedKVCache 算子（paged_memory_copy）

**算子类别**: `paged_memory_copy`　**典型特征**: 按预计算元数据 scatter-copy K/V token 到 paged cache，离散写入 memory-bound。
**性能基准**: 12 accuracy cases 全通过；scalar-loop 变体 1.0×（vs 存档基线，目标 1.2× 未达）→ **2D tile CS32 变体 1.80×**（vs on-box 重测 original，目标 1.2× **已达**），Ascend950PR / `ascend910_9382`，CANN profiler kernel-only（active=5）。

> ⚠️ 存档 `基线性能/` 在本机不可复现，一律与 on-box 重测的 original 对比（见类别卡头部）。

### §1.1 Layer 1 设计约束（算子专属）

- **L1.1 ★** 禁止 `tl.static_range(N)` 展开标量 scatter 循环（8-way → 10× 退化）；标量 `off += 1` 是编译器最擅长模式，但 950PR 上被 2D tile 取代。
- **L1.2 ★** 禁止 `multibuffer/unit_flag`（+30% TensorMove 反优化）。
- **L1.3** 手动预计算 base pointer 零收益（编译器已自动外提）。
- **L1.4** grid 保守：`max(1, min(num_cores, num_chunks))`；num_chunks ≤ 8 用 `grid=(1,)`。

### §1.2 Layer 2 算法骨架

**1D grid strided 骨架**（每个 work item = `(item, head)` 对，标量 token 循环 + head_dim 向量化）：

```python
@triton.jit
def scatter_copy_kernel(src_ptr, dst_ptr, meta_ptr, num_items,
                        stride_src_tok, stride_src_head, stride_src_dim,
                        stride_dst_blk, stride_dst_head, stride_dst_tok, stride_dst_dim,
                        head_dim: tl.constexpr, num_heads):
    pid = tl.program_id(0); num_programs = tl.num_programs(0)
    idx = pid
    while idx < num_items:
        item_idx = idx // num_heads; head_idx = idx % num_heads
        row = meta_ptr + item_idx * stride_meta_row
        src_start  = tl.load(row + 0); dst_block = tl.load(row + 1)
        dst_offset = tl.load(row + 2); item_len  = tl.load(row + 3)
        off = 0
        while off < item_len:
            src_tok = src_start + off; dst_tok = dst_offset + off
            dim_idx = tl.arange(0, head_dim)
            src_addr = src_ptr + src_tok*stride_src_tok + head_idx*stride_src_head + dim_idx*stride_src_dim
            dst_addr = dst_ptr + dst_block*stride_dst_blk + head_idx*stride_dst_head + dst_tok*stride_dst_tok + dim_idx*stride_dst_dim
            tl.store(dst_addr, tl.load(src_addr))
            off += 1
        idx += num_programs
```

**2D tile `[CHUNK_SIZE, head_dim]` CS32 变体（Ascend950PR 最优，1.80×）**：

- Work item = `(logical_chunk, head)`；1D grid `min(num_vectorcore, num_items)`，strided `while`。
- 每 work item：scalar-load 4 个 chunk_metadata ints → token 维 tiled `[CHUNK_SIZE, head_dim]` 2D masked load/store，动态 `while off < chunk_len: off += CHUNK_SIZE`。
- **K 与 V 在同一 program 内共享同一 tile / 索引算术**。

### §1.3 Layer 3 关键技巧（可参考，不可照抄）

- **已验证变体对比**：

  | 变体 | 实测 | 结论 |
  |------|------|------|
  | 8-way constexpr-unrolled（`tl.static_range(8)` + `if off+t < len`） | 0.15×（20.9µs vs 2.4µs） | **禁止**（L1.1） |
  | compiler hints（`multibuffer/unit_flag`） | 0.9×（+30% TensorMove） | **禁止**（L1.2） |
  | 标量循环 + head_dim 向量化 | vs 存档基线 1.0×；**on-box 实测灾难性慢**（163-token case 14.7µs vs tile 版 2.3µs，0.16×） | 950PR 被 2D tile 取代 |
  | **2D tile `[CHUNK_SIZE, head_dim]` CS32** | **1.80×**（vs on-box 重测 original） | **当前最优** |

- **Tile 宽度 sweep（CHUNK_SIZE=32 最佳固定折中）**：sweep `{8,16,32,64,128}`，取 `chunk_size = min(block_size, 32)`（保持 power-of-two 配合 `tl.arange`）：
  - 128/64：长 prefill case 尚可（≈1.0-1.1×），**短/decode chunk（chunk_len≈1）浪费宽向量 → 0.80-0.90×**。
  - 8/16：短 chunk 表现好，**长 chunk 崩溃（0.48-0.64×）**。
  - 32：长 case 1.09-1.10×，短 case ≈0.90×，**geomean 最佳**。
- **大 prefill case 驱动 geomean**：tiled kernel 在这些 case 上比 on-box original 快 **4.3-8.5×**；small/decode case 上 original 略快（0.79-1.08×，绝对差 ~0.3µs）。

### §1.4 性能基准 / 优化记录表（Ascend950PR / `ascend910_9382`）

| 方案 | 几何平均 | 口径 | 目标 1.2× |
|------|---------|------|-----------|
| 2026-07-29 scalar-loop 变体 | **1.0×** | vs 存档 `基线性能/`（不可复现） | 未达 |
| **2026-07-30 2D tile `[CHUNK_SIZE, head_dim]` CS32** | **1.80×** | vs on-box 重测 original（CANN profiler, kernel-only, active=5） | **已达** |

- 12 accuracy cases 全通过；大 prefill case 4.3-8.5×，小/decode case 0.79-1.08×（绝对差 ~0.3µs）。

### §1.5 陷阱 / 证伪方向（不要重跑）

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Constexpr-unrolling scatter 循环 | 地址计算复制 → 寄存器压力 | L1.1（10× 退化） |
| scatter-copy 上加 compiler hints | multibuffer 额外 buffer copy | L1.2（+30% 开销） |
| 手动预计算 base pointers | 编译器已自动外提 | L1.3（零收益） |
| 小 workload 过度并行 | launch 开销主导 | L1.4 用 `grid=(1,)` |
| 950PR 用 per-token 标量循环 | 比向量化 tile 慢 6× | 改用 2D tile |
| Tile 宽度选错 | 128/64 短块浪费宽向量；8/16 长块崩溃 | sweep 取 `min(block_size, 32)` |
| 采信存档 `基线性能/` 数字 | 采集自更快/不同硬件 | 与 on-box 重测 original 对比 |
