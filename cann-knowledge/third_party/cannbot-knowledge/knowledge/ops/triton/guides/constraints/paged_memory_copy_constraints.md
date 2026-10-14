---
type: Guide
title: paged_memory_copy 类算子设计约束（Paged Memory Copy）
description: 按预计算元数据 scatter-copy K/V token 到 paged cache 的 memory-bound 算子：禁止 constexpr-unrolling
  标量 scatter 循环、禁止 compiler hints、2D tile [CHUNK_SIZE, head_dim] CS32 向量化。
tags: [task.design, topic.data_movement.data_copy, topic.data_movement.memory_access, topic.compute.indexing]
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

# paged_memory_copy 类算子设计约束（Paged Memory Copy）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）。

适用范围：**按预计算元数据 scatter-copy K/V token 到 paged cache** 的 memory-bound 算子（锚定 `_store_paged_kv_cache_chunk_kernel`，op19）。
核心优化哲学：与连续搬运（Pad/Split）不同，scatter 使用**预计算索引做离散写入**——优化方向是**减少 launch 开销 + 改善访存模式（2D tile 向量化 + 连续 head_dim 访存）**，而非循环展开。

> ⚠️ **测量基线不可信**：存档 `基线性能/` 数字采集自更快/不同硬件，**在 Ascend950PR 上不可复现**（重测 original 对大 prefill case 比存档慢 4-8×，如 val3 fresh 29.5µs vs stored 3.6µs；geomean stored/fresh = 0.536×）。**必须与 on-box 重测的 original 对比**。

## 本类别的算子级实现手册

- [`operators/paged_memory_copy__store_paged_kv_cache.md`](../operator_playbooks/paged_memory_copy__store_paged_kv_cache.md) — StorePagedKVCache（Scatter-Copy）实现手册：Layer 2 1D grid strided + 2D tile CS32 变体、Layer 3 技巧与性能基准（1.0× → 1.80×）。

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| StorePagedKVCache | `paged_memory_copy` | 按预计算元数据 scatter-copy K/V token 到 paged cache | 2D tile 向量化 + 连续 head_dim 访存 |

- memory-bound scatter 类算子，与连续搬运（Pad/Split）不同：使用**预计算索引**做离散写入。
- 优化方向：减少 launch 开销 + 改善访存模式，**不是**简单循环展开。

## §1 通用经验（跨算子，首次生成必须遵守）

- **scatter 循环永不 unroll**：带 per-token 地址计算的 scatter-copy 循环保持标量循环，禁止 `tl.static_range` 展开（L1.1）。
- **compiler hints 不上 scatter kernel**：`multibuffer/unit_flag` 为计算密集 / 连续搬运设计，scatter 上用会反优化（L1.2）。
- **base pointer 预计算交给编译器**：循环不变量外提编译器已自动做（L1.3）。
- **grid 保守**：小 workload（num_chunks < 10）可能过度并行（L1.4）。

## Layer 1 设计约束（硬性规则，paged_memory_copy 通用）

- **L1.1 ★ 必须 禁止 constexpr-unrolling 标量 scatter 循环**
  - **Why:** 含离散地址计算的 scatter-copy 循环用 `tl.static_range(N)` 展开产生大量代码膨胀与寄存器压力，实测 8-way unrolling **6-10×（约 10×）性能退化**（20.9µs vs 2.4µs）。标量循环（`off += 1`）是编译器最擅长的模式——但 950PR 上应按 L3.1/3.2 改 2D tile。
  - **How to apply:** 不展开；标量 token 循环 + head_dim 向量化，或直接 2D tile（当前最优）。

- **L1.2 ★ 必须 禁止在 scatter-copy kernel 上使用 compiler hints**
  - **Why:** `multibuffer=True` 引发额外 buffer 分配/copy（TensorMove），**实测增加 ~30% 开销**（0.9×）；`unit_flag=True` 无实际收益。hints 为计算密集 / 连续搬运设计，不适用于离散 scatter-copy。
  - **How to apply:** scatter-copy kernel 一律不加 `multibuffer/unit_flag`。

- **L1.3 必须 Precomputed base pointers 无实际收益**
  - **Why:** 编译器已自动执行循环不变量外提，手动预计算 base pointer 与编译器生成代码等效（零可测差异）。
  - **How to apply:** 直接按索引算术写地址，不手动预提。

- **L1.4 必须 Grid 配置需保守**
  - **Why:** `grid = max(1, min(num_cores, num_chunks))` 对小 workload（num_chunks < 10）可能过度并行，launch 开销主导。
  - **How to apply:** 对 num_chunks ≤ 8 的场景考虑 `grid = (1,)` 减少 launch 开销。

## §2 Layer 2 算法骨架要点

1D grid、strided program loop over `(item, head)` 对：

- `idx = pid; while idx < num_items: item_idx = idx // num_heads; head_idx = idx % num_heads`。
- 每 item：scalar-load 4 个 chunk_metadata ints（src_start / dst_block / dst_offset / item_len）。
- 标量 token 循环 `while off < item_len: off += 1` + head_dim 向量化 `tl.arange(0, head_dim)`（coalesced load/store）。
- 终态（950PR 最优）见 playbook L3.1：`[CHUNK_SIZE, head_dim]` 2D tile。

## §3 陷阱表 / 证伪方向

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Constexpr-unrolling scatter 循环 | 地址计算复制导致寄存器压力 | L1.1（10× 退化） |
| scatter-copy 上用 compiler hints | multibuffer 额外 buffer copy | L1.2（+30% 开销） |
| 手动预计算 base pointers | 编译器已自动外提 | L1.3（零收益） |
| 小 workload 过度并行 | num_items < 10 时 launch 开销主导 | L1.4 考虑 `grid=(1,)` |
| 保留未使用的 constexpr 参数（CHUNK_SIZE/num_subchunks） | 设计残留 | 不针对它优化 |
| 950PR 上用 per-token 标量循环 | 标量 `off += 1` 比向量化 tile 慢 6× | L3.1/3.2 改用 2D tile |
| Tile 宽度选错 | 128/64 短块浪费宽向量；8/16 长块崩溃 | L3.2 sweep 后取 `min(block_size, 32)` |
| 直接采信存档 `基线性能/` 数字 | 存档采集自更快/不同硬件 | 头部 ⚠️：与 on-box 重测的 original 对比 |
