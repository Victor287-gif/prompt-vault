---
type: Optimization
title: Gather/Scatter 优化经验聚合
description: 从 13 条 Gather/Scatter/MoE YAML 经验中蒸馏的核心优化模式：Binned Dispatch、原子冲突规避与负载均衡。
tags: [task.optimization, topic.compute.indexing, topic.implementation.atomic, topic.parallelism]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Gather/Scatter 优化经验聚合

从 13 条 Gather/Scatter/MoE YAML 经验中蒸馏。核心挑战：不规则内存访问无法直接向量化，需通过排序、binning 和原子操作优化。

## 共性模式

1. **Binned Dispatch**：按 expert ID 排序后连续写入，减少随机 scatter 开销
2. **寄存器局部聚合**：多核 scatter 时先寄存器聚合再单次 atomic write
3. **Slot 预计算**：host 侧预计算每个 token 的 slot，kernel 只做写入

## 经验汇总

| 经验 ID | 提升 | 核心优化 |
|---------|------|----------|
| exp_gs_bin_003 | 35% | Binned expert dispatch (num_experts × capacity ≥ 1024) |
| exp_gs_mem_002 | 30% | BLOCK_SIZE 元素向量化 scatter |
| exp_gs_mem_005 | 30% | scatter 前索引排序改善局部性 |
| exp_gs_bin_005 | 30% | work-stealing 均衡 expert 负载 |
| exp_moe_binned_scatter_001 | 30% | Bin 分组 scatter |
| exp_gs_mem_001 | 25% | 寄存器局部聚合 + 单次 atomic write |
| exp_gather_001 | 25% | 向量化 gather |
| exp_scatter_001 | 25% | 向量化 scatter |

