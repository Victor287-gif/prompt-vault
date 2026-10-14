---
type: Optimization
title: SSM/KDA 优化经验聚合
description: 从 12 条 SSM YAML 经验中蒸馏的核心优化模式：Chunkwise 分解、State 寄存器复用与反向标量消除（i64→i32,
  block_ptr→手动指针）。
tags: [task.optimization, topic.compute.composite, topic.programming_model, topic.programming_model.execution_unit.scalar, topic.data_movement]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# SSM/KDA 优化经验聚合

从 12 条 SSM YAML 经验中蒸馏。SSM 以线性递推更新隐状态，核心挑战：递推依赖（时间步间有数据依赖）、Chunkwise 分解（块内并行 GEMM + 块间串行状态转移）。

## 共性模式

1. **Chunkwise 分解**：block 内并行 GEMM + block 间串行状态传递
2. **State 寄存器复用**：隐状态 `h` 全程驻留寄存器
3. **反向标量消除**：KDA 反向 kernel 中 i64→i32、i32→fp32、cdiv/min 消除、block_ptr→手动指针

## Chunkwise 前向经验

| 经验 ID | 核心技术 |
|---------|----------|
| exp_chunk_fwd_001 | block_ptr tiling + gating masking |
| exp_chunk_gated_delta_001 | sigmoid gating + 状态更新 |
| exp_fused_delta_rule_npu_001 | sigmoid+softplus+delta rule 三合一 NPU 路径 |

## KDA 反向优化（最大收益）

| 经验 ID | 提升 | 核心优化 |
|---------|------|----------|
| exp_kda_bwd_wy_scalar_opt_001 | 3.33x | i64→i32、i32→fp32 比较、cdiv/min/compile_hint 消除 |
| exp_kda_bwd_intra_scalar_opt_001 | 50% | block_ptr→手动指针、fp32 mask、min→tl.where |
| exp_kda_bwd_scalar_loop_to_dot_001 | 40% | 对角块标量循环 → tl.dot |
| exp_kda_bwd_intra_memory_opt_001 | 35% | 转置 load 合并化、冗余 load 消除 |
| exp_kda_bwd_wy_structure_opt_001 | 30% | Phase split、barrier 外提、LICM |

