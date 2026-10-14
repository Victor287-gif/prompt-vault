---
type: Guide
title: SSM / 状态空间模型算子
description: Triton Ascend SSM（Mamba/GLA/RetNet/Delta Rule）的 Chunkwise 分解、状态寄存器复用与反向标量消除优化。
tags: [task.design, topic.compute.composite, topic.programming_model, topic.programming_model.execution_unit.scalar, topic.data_movement]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# SSM / 状态空间模型算子

SSM 以线性递推方式更新隐状态 `h_t = A·h_{t-1} + B·x_t`，复杂度 O(T) 而非 O(T²)。核心挑战：递推依赖（时间步间有数据依赖）、Chunkwise 分解（块内并行 GEMM + 块间串行状态转移）、State 寄存器复用。

## 检测标准

- 线性递推 `h_t = A·h_{t-1} + B·x_t`
- Chunkwise 线性注意力：`O = (Q·K^T ⊙ M)·V`
- Delta Rule 更新：`h_new = h - k^T·h·v^T + k^T·v`
- 模型：Mamba/Mamba2, GLA, RetNet, Hyena

## Chunkwise 经验路由

| 场景 | 经验 ID | 关键技术 |
|------|---------|----------|
| Chunkwise 前向 | exp_chunk_fwd_001 | block_ptr tiling + gating masking |
| Chunk CumSum | exp_chunk_cumsum_001 | 块内累积和 (gating 因子) |
| Gated Delta | exp_chunk_gated_delta_001 | sigmoid gating + 状态更新 |
| KKT 计算 | exp_chunk_kkt_001 | K·K^T 内积（状态压缩） |
| Delta Rule (NPU) | exp_fused_delta_rule_npu_001 | sigmoid+softplus+delta rule 三合一 |

## KDA 反向优化

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 标量消除 | exp_kda_bwd_wy_scalar_opt_001 | 3.33x | i64→i32、i32→fp32、cdiv/min 消除 |
| 结构优化 | exp_kda_bwd_wy_structure_opt_001 | 30% | Phase split、barrier 外提、LICM |
| Intra 标量消除 | exp_kda_bwd_intra_scalar_opt_001 | 50% | block_ptr→手动指针、fp32 mask |
| 标量循环 → tl.dot | exp_kda_bwd_scalar_loop_to_dot_001 | 40% | 对角块循环改矩阵乘 |

## 关键技术

- **Chunkwise 分解**：block 内并行 GEMM + block 间串行状态传递
- **State 寄存器复用**：隐状态 `h` 全程驻留寄存器
- **标量消除**：i64→i32 地址、i32→fp32 比较、cdiv/min 消除

