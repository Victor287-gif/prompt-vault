---
type: Optimization
title: MatMul/GEMM 优化经验聚合
description: 从 38 条 MatMul YAML 经验中蒸馏的核心优化模式：L0 Tiling、对角 Core Mapping、NZ Layout、融合变体与硬件特定优化。
tags: [task.optimization, topic.compute.composite, topic.programming_model.execution_unit.cube, topic.tiling, topic.memory]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# MatMul/GEMM 优化经验聚合

从 38 条 MatMul YAML 经验中蒸馏的核心优化模式。按提升幅度降序排列。

## 共性模式

所有 MatMul 优化共享三条基石：
1. **L0 级 Tiling**：BLOCK_M/N∈{128,256}, BLOCK_K=256，对齐 Cube L0 访存粒度
2. **fp32 累加**：中间结果在 L0C 寄存器中以 fp32 累积，最终 cast 回目标 dtype
3. **Autotune**：GROUP_SIZE∈{8,16,32}，num_stages=2，num_warps=4

## 高质量经验汇总

| 经验 ID | 提升 | 核心优化 | 适用条件 |
|---------|------|----------|----------|
| exp_matmul_001 | 40% | L0 128³ Tiling + L0C 寄存器复用 | M,N≥256, K≥1024, fp16/bf16 |
| exp_matmul_002 | 35% | Cube fixpipe 量化 + 4×K 展开 | M,N≥128, K≥128, fp16, NZ |
| exp_matmul_005 | 32% | M=256,K=256,N=128 双转置专项 | trans_a & trans_b |
| exp_matmul_diag_tiling_001 | 25% | 对角 Tiling 消除 bank conflict | M,N≥512 |
| exp_matmul_mem_001 | 30% | L0C 寄存器复用，避免重复加载 | 所有 matmul |
| exp_matmul_sched_003 | 25-35% | Persistent Kernel grid-stride | n_rows >> num_cores |
| exp_matmul_sched_004 | 30% | dl.async_task Cube/Vector 并行 | 含后处理激活 |
| exp_grouped_matmul_001 | 30% | 分组矩阵乘 work-stealing | MoE grouped GEMM |
| exp_matmul_hw_001 | 30% | grouped_launch_diagonal | 硬件级对角映射 |
| exp_matmul_hw_004 | 35% | Cube/Vector 分离调度 | 异构并行 |

## 融合变体

| 融合 | 提升 | 说明 |
|------|------|------|
| GEMM + GELU/SiLU | 30% | L0C 寄存器上直接施加激活 |
| GEMM + Bias | 15% | fixpipe 后处理 |
| GEMM + RoPE | 25% | 投影后直接旋转 |
| GEMM + RMSNorm | 30% | 归一化融合 |

