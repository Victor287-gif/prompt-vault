---
type: Optimization
title: 激活函数优化经验聚合
description: 从 14 条激活函数 YAML 经验中蒸馏的核心优化模式：SiLU/GELU/SwiGLU 向量化与激活+量化融合。
tags: [task.optimization, topic.compute.elementwise.activation, topic.programming_model.execution_unit.vector, topic.compute.composite, topic.compute.quantization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/sigmoid.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/exp.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 激活函数优化经验聚合

从 14 条激活函数 YAML 经验中蒸馏。激活函数天然适合向量化——逐元素独立，无跨元素依赖。

## 共性模式

1. **向量化**：BLOCK_SIZE ∈ {256,512,1024,2048,4096}，大张量优先 1024/2048
2. **融合优先**：紧跟 matmul 时内联进 GEMM kernel，消除中间 HBM 读写
3. **fp32 精度**：SwiGLU 等含 sigmoid 的激活在 fp32 中计算以保证精度

## 高质量经验汇总

| 经验 ID | 提升 | 核心优化 | 适用条件 |
|---------|------|----------|----------|
| exp_silu_fwd_001 | 25% | SiLU 前向向量化 | SiLU 激活 |
| exp_gelu_fwd_001 | 25% | GELU 前向向量化 | GELU 激活 |
| exp_swiglu_fwd_001 | 30% | SwiGLU 门控激活前向 | SwiGLU |
| exp_swiglu_quant_001 | 35% | SwiGLU + 量化单核融合 | 量化推理 |
| exp_fused_gating_001 | 30% | gating + 激活融合 | 门控激活融合 |

