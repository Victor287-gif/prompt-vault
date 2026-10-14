---
type: Optimization
title: 归一化优化经验聚合
description: 从 33 条 Normalization YAML 经验中蒸馏的核心优化模式：RMSNorm/LayerNorm/GroupNorm 的融合、列分块与
  Persistent Kernel 策略。
tags: [task.optimization, topic.compute.normalization, topic.compute.composite, topic.data_movement.memory_access, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/reduce.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/05-layer-norm.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 归一化优化经验聚合

从 33 条 Normalization YAML 经验中蒸馏。归一化是 Transformer 中除 Attention/GEMM 外的第三大计算类别。

## 共性模式

1. **Column Blocking**：hidden_dim 超 UB 时分列处理，两次 pass（mean_sq → normalize）
2. **融合优先**：Skip Add / RoPE / QKV Split 与 Norm 在单 kernel 完成
3. **Persistent Kernel**：grid-stride loop，rows 远大于核数时固定 grid
4. **fp32 精度**：mean_sq 用 fp32 累加，`tl.rsqrt` 优于 `1/tl.sqrt`

## 高质量经验汇总

| 经验 ID | 提升 | 核心优化 | 适用条件 |
|---------|------|----------|----------|
| exp_fused_rms_norm_001 | 35% | 原地 residual 更新同时计算方差 | Skip Add + RMSNorm |
| exp_fused_add_layernorm_001 | 30% | Skip + LN 融合单核 | Skip + LayerNorm |
| exp_fused_norm_rope_001 | 35% | Norm + RoPE 融合 | RMSNorm 后接 RoPE |
| exp_fused_qkv_rmsnorm_rope_001 | 40% | QKV+Norm+RoPE 三合一 | Attention QKV 投影 |
| exp_gemma_rmsnorm_001 | 35% | (1+weight) 融合归一化乘法 | Gemma 风格 |
| exp_fused_split_qk_norm_001 | 35% | Q/K RMSNorm 单核融合 | Q/K 分别 Norm |
| exp_groupnorm_001 | 25% | 分组归一化向量化 | GroupNorm |

