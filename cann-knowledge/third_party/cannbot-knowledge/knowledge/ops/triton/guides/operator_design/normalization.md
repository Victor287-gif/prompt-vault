---
type: Guide
title: Normalization 归一化算子
description: Triton Ascend 归一化算子（RMSNorm/LayerNorm/GroupNorm）的融合模式、列分块策略与数值稳定性优化。
tags: [task.design, topic.compute.normalization, topic.compute.composite, topic.data_movement.memory_access, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/reduce.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/05-layer-norm.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# Normalization 归一化算子

归一化是 Transformer 架构中除 Attention/GEMM 外的第三大计算类别。在 910B 上，核心优化在于融合（Skip Add、RoPE）和列分块（超 UB 容量）。

## 检测标准

- 含 `mean/var/rms + normalize` 计算
- PyTorch：`F.layer_norm`/`F.rms_norm`/`F.group_norm`
- 归约沿最后 1-N 维

## 经验路由

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| Skip Add + RMSNorm | exp_fused_rms_norm_001 | 35% | 原地 residual 更新同时计算方差 |
| Skip + LayerNorm | exp_fused_add_layernorm_001 | 30% | Skip+LN 融合单核 |
| RMSNorm + RoPE | exp_fused_norm_rope_001 | 35% | Norm+RoPE 融合 |
| QKV + Norm + RoPE | exp_fused_qkv_rmsnorm_rope_001 | 40% | QKV+Norm+RoPE 三合一 |
| Gemma 风格 | exp_gemma_rmsnorm_001 | 35% | (1+weight) 融合归一化 |
| GroupNorm | exp_groupnorm_001 | 25% | 分组归一化 |
| Q/K 分别 Norm | exp_fused_split_qk_norm_001 | 35% | QK RMSNorm 单核融合 |

## 关键技术

- **Column Blocking**：hidden_dim 超 UB 时分列处理，两次 pass（mean_sq → normalize）
- **Skip Add 融合**：`x + residual` 与 Norm 在单 kernel 完成，避免中间 HBM 写
- **Persistent Kernel**：grid-stride loop，rows 远大于核数时固定核数
- **fp32 精度**：mean_sq 用 fp32 累加，`tl.rsqrt` 优于 `1/tl.sqrt`

