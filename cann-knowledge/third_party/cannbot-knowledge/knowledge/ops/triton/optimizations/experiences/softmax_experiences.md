---
type: Optimization
title: Softmax 优化经验聚合
description: 从 7 条 Softmax YAML 经验中蒸馏的核心优化模式：Online Softmax 单趟算法、数值稳定化与前反向融合。
tags: [task.optimization, topic.compute.normalization, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Softmax 优化经验聚合

从 7 条 Softmax YAML 经验中蒸馏。核心：Online Softmax 将两趟降为单趟，数值稳定化防止 exp 溢出。

## 共性模式

1. **Online Softmax**：维护 running max/sum，单 pass 完成
2. **升 fp32**：exp 前升精度 → 减 max → exp → sum
3. **反向利用导数性质**：`p * (dy - dot(p, dy))`

## 经验汇总

| 经验 ID | 提升 | 核心优化 | 适用条件 |
|---------|------|----------|----------|
| exp_softmax_online_001 | 50% | 运行 m/l 迭代更新（Attention 内融合） | Flash Attention |
| exp_softmax_inner_002 | 30% | Online max+sum 单趟 | inner dim softmax |
| exp_log_softmax_001 | 30% | max+sum → log-sum-exp 单核 | LogSoftmax 前向 |
| exp_softmax_non_inner_001 | 25% | 外层维度归约，转置处理 | 非 inner dim |
| exp_softmax_backward_001 | 25% | `p * (dy - dot(p, dy))` | inner dim 反向 |
| exp_log_softmax_backward_001 | 25% | `dy - softmax(x) * sum(dy)` | LogSoftmax 反向 |

