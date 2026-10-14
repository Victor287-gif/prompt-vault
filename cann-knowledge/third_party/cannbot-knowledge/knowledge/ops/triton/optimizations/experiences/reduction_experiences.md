---
type: Optimization
title: 归约优化经验聚合
description: 从 4 条 Reduction YAML 经验中蒸馏的核心优化模式：Split-K 并行归约、fp32 累加器与数值稳定性。
tags: [task.optimization, topic.compute.reduction, topic.precision, topic.implementation.atomic, topic.parallelism]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/sum.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/reduce.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 归约优化经验聚合

从 4 条 Reduction YAML 经验中蒸馏。归约优化的核心：fp32 累加防止溢出、Split-K 大维度并行、联合归约单遍完成。

## 共性模式

1. **fp32 累加器**：归约 sum/mean/var 必须 fp32，最终 cast 回目标 dtype
2. **单遍联合归约**：argmax+max、log-softmax+CE 在单 pass 完成
3. **Split-K**：K≥4096 时按核划分，atomic_add 合并部分和

## 经验汇总

| 经验 ID | 提升 | 核心优化 | 适用条件 |
|---------|------|----------|----------|
| exp_reduce_split_001 | 35% | K 维按核划分 + atomic_add 归约 | K≥4096 |
| exp_backward_sum_001 | 25% | 反向 sum 优化 | 反向传播 sum 归约 |
| exp_sum_scale_001 | 20% | sum + scale 融合 | sum 后缩放 |
| exp_reduction_fp32_001 | 30% | fp32 累加器 | fp16/bf16 输入 |

