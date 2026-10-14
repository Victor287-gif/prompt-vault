---
type: Guide
title: 归约算子
description: Triton Ascend 归约算子（sum/max/min/logsumexp）的 Split-K 并行、fp32 累加与数值稳定性策略。
tags: [task.design, topic.compute.reduction, topic.precision, topic.implementation.atomic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/sum.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/reduce.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# 归约算子

归约沿指定维度将多个元素合并为更少输出元素。在 910B 上核心挑战：数值精度（fp16 易溢出）、大 K 维度并行度（Split-K）、多遍开销（在线算法压缩为单遍）。

## 检测标准

- Triton: `tl.sum`/`tl.max`/`tl.min`/`tl.reduce`
- PyTorch（含 `dim=`）：`torch.sum(dim=)`/`torch.mean(dim=)`/`torch.max(dim=)`/`torch.logsumexp(dim=)`
- Split-K 并行归约、cross-entropy（含 log-softmax）、cumsum（前缀扫描）

## 经验路由

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| fp16 累加溢出 | exp_reduction_fp32_001 | 30% | fp32 累加器 |
| logsumexp 稳定 | exp_reduction_logsumexp_001 | 35% | log-sum-exp trick（减 max） |
| cross-entropy | exp_reduction_cross_entropy_001 | 30% | 融合 log-softmax + CE |
| argmax+max 联合 | exp_reduction_argmax_001 | 25% | 单遍同时计算 |
| 大 K (≥4096) Split-K | exp_splitk_reduction_001 | 35% | K 维按核划分 + atomic_add |
| 两阶段方案 | exp_splitk_two_pass_001 | 40% | Phase1 部分和，Phase2 全局归约 |
| 直方图 binning | exp_histogram_reduction_001 | 25% | 原子 binning + 寄存器局部聚合 |
| 前缀扫描 cumsum | exp_cumsum_001 | 25% | scan primitive |

## 关键技术

- **fp32 累加器**：归约 sum/mean/var 必须 fp32 累加，最终 cast 回目标 dtype
- **log-sum-exp trick**：先减全局 max 再 exp，防止上溢
- **Split-K**：K≥4096 时按核划分，atomic_add 合并部分和
- **单遍联合归约**：argmax+max、log-softmax+CE 在单 pass 中完成

