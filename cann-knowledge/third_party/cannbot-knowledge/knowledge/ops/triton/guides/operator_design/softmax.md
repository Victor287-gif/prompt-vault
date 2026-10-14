---
type: Guide
title: Softmax / LogSoftmax 算子
description: Triton Ascend Softmax/LogSoftmax 的 Online 算法、数值稳定性与 Attention 内融合策略。
tags: [task.design, topic.compute.normalization, topic.precision]
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
# Softmax / LogSoftmax 算子

Softmax 将实值向量归一化为概率分布：`softmax(x)_i = exp(x_i) / Σ exp(x_j)`。核心挑战：两趟访存开销（标准实现）、数值溢出（fp16/bf16 `exp(x)`）、跨 tile 归约同步。

## 检测标准

- PyTorch: `F.softmax(x, dim=...)` / `F.log_softmax(x, dim=...)`
- 归约沿指定 dim，输出同 shape
- 不含跨 batch 归约

## 经验路由

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 标准 Softmax (inner dim) | exp_softmax_inner_002 | 30% | Online max+sum 单趟 |
| 非 inner dim | exp_softmax_non_inner_001 | 25% | 外层维度归约，转置处理 |
| Online Softmax (Attention) | exp_softmax_online_001 | 50% | 运行 m/l 迭代更新 |
| LogSoftmax 前向 | exp_log_softmax_001 | 30% | max+sum → log-sum-exp 单核 |
| Softmax 反向 | exp_softmax_backward_001 | 25% | `p * (dy - dot(p, dy))` |

## 关键技术

- **Online Softmax**：维护 running max `m_i` 和 running sum `l_i`，一次遍历完成
  - `m_new = max(m_old, x_tile_max)`
  - `l_new = l_old * exp(m_old - m_new) + sum(exp(x_tile - m_new))`
- **数值稳定**：升 fp32 → 减 max → exp → sum → 归一化
- **反向**：`p * (dy - dot(p, dy))`，利用 softmax 导数性质

