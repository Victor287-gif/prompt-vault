---
type: Guide
title: Causal Conv1D 卷积算子
description: Triton Ascend Causal Conv1D 的滑动窗口寄存器复用、2D Grid 调度与 Conv+Bias+SiLU 融合策略。SSM/Mamba
  架构核心层。
tags: [task.design, topic.compute.neighborhood.convolution, topic.compute.composite, topic.data_movement, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Causal Conv1D 卷积算子

Causal Conv1D 用于 SSM/Mamba 架构的序列特征提取：`y_t = Σ w_k · x_{t-k}`（仅依赖当前及过去时间步）。深度卷积（每 channel 独立）、小卷积核（W=2/4）、带宽密集型。

## 检测标准

- 深度卷积（depthwise），每 channel 独立
- 因果掩码（仅过去时间步）
- 通常位于 SSM block 的 input projection 后、gate 前

## 经验路由

| 场景 | 经验 ID | 关键技术 |
|------|---------|----------|
| Prefill 寄存器复用 | exp_causal_conv1d_prefill_002 | col0-3 寄存器移位，每 token 仅 1 次 load（消除 75%） |
| Prefill 2D Grid+BTD | exp_causal_conv1d_prefill_003 | (batch, D-tiles) 并行+BTD 连续访问 |
| Prefill 指针预计算 | exp_causal_conv1d_prefill_004 | 循环不变量外提 |
| Prefill Pad 提前退出 | exp_causal_conv1d_prefill_005 | 填充序列 1 load+cmp 退出 |
| Conv+Bias+SiLU | exp_causal_conv1d_prefill_006 | 三合一融合，bias 寄存器驻留 |
| Decode W=4 展开 | exp_causal_conv1d_update_002 | W_IS_4 编译期分支，D≥1024 时 3.1x |
| Decode block_ptr vs scalar | exp_causal_conv1d_update_003 | 混合 DMA+标量 load |

## 关键技术

- **滑动窗口寄存器复用**：col 值寄存器移位而非重复 load
- **2D Grid 调度**：(batch, D-tiles) 双维度并行
- **编译期特化**：W_IS_4/DIM_IS_MULTIPLE 消除 mask 和循环
- **融合**：Conv+Bias+SiLU 三合一，bias 寄存器驻留

