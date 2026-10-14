---
type: Guide
title: Loss 函数算子
description: Triton Ascend Loss 函数（CrossEntropy/Label-Smoothed CE/GRPO/Entropy）的前向-反向融合与
  Online Softmax+Loss 融合策略。
tags: [task.design, topic.compute.reduction, topic.compute.normalization, topic.precision, topic.compute.composite]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Loss 函数算子

Loss 函数将 logits 与 labels 对比计算标量损失及梯度。核心优化机会：前向-反向融合（消除激活重存）、Online Softmax+Loss 融合（GEMM 输出不写 HBM）、数值稳定化。

## 检测标准

- PyTorch: `F.cross_entropy`/`F.nll_loss`/`nn.CrossEntropyLoss`
- 输出为标量（或 per-sample）损失值
- 含对 vocab_size 或 class 维度的归约

## 经验路由

| 场景 | 经验 ID | 关键技术 |
|------|---------|----------|
| CE Loss 前向 | exp_celoss_fwd_001 | online log-softmax + NLL |
| CE Loss 反向 | exp_celoss_bwd_001 | softmax - one_hot(label) 单步 grad |
| Label-Smoothed CE | exp_ls_celoss_001 | (1-ε)·one_hot + ε/V uniform |
| GRPO Loss | exp_grpo_loss_001 | 含 LogSumExp 归约的策略梯度 |
| Entropy Loss | exp_entropy_loss_001 | H = -Σ p·log(p) KL 散度前驱 |

## 关键技术

- **前向-反向融合**：CE Loss 前向和 grad 在同 kernel 计算，消除激活重存
- **Online Softmax + Loss**：GEMM 输出不写 HBM，直接在线计算
- **log-sum-exp trick**：先减 max 再 exp，防止上溢

