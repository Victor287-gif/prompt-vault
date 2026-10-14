---
type: Optimization
title: 杂项优化经验聚合
description: 从 40 条杂项融合 + 17 条 Conv + 12 条 Loss + 3 条 Embedding + 2 条 Elementwise
  + 6 条 solve_tril YAML 经验中蒸馏的核心模式。
tags: [task.optimization, topic.compute.composite, topic.compute.neighborhood.convolution, topic.compute.indexing, topic.compute.elementwise.arithmetic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 杂项优化经验聚合

从小类别 YAML 经验中蒸馏的共性模式。涵盖卷积、损失函数、嵌入查找、三角求解等。

## Causal Conv1D（17 条）

| 经验 ID | 提升 | 核心优化 |
|---------|------|----------|
| exp_causal_conv1d_prefill_002 | — | 滑动窗口寄存器复用：col0-3 移位，每 token 仅 1 load |
| exp_causal_conv1d_prefill_003 | 1.60x | 2D Grid + BTD 布局并行 |
| exp_causal_conv1d_prefill_006 | — | Conv+Bias+SiLU 三合一融合 |
| exp_causal_conv1d_update_002 | 3.1x | W=4 循环展开 + D 阈值控制 |

## Loss 函数（12 条）

| 经验 ID | 核心优化 |
|---------|----------|
| exp_celoss_fwd_001 | Online log-softmax + NLL |
| exp_celoss_bwd_001 | `softmax - one_hot(label)` 单步 grad |
| exp_ls_celoss_001 | Label-Smoothed CE: (1-ε)·one_hot + ε/V uniform |

## Embedding（3 条）

| 经验 ID | 核心优化 |
|---------|----------|
| exp_embedding_001 | 2D tile (M×N) + 软件流水线 |
| exp_embedding_backward_001 | scatter-add 原子操作 + 索引排序 |

## solve_tril（6 条）

| 经验 ID | 提升 | 核心优化 |
|---------|------|----------|
| exp_solve_tril_001 | 1.33-1.87x | 结构零元 mask 消除 |
| exp_solve_tril_002 | — | 寄存器预加载 + extract_slice 替代 HBM load |
| exp_solve_tril_003 | — | Constexpr 双路径循环展开 |
| exp_solve_tril_004 | — | BT=16/32/64 三层分级调度 |

