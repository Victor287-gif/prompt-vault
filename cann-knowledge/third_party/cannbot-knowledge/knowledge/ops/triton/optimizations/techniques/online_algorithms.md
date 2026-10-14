---
type: Optimization
title: 在线算法
description: Triton Ascend 在线算法：Online Softmax（running max/sum 单 pass）、Online RMSNorm（running
  mean_sq）与 Welford 方差。
tags: [task.optimization, topic.compute.normalization, topic.precision, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 在线算法

在线算法通过维护 running statistics，将原本需要两趟（或多趟）遍历的计算压缩为单趟，消除中间 HBM 读写。

## OPT-triton-online-001: Online Softmax

标准 Softmax 需两趟：一趟求 max，一趟求 exp+sum。Online Softmax 维护 running max/sum 单趟完成：

```
m_new = max(m_old, x_tile_max)
l_new = l_old * exp(m_old - m_new) + sum(exp(x_tile - m_new))
```

最终输出：`softmax(x)_i = exp(x_i - m_final) / l_final`

适用：Flash Attention、CrossEntropy Loss 融合。

## OPT-triton-online-002: Online RMSNorm

维护 running mean_sq，单趟完成归一化（需两趟时第一趟求 mean_sq，第二趟 normalize）。

## OPT-triton-online-003: Welford 方差

Welford 在线算法计算均值和方差：
```
delta = x - mean_old
mean_new = mean_old + delta / n
M2_new = M2_old + delta * (x - mean_new)
```

