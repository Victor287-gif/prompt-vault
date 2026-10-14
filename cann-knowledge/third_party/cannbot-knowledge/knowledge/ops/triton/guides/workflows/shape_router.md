---
type: Guide
title: 形状驱动算子决策树
description: 从 PyTorch 算子代码自动路由到对应 ops/techniques/experiences 的决策树。覆盖 MatMul、Attention、Normalization、Reduction、Gather/Scatter、Elementwise
  六类。
tags: [task.implementation, topic.compute.composite]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 形状驱动算子决策树

> 拿到一个 PyTorch 算子但不确定走哪个 ops 页时，按此决策树逐级判断。

## 第一步：识别计算结构

```
输入：PyTorch 算子代码 / 原子分解列表 / 算子描述

├── 含 mm/bmm/matmul/linear/einsum(ik,kj→ij) ?
│   └── YES → MatMul 路径 → ops/matmul
│
├── 含 softmax(QK^T) @ V 结构 ?
│   └── YES → Attention 路径 → ops/attention
│
├── 含 mean/var/rms 后接归一化 ?
│   └── YES → Normalization 路径 → ops/normalization
│
├── 含 sum/max/min(dim=)/logsumexp ?
│   └── YES → Reduction 路径 → ops/reduction
│
├── 含 tensor[indices] 或 out[indices] = val ?
│   └── YES → Gather/Scatter 路径 → ops/gather_scatter
│
└── 输入输出同形状，无跨元素依赖 ?
    └── YES → Elementwise 路径 → ops/elementwise
```

## 第二步：按形状/属性细分

### MatMul 路径
- `trans_a=True, trans_b=True` → 双转置专项
- M,N≥512 → diagonal core mapping
- M,N,K 任一 <128 → micro-tiling
- 后接激活/RMSNorm/RoPE → fusion

### Attention 路径
- seq≥1024 → Flash Attention
- GQA/MQA → 多头共享 KV 优化
- 推理分页 KV → Paged Attention
- causal mask → Causal Flash Attention

### Normalization 路径
- 有 Skip Connection → Skip+Norm 融合
- hidden_dim 超 UB → 列分块
- 后接 RoPE → Norm+RoPE 融合

### Reduction 路径
- K≥4096 → Split 归约
- 非末轴 → 二分折叠树

### Gather/Scatter 路径
- MoE 路由 → binned dispatch/scatter
- 大词表 → 向量化 gather

### Elementwise 路径
- SiLU/GELU/SwiGLU → 对应激活优化
- 含广播 → 广播路径
- 多输入融合 → 多路拼接

