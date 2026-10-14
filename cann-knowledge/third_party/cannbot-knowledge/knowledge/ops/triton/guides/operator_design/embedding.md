---
type: Guide
title: Embedding / 索引拷贝算子
description: Triton Ascend Embedding 查找的 2D Tile 分块、软件流水线与反向 scatter-add 原子更新优化。
tags: [task.design, topic.compute.indexing, topic.implementation.atomic, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Embedding / 索引拷贝算子

Embedding 查找本质是不规则 gather：`y[i] = weight[index[i]]`，反向为 scatter-add：`grad_weight[index[i]] += grad_y[i]`。

## 检测标准

- `nn.Embedding` 前向/反向
- 按索引从权重矩阵 gather 行
- 反向含 scatter-add 梯度累积

## 经验路由

| 场景 | 经验 ID | 关键技术 |
|------|---------|----------|
| Embedding 前向 | exp_embedding_001 | 2D tile (M×N)；软件流水线 num_stages；向量化加载 |
| Embedding 反向 | exp_embedding_backward_001 | scatter-add 原子操作；按 index 排序提升连续性 |
| 索引拷贝 | exp_idx_copy_001 | gather+copy，支持非连续索引批量拷贝 |

## 关键技术

- **2D Tiling**：沿 (token, embedding_dim) 双维度 tile，最大化向量化
- **软件流水线**：多 stage 预取（num_stages≥2）掩盖访存延迟
- **反向原子加**：scatter-add 先按 index 排序再写入，减少原子冲突

