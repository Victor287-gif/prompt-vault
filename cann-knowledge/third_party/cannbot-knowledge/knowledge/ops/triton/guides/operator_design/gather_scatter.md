---
type: Guide
title: Gather/Scatter 算子
description: Triton Ascend Gather/Scatter 与 MoE 路由的不规则内存访问优化：Binned Dispatch、原子冲突规避与负载均衡。
tags: [task.design, topic.compute.indexing, topic.implementation.atomic, topic.parallelism]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Gather/Scatter 算子

Gather（`output[i] = input[index[i]]`）和 Scatter（`output[index[i]] = input[i]`）通过索引进行不规则内存访问，是 MoE 路由、Token 排列、嵌入查找的核心原语。

## 检测标准

- Triton: `tl.gather`/`tl.scatter`/手动 `ptr + index[i]`
- PyTorch: `torch.gather`/`torch.scatter`/`torch.index_select`/`Tensor[index]`
- MoE top-k 专家选择+token 分发/收集
- Token 排列/反排列
- 嵌入表查找 `nn.Embedding`

## 经验路由

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 分散 scatter 原子冲突 | exp_gs_mem_001 | 25% | 寄存器局部聚合，单次 atomic write |
| 小粒度 scatter | exp_gs_mem_002 | 30% | BLOCK_SIZE 元素向量操作 |
| 不规则索引 | exp_gs_mem_004 | 25% | Block pointer 索引访问 |
| 索引排序改善局部性 | exp_gs_mem_005 | 30% | scatter 前排序索引 |
| MoE Binned Dispatch | exp_gs_bin_003 | 35% | 按 expert 分 bin 逐 bin 处理 |
| Token 排列 | exp_gs_bin_004 | 25% | shuffle token 排列 |
| Expert 负载均衡 | exp_gs_bin_005 | 30% | work-stealing |

## 关键技术

- **Binned Dispatch**：按 expert ID 排序后连续写入，减少随机 scatter
- **原子冲突规避**：寄存器局部聚合后单次 atomic write
- **预计算偏移**：host 侧预计算 slot，kernel 只做写入
- **负载均衡**：MoE 中不同 expert 的 token 数量不均时用 work-stealing

