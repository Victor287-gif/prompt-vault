---
type: Guide
title: Attention 算子
description: Triton Ascend Attention 算子的 Flash Attention、GQA/MQA、Paged KV Cache 与
  Split-K 优化策略。
tags: [task.design, topic.attention, topic.compute.normalization, topic.compute.composite, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# Attention 算子

Attention 是 Transformer 架构的核心，计算 `softmax(QK^T/√d_k) @ V`。在 910B 上主要通过 Flash Attention 实现 O(S) 内存，消除完整 N×N 注意力矩阵。

## 检测标准

- 计算 `softmax(QK^T / sqrt(d_k)) @ V` 或其变体
- 含 `F.scaled_dot_product_attention`
- 含因果掩码 (causal mask) 或 padding mask

## 经验路由

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 标准 Flash Attention (seq≥1024) | exp_fused_attention_002 | 50% | Online softmax, O(S) 内存 |
| GQA/MQA 多头共享 | exp_flashattn_gqa_001 | 40% | SIMD index select 多头复用 |
| Paged KV Cache（推理） | exp_paged_attn_flash_001 | 35% | Block Pointer 分页注意力 |
| Split-K GQA（大 seq） | exp_flashattn_splitk_gqa_001 | 40% | Split-K + 分页 KV |
| Ascend 双流水线 (seq≥2048) | exp_attn_ascend_dual_pipeline_001 | 45% | 双流水线软件流水 |
| 稀疏 Attention (prefill) | exp_sparse_attn_topk_001 | 30% | top-k 稀疏 mask |
| FP8 输入 | exp_attn_fp8_ub_slice_001 | 35% | UB 溢出分片处理 |

## 关键技术

- **Online Softmax**：维护 running max/sum，单 pass 完成 softmax
- **GQA/MQA 支持**：多 Q head 共享 KV，通过 head mapping 索引
- **Causal Mask**：Q-block 对应 K-block ≤ Q-block 索引
- **Paged KV**：Block Pointer 分页管理 KV cache

