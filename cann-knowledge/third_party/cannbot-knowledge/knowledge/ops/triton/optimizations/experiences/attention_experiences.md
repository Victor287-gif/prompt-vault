---
type: Optimization
title: Attention 优化经验聚合
description: 从 47 条 Attention YAML 经验中蒸馏的核心优化模式：Flash Attention、GQA/MQA、Paged KV Cache、Split-K
  与稀疏注意力。
tags: [task.optimization, topic.attention, topic.compute.normalization, topic.data_movement.memory_access, topic.compute.composite]
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
# Attention 优化经验聚合

从 47 条 Attention YAML 经验中蒸馏的核心优化模式。

## 共性模式

所有 Attention 优化共享：
1. **Online Softmax**：维护 running max/sum，单 pass 完成 softmax，O(S) 内存
2. **分块 QK^T**：按 BLOCK_M × BLOCK_N tile 计算注意力分数
3. **fp32 精度**：softmax 中间计算全部升 fp32

## 高质量经验汇总

| 经验 ID | 提升 | 核心优化 | 适用条件 |
|---------|------|----------|----------|
| exp_fused_attention_002 | 50% | Flash Attention: 分块+Online Softmax, O(S) 内存 | seq≥1024, fp16/bf16 |
| exp_flashattn_gqa_001 | 40% | SIMD index select 多头复用 KV | GQA/MQA |
| exp_flashattn_splitk_gqa_001 | 40% | Split-K + 分页 KV | GQA, 大 seq |
| exp_attn_ascend_dual_pipeline_001 | 45% | 双流水线软件流水 | seq≥2048 |
| exp_paged_attn_flash_001 | 35% | Block Pointer 分页注意力 | 推理, 分页 KV |
| exp_sparse_attn_topk_001 | 30% | top-k 稀疏 mask | 稀疏注意力 prefill |
| exp_sdpa_flash_001 | 35% | SDPA + Cube hints | PyTorch SDPA 对齐 |
| exp_attn_fp8_ub_slice_001 | 35% | UB 溢出分片处理 | fp8 输入 |
| fusion-search 3_FusionAttention | 59% | 29 基因编码存档：G023 反向倾斜 + G017 GQA 摊销 + 连续 kernel 结构重构 → 1.5933x；18 基因证伪全表（split-KV/num_stages/batched-rescale 均判否） | npu_fusion_attention 复刻，910b2，61 case（详见 operator playbook §2.5） |

