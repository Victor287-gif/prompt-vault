---
type: Example
title: Flash Attention Causal 代码骨架
description: Flash Attention causal mask 变体骨架，支持 GQA/MQA。适用 seq_len≥1024, fp16/bf16。
tags: [task.implementation, topic.attention, topic.compute.normalization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# Flash Attention Causal 代码骨架

> 经验来源：exp_fused_attention_002 (50% boost), exp_flash_attention_gqa_001 (40% boost)

## 关键设计点

1. **Online Softmax**：逐 block 更新 running max 和 running sum，单 pass 完成，O(S) 内存
2. **Causal Mask**：Q-block 对应的 K-block 索引 ≤ Q-block 索引，超出部分 mask 为 -inf
3. **GQA/MQA 支持**：多个 Q head 共享同一 KV head，通过 head 映射索引实现
4. **AutoTune**：`BLOCK_M ∈ {64,128}`, `BLOCK_N ∈ {64,128}`

## 核心结构

```python
@triton.autotune(configs=[...], key=['SEQ_LEN', 'HEAD_DIM'])
@triton.jit
def flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, ...,
    SEQ_LEN, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # QKV block pointer → online softmax 循环 →
    # causal mask 条件 → P*V 累加 → 最终 rescale → store
```

完整骨架见源文件 `skeletons/flash_attention_causal.py`。

