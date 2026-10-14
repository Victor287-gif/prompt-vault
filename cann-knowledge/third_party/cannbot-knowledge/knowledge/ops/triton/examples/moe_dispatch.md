---
type: Example
title: MoE Dispatch 代码骨架
description: MoE Token Scatter + Expert GEMM 两阶段代码骨架。适用 MoE 专家路由与 top-k 分发场景。
tags: [task.implementation, topic.compute.indexing, topic.compute.composite]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# MoE Dispatch 代码骨架

> 经验来源：exp_gs_bin_003 Binned dispatch (35%), exp_gs_bin_005 work-stealing (30%)

## 关键设计点

1. **两阶段设计**：Phase 1 Token Scatter（按 expert 分组写入 buffer）→ Phase 2 Expert GEMM（每个 expert 独立矩阵乘）
2. **Binned Dispatch**：按 expert ID 排序后连续写入，减少随机 scatter 开销
3. **Slot 预计算**：host 侧预计算每个 token 在 expert buffer 中的 slot，kernel 只做写入
4. **BLOCK_H tiling**：hidden_dim 维度分块处理，每块独立 scatter

## 核心结构

```python
# Phase 1: Token Scatter
@triton.jit
def token_scatter_kernel(tokens_ptr, expert_buf_ptr, routing_ptr,
                         slot_ptr, num_tokens, hidden_dim, ...):
    pid = tl.program_id(0)
    # 每个 program 处理一个 (token, top_k_slot) 对
    # 按 slot 写入 expert_buf

# Phase 2: Expert GEMM
# 每个 expert 独立调用 GEMM kernel
```

完整骨架见源文件 `skeletons/moe_dispatch.py`。

