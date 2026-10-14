---
type: Example
title: RMSNorm 融合代码骨架
description: 支持 Skip Add + RMSNorm 融合的代码骨架。适用 fp16/bf16，任意 batch/seq/hidden 维度。
tags: [task.implementation, topic.compute.normalization, topic.compute.composite]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/unittest/pytest_ut/test_rms_norm.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# RMSNorm 融合代码骨架

> 经验来源：exp_rmsnorm_npu_001 列分块 (35%), exp_skip_layernorm_001 Skip+Norm (30%)

## 关键设计点

1. **Auto BLOCK_SIZE_N**：用 `@triton.heuristics` 自动选整行或最大 4096，不超 UB
2. **Column Blocking**：hidden_dim 超过 UB 容量时分列处理，两次 pass 完成（一次 mean_sq，一次 normalize）
3. **Skip Add 融合**：`HAS_RESIDUAL=True` 时 `x + residual` 与 RMSNorm 在单 kernel 完成，避免中间 HBM 写
4. **Persistent Kernel**：用 grid-stride loop 确保核数固定，减少 launch 开销
5. **fp32 精度**：mean_sq 在 fp32 中累加，`tl.rsqrt` 倒数平方根

## 核心结构

```python
@triton.heuristics({'BLOCK_SIZE_N': lambda args: ...})
@triton.jit
def rmsnorm_kernel(x_ptr, residual_ptr, weight_ptr, out_ptr,
                   residual_out_ptr, M, N, ...,
                   HAS_RESIDUAL: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    # grid-stride over rows → load row + (optional residual add)
    # → compute mean_sq in fp32 → rsqrt → scale by weight → store
```

完整骨架见源文件 `skeletons/rmsnorm_fused.py`。

