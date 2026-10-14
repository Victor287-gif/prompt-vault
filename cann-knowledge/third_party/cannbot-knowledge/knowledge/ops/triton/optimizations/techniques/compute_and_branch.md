---
type: Optimization
title: 计算与分支优化
description: Triton Ascend 计算与分支优化：constexpr 编译期特化、tl.where 替代分支、tl.cdiv 消除与 tl.min/tl.maximum
  选择。
tags: [task.optimization, topic.compilation, topic.programming_model.execution_unit.scalar, topic.programming_model, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 计算与分支优化

Triton kernel 不支持 `return`/`break`/`continue`。所有控制流必须通过 mask 或 constexpr 编译期特化实现。

## OPT-triton-comp-001: constexpr 编译期特化

`tl.constexpr` 参数在 JIT 编译期求值，启用死代码消除：
```python
@triton.jit
def kernel(..., HAS_BIAS: tl.constexpr, IS_NEOX_STYLE: tl.constexpr):
    if HAS_BIAS:       # 编译期分支，零运行时开销
        ...
```
适用场景：`HAS_BIAS`/`HAS_RESIDUAL`/`IS_VARLEN`/`IS_NEOX_STYLE`/`W_IS_4`/`DIM_IS_MULTIPLE`

## OPT-triton-comp-002: tl.where 替代条件分支

```python
# 替代 if-else
result = tl.where(condition_mask, value_if_true, value_if_false)
```

## OPT-triton-comp-003: tl.cdiv 消除

`tl.cdiv` 产生运行时除法 → 用编译期常量替代：
- ❌ `n_blocks = tl.cdiv(N, BLOCK_SIZE)`
- ✅ 已知 N 和 BLOCK_SIZE 均为 constexpr 时，编译器直接求值

## OPT-triton-comp-004: min/maximum 选择

```python
# tl.minimum/tl.maximum 在 Vector 单元执行
safe_offset = tl.maximum(offset, 0)
denom = tl.maximum(denom, eps)
```

