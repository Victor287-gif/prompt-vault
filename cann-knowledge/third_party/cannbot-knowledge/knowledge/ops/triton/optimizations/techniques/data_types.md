---
type: Optimization
title: 数据类型优化：避免 Scalar Fallback
description: Triton Ascend 数据类型选择与 Scalar Fallback 诊断：int64/int32→fp32 转换、混合精度策略与
  profiling 指标解读。
tags: [task.optimization, topic.programming_model.execution_unit.scalar, topic.programming_model.execution_unit.vector, topic.performance.analysis]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Creation_Ops/cast.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# 数据类型优化：避免 Scalar Fallback

910B Vector 单元仅支持特定数据类型。使用不支持的类型会导致编译器从 Vector 执行回退到 Scalar 执行——性能严重退化。

## 导致 Scalar Fallback 的操作

| 操作 | 不支持类型 | 影响 |
|------|-----------|------|
| Vector ADD | int64 | 完全 Scalar |
| Vector CMP | int64, int32 | 完全 Scalar |
| Vector MUL | int64 | 完全 Scalar |
| Vector DIV | int64, int32 | 完全 Scalar |

**关键陷阱**：`tl.arange(0, N)` 默认产生 int64。任何直接从 `tl.arange` 派生的 mask 计算都会使用 int64 比较。

## Profiling 指标

| aiv_scalar_ratio | 解读 |
|-----------------|------|
| <5% | 正常 |
| 5-10% | 需关注 |
| >10% | 显著回退，必须修复 |
| >30% | 严重，紧急修复 |

## OPT-triton-dtype-001: Scalar Fallback 修复

```python
# 修复：arange → fp32
cols = tl.arange(0, BLOCK_N).to(tl.float32)
mask = cols < N  # fp32 CMP → Vector 单元
```

## OPT-triton-dtype-002: 混合精度累加

- 归约（sum/mean/var）必须 fp32 累加器
- fp16/bf16 输入 → fp32 中间计算 → 最终 cast 回
- `tl.rsqrt` 优于 `1/tl.sqrt`

