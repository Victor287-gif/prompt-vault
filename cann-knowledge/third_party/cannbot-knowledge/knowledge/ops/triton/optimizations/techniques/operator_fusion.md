---
type: Optimization
title: 算子融合策略
description: Triton Ascend 算子融合决策框架：融合收益分析、Cube/Vector 并行融合模式与 constexpr 条件编译门控。
tags: [task.optimization, topic.compute.composite, topic.implementation.pipeline, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 算子融合策略

融合消除连续算子间的中间 HBM 读写。对内存密集型算子（elementwise, normalization），HBM 往返是主要成本。融合将数据保留在 UB（192KB, 片上, 零 HBM 开销）。

## 融合收益分析

分离执行：GEMM (load A,B→compute→store C) + RMSNorm (load C→compute→store Y) = 3 次 HBM 操作

融合执行：GEMM (load A,B→compute C)→**C 留 UB**→RMSNorm→store Y = 1 次 HBM 操作（消除 66% HBM 流量）

## OPT-triton-fusion-001: 融合决策

满足以下条件时融合：
- 连续算子间无跨 tile 数据依赖
- 中间 tensor 尺寸 ≤ UB 容量
- 后序算子为 element-wise（非 Softmax/归约）

## OPT-triton-fusion-002: constexpr 条件门控

```python
@triton.jit
def fused_kernel(..., HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr):
    # 编译期分支消除：不影响无此 feature 的代码路径
```

## OPT-triton-fusion-003: Cube/Vector 并行

```python
with dl.async_task(0):   # Cube
    acc = tl.dot(a, b)
with dl.async_task(1):   # Vector
    out = activation(acc)
```

