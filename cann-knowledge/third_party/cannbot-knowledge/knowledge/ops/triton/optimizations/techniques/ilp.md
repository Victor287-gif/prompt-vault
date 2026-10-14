---
type: Optimization
title: 指令级并行 (ILP) 优化
description: Triton Ascend ILP 优化：care_padding、for-loop tiling、dl.async_task 异构流水与多缓冲策略。
tags: [task.optimization, topic.implementation.pipeline, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/cv_autotune/hardware_consts.py
  role: constraint
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 指令级并行 (ILP) 优化

提升单个 AI Core 内计算与访存指令的并行度。核心思路：消除 load/compute 串行依赖、Cube/Vector 异步并行、多缓冲隐藏延迟。

## OPT-triton-ilp-001: care_padding=False

当 padding 不影响正确性时，设置 `care_padding=False` 使 load 不等待 compute：
```python
data = tl.load(ptr + offs, mask=mask, care_padding=False)
```

## OPT-triton-ilp-002: for-loop tiling

将单次大 tile 的 load+compute+store 拆为多个小 tile 循环，使 load 和 compute 重叠：
```python
for i in range(0, n, BLOCK_SIZE * FACTOR):
    # load sub-tile → compute → accumulate
```

## OPT-triton-ilp-003: dl.async_task 异构流水

将 Cube 计算和 Vector 后处理分配到不同单元并行：
```python
with dl.async_task(0):   # Cube：矩阵乘
    acc = tl.dot(a, b)
with dl.async_task(1):   # Vector：激活
    out = tl.sigmoid(acc)
```

## OPT-triton-ilp-004: 多缓冲

双缓冲或三缓冲让 MTE2 数据搬运与 Vector 计算重叠。

