---
type: Optimization
title: 并行化与循环优化
description: Triton Ascend 并行化与循环优化：grid-stride 循环、multi-round 调度、LICM（循环不变量外提）与负载均衡策略。
tags: [task.optimization, topic.parallelism, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 并行化与循环优化

将工作量高效分配到多核并优化核内循环结构。核心策略：grid-stride 均衡、LICM、循环展开与 multi-round 调度。

## OPT-triton-par-001: Grid-Stride Loop

```python
grid = (min(n_blocks, NUM_CORES),)
for block_id in range(pid, tl.cdiv(n, BLOCK_SIZE), tl.num_programs(0)):
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

适用：n_blocks >> NUM_CORES

## OPT-triton-par-002: LICM（循环不变量外提）

将循环中不变的地址计算、mask 生成提到循环外：
```python
# ❌ 循环内重复计算
for k in range(0, K, BLOCK_K):
    offs_k = k + tl.arange(0, BLOCK_K)  # 每次重建
# ✅ 外提
base_k = tl.arange(0, BLOCK_K)
for k in range(0, K, BLOCK_K):
    offs_k = k + base_k
```

## OPT-triton-par-003: 循环展开

手动展开 K 轴循环以隐藏访存延迟（适用于 matmul）：
```python
for k in range(0, K, 4*BLOCK_K):
    a0 = tl.load(...); a1 = tl.load(...)
    a2 = tl.load(...); a3 = tl.load(...)
    acc += tl.dot(a0,b0) + tl.dot(a1,b1) + tl.dot(a2,b2) + tl.dot(a3,b3)
```

## OPT-triton-par-004: 负载均衡

- MoE 场景：work-stealing 动态分配 expert 到 core
- Batch/GQA 场景：按 head 维度划分确保每核等量

