---
type: Optimization
title: 内存访问优化
description: Triton Ascend 内存访问优化：连续访问、Block Pointer、对齐、Cache 复用与软件流水线策略。
tags: [task.optimization, topic.data_movement.memory_access, topic.data_movement, topic.alignment, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 内存访问优化

HBM→UB 带宽优化策略，覆盖对齐、连续访问、block_ptr、cache 复用和软件流水线。

## OPT-triton-mem-001: 连续内存访问优先

避免 stride 跳跃访问导致 cache miss 率 >50%：
- ✅ 一维连续 `ptr + pid*BLOCK + tl.arange(0, BLOCK)`
- ❌ 多维 stride 偏移产生离散地址

## OPT-triton-mem-002: 2D Block Pointer

使用 `tl.make_block_ptr` 替代手动指针计算：
```python
a_block_ptr = tl.make_block_ptr(
    base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m*BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
```
编译器可生成更优的地址计算指令。

## OPT-triton-mem-003: 512B 行宽对齐

matmul 的 BLOCK_N × dtype_bytes 必须是 512 的整数倍。fp16: BLOCK_N≥256。

## OPT-triton-mem-004: 转置 Load 优化

行优先 load + `tl.trans` 寄存器转置，消除转置访存开销。

## OPT-triton-mem-005: 跨阶段寄存器复用

barrier 前后重复加载同一数据 → 寄存器复用消除重复 load。

