---
type: Optimization
title: Tiling 策略
description: Triton Ascend 三级 Tiling 层次（ncore/xblock/xblock_sub）、UB 容量约束、块大小选择与 AutoTiling
  搜索空间。
tags: [task.optimization, topic.tiling, topic.implementation.memory_management, topic.alignment]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# Tiling 策略

Tiling 是将计算 workload 映射到 AI Core UB 内存的主要机制。不正确的 tile sizing 导致 UB 溢出、对齐惩罚或利用率不足。

## 三级层次

| 层次 | 名称 | 范围 | 说明 |
|------|------|------|------|
| L1 | ncore | 跨核拆分 | 将总工作量分配到各 AI Core（grid 维度） |
| L2 | xblock | 每核数据 | 单个 core 的 program 处理的 tile |
| L3 | xblock_sub | 核内细粒度 | 在 kernel 内 for-loop 迭代的子 tile |

## UB 容量约束

Atlas 800T/I A2 每个 AI Core 的 UB 为 **192KB**。所有 tile 数据必须在此范围内：
- fp16/bf16: BLOCK ≤ 98304 elements
- fp32: BLOCK ≤ 49152 elements
- matmul 同时持有 A/B/C tile，实际可用约 1/3（~64KB per matrix）

## 块大小选择

| 算子类型 | 推荐 BLOCK | autotune 范围 |
|----------|-----------|---------------|
| MatMul | BLOCK_M/N∈{128,256}, BLOCK_K=256 | `[128,256]` |
| Elementwise | BLOCK_SIZE 按 2 的幂 | `{256,512,1024,2048,4096}` |
| Normalization | BLOCK_SIZE_N 整行或 max 4096 | heuristic |

## OPT-triton-tiling-001: 对齐检查

- matmul 的 BLOCK_N × dtype_bytes 为 512 的整数倍
- BLOCK_M, BLOCK_N, BLOCK_K 均为 2 的幂次
- N≥512 时启用对角 tiling

