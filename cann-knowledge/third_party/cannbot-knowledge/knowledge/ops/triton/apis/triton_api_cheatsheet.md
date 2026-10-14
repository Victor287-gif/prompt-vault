---
type: API
title: Triton Ascend API 速查
description: Triton Ascend API 完整速查表：Kernel 声明、Program Identity、内存访问、计算、同步与 Ascend
  扩展（dl/tle）。
tags: [task.lookup, topic.programming_model, topic.data_movement.memory_access, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# Triton Ascend API 速查

## Kernel 声明

| API | 说明 |
|-----|------|
| `@triton.jit` | 编译 kernel 为 device 可执行代码 |
| `@triton.autotune(configs, key)` | 从配置列表中自动选择最优 |
| `@triton.heuristics({'KEY': fn})` | 基于输入的动态配置 |
| `tl.constexpr` | 编译期常量（启用特化与死代码消除） |

## Program Identity

| API | 说明 |
|-----|------|
| `tl.program_id(axis)` | 当前 program 在 axis 轴的 ID |
| `tl.num_programs(axis)` | axis 轴的 program 总数 |

## 内存访问

| API | 说明 | Ascend 注意 |
|-----|------|-------------|
| `tl.load(ptr, mask, other)` | 从 GM/L1 加载到 UB | 必须带 mask 或 boundary_check |
| `tl.store(ptr, val, mask)` | 从 UB 写回 GM/L1 | |
| `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` | 2D Block Pointer | Ascend 推荐用于 matmul，简化地址计算 |
| `tl.advance(ptr, offsets)` | 推进 block pointer | |

## 计算

| API | 说明 | Ascend 注意 |
|-----|------|-------------|
| `tl.dot(a, b)` | 矩阵乘（Cube 单元） | A: `[M,K]` 行主序，B: `[K,N]` 行主序 |
| `tl.sum(x, axis)` | 归约求和 | 用 fp32 累加器 |
| `tl.max(x, axis)` | 归约最大值 | |
| `tl.exp(x)` / `tl.log(x)` | 指数/对数 | fp16 下注意溢出防护 |
| `tl.sqrt(x)` / `tl.rsqrt(x)` | 平方根/倒数平方根 | `tl.rsqrt` 比 `1/tl.sqrt` 更快更准 |
| `tl.where(cond, x, y)` | 条件选择 | cond 必须是 mask tensor |
| `tl.arange(start, end)` | 整数序列 | 做比较前 `.to(tl.float32)` |
| `tl.zeros(shape, dtype)` | 零张量 | |
| `tl.full(shape, val, dtype)` | 全填充张量 | |

## 同步

| API | 说明 |
|-----|------|
| `tl.debug_barrier()` | 调试用同步屏障 |
| `tl.maximum(x, y)` / `tl.minimum(x, y)` | 逐元素最大/最小 |

## Ascend 扩展（dl 模块）

| API | 说明 |
|-----|------|
| `dl.async_task(unit_id)` | 将代码块分配到指定单元并行执行（Cube=0, Vector=1） |
| `dl.compile_hint(hint)` | 编译提示：`"dot_pad_only_k"` 仅对 K 维 padding |
| `dl.grouped_launch_diagonal` | 对角核映射（替代 tl.swizzle2d） |

