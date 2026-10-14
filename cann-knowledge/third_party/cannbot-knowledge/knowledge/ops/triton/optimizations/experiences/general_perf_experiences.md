---
type: Optimization
title: 通用性能优化经验聚合
description: 从 16 条通用性能 YAML 经验中蒸馏的跨算子优化模式：标量消除、内存访问优化、autotune 调参与硬件约束规避。
tags: [task.optimization, topic.programming_model.execution_unit.scalar, topic.data_movement, topic.tiling, topic.performance.analysis]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 通用性能优化经验聚合

从 16 条通用性能 YAML 经验中蒸馏。这些优化跨算子类型复用，是解决 `aiv_scalar_ratio` 高和 MTE2 瓶颈的核心武器。

## 共性模式

1. **标量消除**：i32/i64 CMP/ADD → fp32，消除 Vector→Scalar fallback
2. **block_ptr 审慎使用**：小 tile + boundary_check → 改手动指针 + 向量化 mask
3. **constexpr 编译期求值**：tl.cdiv/min → 常量替代

## 高质量经验

| 经验 ID | 提升 | 核心优化 | 适用条件 |
|---------|------|----------|----------|
| exp_ascend_blockptr_boundary_scalar_001 | **60%** | block_ptr+边界检查→手动指针+fp32 mask | aiv_scalar_ratio>30%，小 tile 频繁 load/store |
| exp_perf_011 | 50% | block_ptr store/load→手动指针+向量化 mask | block_ptr+boundary_check |
| exp_ascend_int_compare_scalar_001 | 40% | i32/i64 标量降级规则与修复 | aiv_scalar_ratio 高 |
| exp_ascend_int64_address_001 | 50% | i64 地址计算 → i32 | 地址计算含 int64 |
| exp_perf_012 | 35% | i32 比较 → fp32 比较 | 含 i32 LT/GT/LE/GE |
| exp_perf_015 | 30% | 行优先 load + tl.trans 寄存器转置 | 转置视角加载行优先张量 |
| exp_perf_017 | 20% | 扩展 num_warps/num_stages/tile autotune | 搜索空间过窄 |
| exp_perf_013 | 15% | min() → tl.where | 运行时 min() 求值 |
| exp_perf_014 | 10% | tl.cdiv → 编译期常量 | 循环边界含 tl.cdiv |
| exp_perf_016 | 15% | 跨阶段寄存器复用 | barrier 前后重复加载同数据 |
| exp_ascend_compile_hint_001 | 25% | 移除冗余 compile_hint | block_ptr/load 已加 compile_hint |

