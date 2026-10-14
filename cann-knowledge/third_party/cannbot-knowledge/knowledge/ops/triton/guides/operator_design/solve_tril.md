---
type: Guide
title: solve_tril 三角矩阵求逆
description: Triton Ascend solve_tril 的分层分块递推架构：16×16 原子单元、逐层 merge、结构零元消除与寄存器预加载策略。
tags: [task.design, topic.compute.composite, topic.programming_model, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# solve_tril 三角矩阵求逆

计算 `(I+A)^(-1)`，A 为严格下三角矩阵。用于 Flash Linear Attention (FLA) 中分块求解注意力矩阵。

核心架构：以 16×16 为原子单元，BT=16 直接递推，BT=32/64 逐层 merge。所有 (batch, head) 对完全独立，grid = (NT, B×H)。

## 检测标准

- 严格下三角矩阵求逆
- BT ∈ {16, 32, 64}
- Flash Linear Attention 分块求解场景

## 经验路由

| 场景 | 经验 ID | 关键技术 |
|------|---------|----------|
| 结构零元 mask 消除 | exp_solve_tril_001 | 利用矩阵结构删除 `tl.where` (1.33-1.87x) |
| 寄存器预加载 | exp_solve_tril_002 | extract_slice 替代 HBM load |
| Constexpr 双路径展开 | exp_solve_tril_003 | 完整 tile 编译期展开+边界 tile 运行时循环 |
| 三层分级调度 | exp_solve_tril_004 | BT=16/32/64 分治架构 |
| CANN 兼容审计 | exp_solve_tril_005 | 删除不兼容参数 |

## 关键技术

- **分层分块递推**：16×16 原子单元 → 逐层 merge（32×32, 64×64）
- **结构零元消除**：利用严格下三角结构删除冗余 mask/where
- **寄存器预加载**：load 到寄存器后用 `extract_slice` 切片，消除 HBM 重读

