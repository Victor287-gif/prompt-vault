---
type: Guide
title: Sort / TopK 算子（index-sort） 实现手册
description: '算子类别: indexsort（基于比较重排并返回索引。'
tags: [task.implementation, topic.compute.ordering, topic.compute.indexing, topic.compute.reduction]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Scan_Sort_Ops/sort.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Sort / TopK 算子（index-sort） 实现手册

> ⚠️ **排序原语写 `tl.sort`**，`ext.sort` 在本环境（Triton 3.2.0 + Ascend）会
> `MLIRCompilationError`。详见 [`math_compute__sort.md`](math_compute__sort.md) 顶部说明。

> **更完整的 Sort 实现手册**：本卡源自 `template/index-computation.md §5`，内容较薄
> （原文即标注「当前 memory 目录无专项文档」「待补充」）。同一算子另有
> [`math_compute__sort.md`](math_compute__sort.md)（353 行，含 `tl.sort` UB 上限、
> 升序取负 trick、pad 方向、31 case 实测基准 geomean 2.04），**Sort/TopK 生成时应优先加载它**。
> Sort 的类别判定在 `index_computation` 与 `math_compute` 之间存在歧义，两条路都应能拿到完整知识。

> 类别约束见 [index_computation_constraints.md](../constraints/index_computation_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §5 Sort / TopK 算子（index-sort）

**算子类别**: `index-sort`（基于比较重排并返回索引；`sort` / `topk` / `argsort`）
**典型特征**: 比较-选择模式，与 Gather/Scatter 的索引映射模式差异较大；需注意输出索引 `int64` 与数据类型的对齐
**性能基准**: Level1 索引计算类算子中 8_Sort 当前 memory 目录无专项文档，建议走 sort-topk 专项模板

### §5.0 首次生成必读：为什么必须把主要框架写对

Sort / TopK 是**比较-选择**模式，与 Gather/Scatter 的索引映射模式差异较大。问题规模差异大：小 `K` 可用 bitonic/odd-even 排序网络，大 `K` 需 reduce-split 或分桶 + 局部排序。**首次生成如果把框架写偏（例如大 K 用全排序网络、未注意 int64 索引对齐），性能和精度都很难救回**。

### §5.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 根据问题规模选择排序策略

- **必须**根据 `K` 大小选择策略：小 `K` 用 bitonic/odd-even 排序网络；大 `K` 用 reduce-split 或分桶 + 局部排序。
- **Why:** 全排序网络复杂度随 K 增长过快，大 K 时性能极差。

#### L1.2 输出索引 `int64` 必须与数据类型对齐

- **必须**注意输出索引的 `int64` 与数据类型的对齐。
- **Why:** 索引 dtype 不匹配会导致下游算子错误。

#### L1.3 优先使用向量比较原语

- **必须**在 Ascend 上比较算子优先使用向量比较原语。
- **Why:** 标量比较会破坏 SIMD，无法发挥 Vector Core 并行能力。

### §5.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 策略选择

- 小 `K` 用 bitonic/odd-even 排序网络。
- 大 `K` 用 reduce-split 或分桶 + 局部排序。
- 当前 memory 目录中无 Sort 专项经验，建议在生成时单独走 `sort-topk` 模板或做针对性设计。

### §5.3 Sort / TopK 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 8_Sort | 待补充 | 待补充 | 当前 memory 目录无专项文档，建议走 sort-topk 专项模板 |

**关键结论**:
1. Sort/TopK 与 Gather/Scatter 的索引映射模式差异较大，不要套用 Gather/Scatter 经验。
2. 大 K 时不要使用全排序网络。
3. 注意输出索引 `int64` 与数据类型的对齐。

---
