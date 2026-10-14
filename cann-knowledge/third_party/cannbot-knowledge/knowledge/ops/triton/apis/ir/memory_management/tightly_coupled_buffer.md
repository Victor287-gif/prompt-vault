---
type: API
title: 紧耦合缓冲区 Tightly Coupled Buffer（Mix CV 场景 Cube↔Vector 直传）
description: Mix CV 场景下 Cube 与 Vector 核心间免 GM 中转的数据传递机制：hivm-insert-cv-tight-coupled-buffer
  Pass、TightlyCoupledBufferAttr、工作原理与配套 Mix CV Pass（insert-load-store-for-mix-cv /
  split-mixed-if-conditionals / insert-cv-data-movement）。
tags: [task.lookup, topic.implementation.memory_management, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# 紧耦合缓冲区 — Tightly Coupled Buffer

紧耦合缓冲区是 Mix CV（Cube-Vector 混合）场景下 Cube 与 Vector 核心间高效数据传递的机制，利用硬件紧耦合存储特性避免数据经 GM 中转，降低延迟与带宽消耗。硬件通路背景（L0C→UB 直通、950 特有）见 [memory_hierarchy](../architecture/memory_hierarchy.md)。

## 核心 Pass：hivm-insert-cv-tight-coupled-buffer（FuncOp）

在 Cube 操作输出与 Vector 操作输入之间插入紧耦合缓冲区，使数据直接从 Cube 核心传到 Vector 核心。依赖方言：hivm、bufferization、annotation。

## TightlyCoupledBufferAttr

- 属性名：`hivm.tightly_coupled_buffer`，参数 `id`（`std::optional<int32_t>`），Assembly 格式 `<id>`；
- 语义：标记某缓冲区为紧耦合缓冲区，用于 Cube↔Vector 数据传递。

```mlir
annotation.mark %buffer {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
  : memref<16x16xf16, #hivm.address_space<ub>>
```

## 工作原理

```
方式 1（无紧耦合缓冲区）：
  Cube L0C → fixpipe → GM → load → Vector UB      （经全局内存，高延迟、带宽受限）

方式 2（有紧耦合缓冲区）：
  Cube L0C → fixpipe → TightlyCoupledBuffer → Vector UB   （直传，低延迟、高带宽）
```

**内存规划联动**：PlanMemory 的 `disable-tightly-coupled-buffer-reuse` 选项控制复用——默认 false 允许紧耦合缓冲区与其他缓冲区共享物理内存；设 true 则独占内存以保证传递可靠性（见 [memory_planning.md](memory_planning.md)）。

## 配套 Mix CV Pass

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hivm-insert-load-store-for-mix-cv` | FuncOp | 为 Mix CV 场景插入 load/store，处理 Cube↔Vector 数据搬运（非 950 设备的 L0C→GM→UB 路径） |
| `hivm-split-mixed-if-conditionals` | FuncOp | 拆分 Mix CV 场景的 if 条件语句，保证 Cube/Vector 操作在各自核心正确执行 |
| `hivm-insert-cv-data-movement` | FuncOp | 插入 Cube-Vector 数据搬运操作（950 布局优化路径） |

三条 CV 数据搬运路径（InsertCVTightCoupledBuffer / InsertCVDataMovement / InsertLoadStoreForMixCV）的选择逻辑与各自使用的 Pipeline 组合，见 [pipeline_execution_model](../architecture/pipeline_execution_model.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td#L559-L566](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L559-L566)
- [HIVMAttrs.td#L1010-L1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017)
- [Passes.td#L552-L557](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L552-L557)
- [Passes.td#L544-L550](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L544-L550)
- [Passes.td#L802-L806](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L802-L806)
