---
type: API
title: 内存规划（hivm-plan-memory Pass：物理偏移分配、Inplace 复用、溢出检测）
description: hivm-plan-memory Pass 的两种规划模式（local-mem-plan/global-workspace-plan）、核心算法（生命周期分析、Inplace
  复用、控制流与子视图处理）、选项表与 UB 溢出检测。
tags: [task.lookup, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]

---

# 内存规划 — PlanMemory Pass

`hivm-plan-memory`（ModuleOp 作用域）是内存管理子系统的核心：把逻辑缓冲区（`memref.alloc`）映射到物理内存偏移，通过生命周期与依赖分析实现 Inplace 复用与偏移分配。

## 选项

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `mem-plan-mode` | MemPlanMode | `LOCAL_MEM_PLAN` | 规划模式 |
| `enable-global-workspace-reuse` | bool | false | 全局工作空间复用 |
| `enable-print-memory-allocated-size` | bool | false | 打印已分配内存大小 |
| `restrict-inplace-as-isa` | bool | false | 限制内存 inplace 与 ISA 一致 |
| `simt-vf-dynamic-size` | int | 216 | SIMT VF 的动态 UB 大小（KB） |
| `disable-tightly-coupled-buffer-reuse` | bool | false | 禁用紧耦合缓冲区复用（见 [tightly_coupled_buffer.md](tightly_coupled_buffer.md)） |

`MemPlanMode`：`LOCAL_MEM_PLAN(0)` 为 `memref.alloc` 分配局部偏移；`GLOBAL_WORKSPACE_PLAN(1)` 为 `memref_ext.alloc_workspace` 分配全局工作空间偏移（见 [workspace_management.md](workspace_management.md)）。

## 规划效果

```mlir
// 变换前
%copy_in_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
%dst1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>

// 变换后：alloc 替换为地址空间内字节偏移的 pointer_cast
%0 = arith.constant 0 : i64
%copy_in_ub = hivm.hir.pointer_cast(%0) : memref<16x16x16xf16, #hivm.address_space<ub>>
%1 = arith.constant 8192 : i64
%dst1 = hivm.hir.pointer_cast(%1) : memref<16x16x16xf16, #hivm.address_space<ub>>
```

## 核心算法

1. **生命周期分析**：确定每个缓冲区定义点到末次使用的活跃区间（liveness interval）。
2. **Inplace 复用**：活跃区间不重叠的缓冲区共享物理偏移。典型用例中 4 个 UB 缓冲区经复用只占 2 个偏移：`copy_in_ub→0`、`dst1→8192`、`dst2→0`（复用 copy_in）、`copy_out_ub→8192`（复用 dst1）。
3. **控制流处理**：`scf.for` 循环内缓冲区跨迭代不能 Inplace 复用；`scf.if` 分支间需考虑冲突；`scf.while` 的 before/after 区域、`cf.cond_br` 的块间缓冲区传递均需特殊处理。
4. **子视图追踪**：支持 `memref.subview` / `collapse_shape` / `expand_shape` / `reshape` / `view` / `reinterpret_cast`，正确关联子视图与基础分配。
5. **按操作类型的 Inplace 策略**：
   - vcast：源/目标类型大小相同可 Inplace，不同则独立偏移；
   - 带 broadcast 的操作：需要额外 temp_buffer，策略取决于 broadcast 维度；
   - 带 temp_buffer 的操作：temp_buffer 与主缓冲区独立分配（见 [extra_buffer.md](extra_buffer.md)）。

## UB 溢出检测

分配完成后检测 UB 需求是否超容量（910B：1572864 bits = 192KB），超出即编译期报错：

```
ub overflow, requires 2560000 bits while 1572864 bits available!
```

## 前置 Pass

`hivm-normalize-loop-iterator`（FuncOp）：在 PlanMemory 之前规范化循环迭代器的特殊状态，保证内存规划正确性。

多缓冲区场景下 `pointer_cast` 会携带多个偏移，见 [multi_buffer.md](multi_buffer.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td#L130-L161](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L130-L161)
- [plan-memory.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/plan-memory.mlir)
- [HIVMAttrs.td#L770-L777](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L770-L777)
- [Passes.td#L653-L658](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L653-L658)
