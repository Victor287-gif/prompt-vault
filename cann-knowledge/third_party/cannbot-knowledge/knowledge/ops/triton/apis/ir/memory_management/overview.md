---
type: API
title: 内存管理子系统总览（Tensor→MemRef→物理地址变换链与 8 个核心阶段）
description: AscendNPU-IR 内存管理子系统的完整变换链（Bufferization→作用域推断→PlanMemory→对齐→后端 Lowering）、各阶段对应
  Pass、地址空间体系、Inplace 复用与 UB 溢出检测。
tags: [task.lookup, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]

---

# 内存管理子系统总览

内存管理子系统把高层 Tensor 抽象逐步映射到 NPU 物理地址空间，是编译管线最核心的子系统之一。

## 变换链

```
Tensor（高层抽象）
  ▼ Bufferization（Tensor → MemRef）
MemRef + memref.alloc（逻辑缓冲区）
  ▼ Memory Scope Inference（hivm-infer-mem-scope，推断地址空间）
MemRef + #hivm.address_space<ub/cbuf/cc/...>
  ▼ Memory Planning（hivm-plan-memory，分配物理偏移）
MemRef + hivm.hir.pointer_cast(offset)
  ▼ Memory Alignment（对齐约束）
Aligned MemRef
  ▼ Backend Lowering
物理地址访问
```

## 八个核心阶段与对应 Pass

| 阶段 | 关键 Pass | 详卡 |
|------|-----------|------|
| 1. Bufferization：Tensor→MemRef，消除 Tensor 值语义 | `hivm-clone-tensor-empty` 等 | [bufferization.md](bufferization.md) |
| 2. 内存作用域推断：为每个 alloc 标注地址空间 | `hivm-infer-mem-scope` | 见 [memory_hierarchy](../architecture/memory_hierarchy.md) |
| 3. 内存规划：逻辑缓冲区→物理偏移，含 local-mem-plan / global-workspace-plan 两种模式 | `hivm-plan-memory` | [memory_planning.md](memory_planning.md) |
| 4. 多缓冲区：双/多缓冲使 DMA 与计算重叠 | `hivm-mark-multi-buffer` + `hivm-enable-multi-buffer` | [multi_buffer.md](multi_buffer.md) |
| 5. 额外缓冲区：为 temp_buffer 需求分配临时存储 | `hivm-alloc-extra-buffer`、`hivm-outline-alloc-in-VF` | [extra_buffer.md](extra_buffer.md) |
| 6. 内存对齐：满足硬件对齐约束 | `hivm-align-alloc-size`、`hivm-mark/enable-stride-align`、`hivm-lift-lowest-stride` | [memory_alignment.md](memory_alignment.md) |
| 7. 紧耦合缓冲区：Mix CV 下 Cube↔Vector 直传 | `hivm-insert-cv-tight-coupled-buffer` | [tightly_coupled_buffer.md](tightly_coupled_buffer.md) |
| 8. 工作空间管理：跨函数全局内存 | `hivm-insert-infer-workspace-size-func`、`hivm-bind-workspace-arg`、`insert-workspace-for-mix-cv` | [workspace_management.md](workspace_management.md) |

## 地址空间体系

`#hivm.address_space<...>`：zero(0) / gm(1) / cbuf=L1(2) / ca=L0A(3) / cb=L0B(4) / cc=L0C(5) / ub(6)。容量、对齐与数据通路详见 [memory_hierarchy](../architecture/memory_hierarchy.md)。

## PlanMemory 的两个关键行为

**Inplace 复用**：生命周期不重叠的缓冲区共享同一物理偏移。从 plan-memory.mlir 测试可见的典型规律：

- 线性分配：`pointer_cast(0)`, `pointer_cast(8192)`, ... 顺序排布；
- 复用：`pointer_cast(0)` 可被多个生命周期错开的缓冲区复用；
- 循环内分配的缓冲区需独立偏移，不能与循环外缓冲区复用；
- scf.if / scf.for / scf.while 中的缓冲区需特殊处理 yield 操作数的内存规划。

**UB 溢出检测**：分配完成后校验总需求是否超过 UB 容量（910B 为 1572864 bits = 192KB），超出即编译报错：

```mlir
// expected-error {{ub overflow, requires 2560000 bits while 1572864 bits available!}}
%arg0_ub = memref.alloc() : memref<80000xf32, #hivm.address_space<ub>>
```

读 IR 时可用此对应关系定位 UB 容量问题：报错中的 requires/available 单位均为 bits。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td#L171-L188](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L171-L188)
- [plan-memory.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/plan-memory.mlir)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
