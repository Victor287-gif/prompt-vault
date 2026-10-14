---
type: API
title: 多缓冲区 Multi-Buffer（mark/enable 两 Pass 与 cv-pipelining）
description: 多缓冲区流水线优化：hivm-mark-multi-buffer 标记 hivm.multi_buffer 属性、hivm-enable-multi-buffer
  转换分配、MultiBufferStrategy 选项、PlanMemory 后多偏移 pointer_cast 形态与 cv-pipelining 配合。
tags: [task.lookup, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# 多缓冲区 — Multi-Buffer

多缓冲区是流水线优化技术：为同一逻辑缓冲区分配多个物理实例，让 DMA 传输与计算重叠执行。Ascend NPU 上通常表现为双缓冲（Double Buffer）或多缓冲。

## 两个核心 Pass

### hivm-mark-multi-buffer（FuncOp）

为操作标记 `hivm.multi_buffer` 属性。`enable-auto=true` 时自动为符合条件的操作标记；为 false 时不做任何事。**L0C 地址空间的缓冲区不会被标记为多缓冲**。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enable-auto` | bool | false | 自动标记多缓冲 |
| `limit-auto-multi-buffer-only-for-local-buffer` | bool | false | 禁止 workspace 的自动多缓冲标记 |
| `limit-auto-multi-buffer-of-local-buffer` | MultiBufferStrategy | `CUBE_NO_L0C` | 限制 local buffer 的自动多缓冲策略 |
| `limit-mix-auto-multi-buffer-buffer` | MultiBufferStrategy | `ONLY_CUBE` | 限制 Mix 场景的自动多缓冲策略 |
| `set-workspace-multibuffer` | unsigned | 2 | 覆盖 workspace 的多缓冲数量 |

MultiBufferStrategy 取值：`no-limit`（不限制）、`no-l0c`（禁止 L0C 多缓冲）。

### hivm-enable-multi-buffer（FuncOp）

读取 `hivm.multi_buffer` 属性，把单缓冲区分配实际转换为多缓冲区分配。

## 工作流程

```
1. hivm-mark-multi-buffer     标记 annotation.mark %buf {hivm.multi_buffer = 2 : i32}
2. hivm-enable-multi-buffer   单缓冲分配 → 多缓冲分配
3. hivm-plan-memory           为每个实例分配独立物理偏移
```

## IR 形态

标记阶段：
```mlir
%src_ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
annotation.mark %src_ub {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
```

PlanMemory 后，`pointer_cast` 携带与实例数相同数量的偏移：
```mlir
// 双缓冲
%src_ub = hivm.hir.pointer_cast(%const0, %const1) : memref<16xf16, #hivm.address_space<ub>>
// 四缓冲
%src_ub = hivm.hir.pointer_cast(%const0, %const1, %const2, %const3) : memref<16xf16, #hivm.address_space<ub>>
```

`hivm.multi_buffer` 属性为 i32，语义即缓冲区实例数（2=双缓冲，4=四缓冲）。读 last_pass.mlir 时，`pointer_cast` 的偏移个数可直接反映多缓冲深度。

## cv-pipelining

`cv-pipelining`（FuncOp）与多缓冲配合，对多缓冲化的 Mix CV 操作做 Cube-Vector 流水线调度；选项 `pipeline-depth`（int，默认 -1）指定流水线深度。Cube-Vector 数据通路背景见 [memory_hierarchy](../architecture/memory_hierarchy.md) 与 [tightly_coupled_buffer.md](tightly_coupled_buffer.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td#L55-L90](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L55-L90)
- [Passes.td#L92-L100](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L92-L100)
- [HIVMAttrs.td#L962-L966](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L962-L966)
- [Passes.td#L750-L759](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L750-L759)
