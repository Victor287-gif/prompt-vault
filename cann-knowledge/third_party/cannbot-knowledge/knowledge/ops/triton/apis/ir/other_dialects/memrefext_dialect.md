---
type: API
title: MemRefExt 方言（memref_ext.alloc_workspace）
description: MemRefExt 方言扩展标准 MemRef 方言，提供 NPU 设备特有的 memref_ext.alloc_workspace 工作空间分配操作，与
  HACC 方言协作完成 Host 端大小推断、传参和 Device 端子区域分配。
tags: [task.lookup, topic.compilation, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# MemRefExt 方言

MemRefExt 方言（方言名 `memref_ext`，C++ 命名空间 `::bishengir::memref_ext`，依赖 arith）扩展 MLIR 标准 MemRef 方言，提供 NPU 设备特有的内存分配操作，核心是 Kernel 临时存储用工作空间（Workspace）的分配。工作空间管理的完整机制见 [workspace_management](../memory_management/workspace_management.md)。

## memref_ext.alloc_workspace

分配工作空间内存。工作空间大小由 Host 端 `hacc.infer_workspace_shape_function` 推断，Device Kernel 内用本操作从传入的工作空间中切出子区域。内存效果为 `MemAlloc<DefaultResource, 0, FullEffect>`。

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `workspaceArg` | `Optional<AnyMemRef>` | 已有工作空间参数（复用场景，配合 `from ... offset = [...]` 语法） |
| `dynamicSize` | `Variadic<Index>` | 动态维度大小 |
| `offset` | `Variadic<Index>` | 偏移量 |
| `memref` | `AnyMemRef` | 分配的 MemRef 结果 |

```mlir
// 静态大小分配
%ws = memref_ext.alloc_workspace() : memref<100xi8>

// 动态大小分配
%ws = memref_ext.alloc_workspace(%dynamic) : memref<2x?xi32>

// 从已有工作空间带偏移分配
%ws = memref_ext.alloc_workspace(%dynamic) from %arg offset = [%offset]
  : from memref<?xi8> to memref<2x?xi32>
```

## Lowering 路径

编译流程中本操作被 lowering 为两条路径之一：HIVM 层的内存分配，或标准 `memref.alloc` 加偏移计算。

## 与 HACC 方言的协作

工作空间内存生命周期由 HACC 方言管理（HACC 方言总览见 [overview](../hacc/overview.md)，参数类型见 [kernel_args](../hacc/kernel_args.md)）：

```
1. Host 端 hacc.infer_workspace_shape_function 计算工作空间大小
2. Host 端分配工作空间内存
3. 工作空间以 hacc.arg_type = #hacc.arg_type<workspace> 传入 Device Kernel
4. Device Kernel 内 memref_ext.alloc_workspace 从传入空间分配子区域
```

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [MemRefExtOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtOps.td)
- [MemRefExtBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtBase.td)
