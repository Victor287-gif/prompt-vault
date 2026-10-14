---
type: API
title: 工作空间管理 Workspace Management（全局内存分配、大小推断与参数绑定）
description: 跨函数全局内存的 Workspace 机制：hivm-insert-infer-workspace-size-func / hivm-bind-workspace-arg
  / insert-workspace-for-mix-cv 三个 Pass、memref_ext.alloc_workspace、分配流程与局部内存规划的对比。
tags: [task.lookup, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# 工作空间管理 — Workspace Management

工作空间（Workspace）是编译模型中的全局内存分配机制：与局部内存（`memref.alloc`）不同，其生命周期跨越整个模型执行，由运行时统一分配管理。Workspace 相关 Pass 负责大小推断、参数绑定和 Mix CV 场景的工作空间插入。

## 三个核心 Pass

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hivm-insert-infer-workspace-size-func` | ModuleOp | 为模块中每个函数生成对应的工作空间大小推断函数，根据缓冲区分配计算所需总大小 |
| `hivm-bind-workspace-arg` | ModuleOp | 为每个函数添加工作空间参数（`memref_ext.alloc_workspace`），并把函数内 `memref.alloc` 替换为工作空间内偏移分配 |
| `insert-workspace-for-mix-cv` | ModuleOp | Mix CV 场景插入额外工作空间，存 Cube 与 Vector 核心间的中间数据 |

## 分配流程

```
1. hivm-plan-memory (global-workspace-plan)   为 memref_ext.alloc_workspace 分配全局偏移
2. hivm-insert-infer-workspace-size-func      生成 infer_workspace_size 函数，算总需求
3. hivm-bind-workspace-arg                    workspace 作为函数参数传入，alloc → 偏移
4. insert-workspace-for-mix-cv                （Mix CV 场景）插入 CV 中间数据工作空间
```

## MemRefExt 方言操作

```mlir
%workspace = memref_ext.alloc_workspace() : memref<256xi8>
```

- `memref_ext.alloc_workspace`：分配全局工作空间缓冲区；
- `memref_ext.bind_workspace_arg`：把工作空间绑定到函数参数。

## 复用与大小推断

- PlanMemory 选项 `enable-global-workspace-reuse`（默认 false）：开启后生命周期不重叠的缓冲区共享工作空间偏移。
- 推断函数逻辑：遍历函数内所有 `memref.alloc` 与 `memref_ext.alloc_workspace` → 计算各分配大小（计对齐）→ 计 Inplace 复用后实际大小 → 返回总字节数。

```mlir
func.func @infer_workspace_size() -> i64 {
  %size = arith.constant 65536 : i64
  return %size : i64
}
```

## 与局部内存规划对比

| 特性 | local-mem-plan | global-workspace-plan |
|------|----------------|------------------------|
| 分配方式 | `memref.alloc` → `pointer_cast(offset)` | `memref_ext.alloc_workspace` → 偏移绑定 |
| 生命周期 | 函数内 | 跨函数 |
| 地址空间 | UB/L1/L0C 等 | GM |
| 管理方式 | 编译时静态分配 | 运行时动态分配 |
| 复用 | Inplace 复用 | 全局复用（需开启选项） |

两种规划模式均由 `hivm-plan-memory` 的 `mem-plan-mode` 选项切换，见 [memory_planning.md](memory_planning.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td#L163-L169](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L163-L169)
- [Passes.td#L171-L178](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L171-L178)
- [Passes.td#L180-L188](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L180-L188)
- [Passes.td#L130-L161](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L130-L161)
