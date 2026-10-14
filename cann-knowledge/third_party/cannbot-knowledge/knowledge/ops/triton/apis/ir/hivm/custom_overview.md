---
type: API
title: HIVM Custom 操作机制总览（Builtin 与自定义扩展）
description: HIVM Custom 扩展机制：hir.custom（单 Pipe）与 hir.custom_macro（跨 Pipe）两级粒度，name
  属性标识操作，__builtin 前缀内置操作（__builtin_gather_load / __builtin_index_select），BuiltinInfo
  结构与必需属性清单。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM Custom 操作机制总览

Custom 机制是 HIVM 的通用扩展接口：现有操作无法满足需求（功能缺失、性能不佳或需私有实现）时，用 Custom 操作承载自定义实现。编译器按 `name` 属性查找实现——`__builtin` 前缀为编译器自带实现，其余 name 需用户提供。

## 两级粒度

| 操作 | 助记符 | Trait | Pipe 属性 | 详细卡 |
|------|--------|-------|----------|--------|
| CustomOp | `hir.custom` | SinglePipeOpTrait | `hivm.pipe`（单个） | [custom_custom_op.md](custom_custom_op.md) |
| CustomMacroOp | `hir.custom_macro` | MacroOpTrait | `hivm.pipe_in` + `hivm.pipe_out` | [custom_custom_macro_op.md](custom_custom_macro_op.md) |

跨 Pipe 的 custom_macro 会被 InjectSync/GraphSyncSolver 按宏操作处理（前后自动插跨 Pipe 同步，见 [sync_sync_injection.md](sync_sync_injection.md)）。

## Builtin 操作

| Builtin Name | 说明 | Core Type | Pipe |
|-------------|------|-----------|------|
| `__builtin_gather_load` | Gather Load | VECTOR | PIPE_V |
| `__builtin_index_select` | Index Select | VECTOR | PIPE_V |

## BuiltinInfo 结构

| 字段 | CustomOp | CustomMacroOp |
|------|----------|---------------|
| coreType | TCoreType | TCoreType |
| pipe / inPipe | PIPE | PIPE |
| outPipe | — | PIPE |
| vfMode | VFMode | VFMode |
| getOpLibraryCallName | function\<string(Op)\> | function\<string(Op)\> |
| gmAddrArgsIndices | SmallVector\<size_t\> | SmallVector\<size_t\> |

## 属性速查

### 必需属性

| 属性 | 适用 | 说明 |
|------|------|------|
| `hivm.tcore_type` | 两者 | 执行的 Core 类型（TCoreTypeAttr，见 [attr_enumerations.md](attr_enumerations.md)） |
| `hivm.vf_mode` | 两者 | Vector 运行模式（VFModeAttr；Cube Core 时忽略） |
| `hivm.pipe` | 仅 custom | 执行的 Pipe |
| `hivm.pipe_in` / `hivm.pipe_out` | 仅 custom_macro | 输入 / 输出 Pipe |

### 可选属性

| 属性 | 说明 |
|------|------|
| `gm_addr_args_indices` | DenseI32ArrayAttr，指示哪些参数是 GM 纯地址，lowering 时保留地址值而不从 memref 取数 |

内置操作可省略上述必需属性，编译器自动推断并规范化。

相关：[custom_custom_op.md](custom_custom_op.md)、[custom_custom_macro_op.md](custom_custom_macro_op.md)、[macro_overview.md](macro_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMOps.td#L904-L1167](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L904-L1167)
- [custom-op.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/custom-op.mlir)
