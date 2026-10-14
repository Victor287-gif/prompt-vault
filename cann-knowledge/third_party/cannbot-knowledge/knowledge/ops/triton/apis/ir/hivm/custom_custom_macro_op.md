---
type: API
title: HIVM 自定义跨 Pipe 宏操作（hivm.hir.custom_macro）
description: hivm.hir.custom_macro 操作语义：MacroOpTrait 标记的跨 Pipeline 自定义宏操作，hivm.pipe_in
  / hivm.pipe_out 双 Pipe 属性，编译器按宏操作自动注入跨 Pipe 同步。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 自定义跨 Pipe 宏操作（hivm.hir.custom_macro）

`hir.custom_macro` 是 [hir.custom](custom_custom_op.md) 的跨 Pipeline 版本：以 `MacroOpTrait` 标记为宏操作，适合"数据加载 + 计算 + 后处理"融合这类需要多 Pipeline 协作的自定义实现。

## 操作定义

```
def CustomMacroOp : HIVM_CustomOp<"custom_macro", [MacroOpTrait]> {
  let arguments = (ins StrAttr:$name,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outputs);
  let results = (outs Variadic<AnyType>:$results);
}
```

## 与 CustomOp 的差异

| 特性 | CustomOp | CustomMacroOp |
|------|----------|---------------|
| Trait | SinglePipeOpTrait | MacroOpTrait |
| Pipe 属性 | `hivm.pipe`（单个） | `hivm.pipe_in` + `hivm.pipe_out`（两个） |
| Pipeline 数量 | 单个 | 跨多个 |
| 同步需求 | Pipe 内同步 | 跨 Pipe 同步（InjectSync/GraphSyncSolver 自动注入 set_flag/wait_flag） |
| BuiltinInfo 字段 | coreType, pipe, vfMode | coreType, inPipe, outPipe, vfMode |

其余参数（`$name`、`$inputs`、`$outputs`、`hivm.tcore_type`、`hivm.vf_mode`、`gm_addr_args_indices`）语义同 custom。Pipe 属性名常量：`"hivm.pipe_in"` / `"hivm.pipe_out"`。

## IR 示例

```mlir
%0 = hivm.hir.custom_macro
     { hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V> }
     "my_custom_op"
     ins(%arg0, %arg1, ... : memref<?xf32>, tensor<3x3xi64>, ...)
     outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
```

内置宏操作形式同样存在（如 `"__builtin_gather_load"` 的 macro 形态），可省略属性由编译器推断。

## IR 层约束

1. `pipe_in` 是数据进入的 Pipeline（如 PIPE_MTE2 表示 GM 加载），`pipe_out` 是结果输出 Pipeline（如 PIPE_V 表示 Vector 处理）。
2. 同步由编译器按 pipe_in/pipe_out 自动在操作前后插入（机制见 [sync_sync_injection.md](sync_sync_injection.md)，同步原语见 [sync_pipe_sync.md](sync_pipe_sync.md)）。
3. `getInPipe/setInPipe`、`getOutPipe/setOutPipe` 等方法替代 custom 的单 Pipe 访问器；`isBuiltin()` 同义。
4. outputs 仍为 DestinationStyle DPS init 操作数。

相关：[custom_overview.md](custom_overview.md)、[custom_custom_op.md](custom_custom_op.md)、[macro_overview.md](macro_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMOps.td#L1105-L1167](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L1105-L1167)
- [custom-op.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/custom-op.mlir)
