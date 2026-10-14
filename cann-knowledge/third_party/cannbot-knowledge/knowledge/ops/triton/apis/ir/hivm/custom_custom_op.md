---
type: API
title: HIVM 自定义单 Pipe 操作（hivm.hir.custom）
description: hivm.hir.custom 操作语义：单 Pipeline 自定义扩展操作，name 属性标识（__builtin 前缀内置），tcore_type/pipe/vf_mode
  必需属性，DestinationStyle 输出，gm_addr_args_indices 地址参数标记。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 自定义单 Pipe 操作（hivm.hir.custom）

`hir.custom` 在**单个 Pipeline** 上执行自定义实现，用 `name` 属性标识操作类型。适用：现有操作功能缺失、性能不佳或需要私有实现。Triton 自定义操作可经 `tl.extern` 类机制映射为 hir.custom。机制总览与 Builtin 清单见 [custom_overview.md](custom_overview.md)。

## 操作定义

```
def CustomOp : HIVM_CustomOp<"custom", [SinglePipeOpTrait]> {
  let arguments = (ins StrAttr:$name,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outputs);
  let results = (outs Variadic<AnyType>:$results);
}
```

基类 `HIVM_CustomOp` 自带 `AttrSizedOperandSegments`、`MemoryEffects<[MemRead, MemWrite]>`、`HIVMInferCoreTypeInterface`。

## 参数与属性

| 项 | 类型 | 说明 |
|----|------|------|
| `$name` | StrAttr | 操作名，`__builtin_` 前缀为内置 |
| `$inputs` | Variadic\<AnyType\> | 输入（可变数量，memref/tensor/标量混合） |
| `$outputs` | Variadic\<AnyType\> | 输出/初始化参数（DestinationStyle DPS init） |
| `hivm.tcore_type` | TCoreTypeAttr | 必需：Core 类型 |
| `hivm.pipe` | PipeAttr | 必需：执行的 Pipeline |
| `hivm.vf_mode` | VFModeAttr | 必需：Vector 运行模式（Cube Core 时忽略） |
| `gm_addr_args_indices` | DenseI32ArrayAttr | 可选：标记含 GM 纯地址的参数，lowering 保留地址值 |

内置操作（`__builtin_gather_load` / `__builtin_index_select`）可省略必需属性，编译器自动推断并校验参数。

## IR 示例

```mlir
// 自定义操作（显式属性）
%0 = hivm.hir.custom
     { hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD> }
     "my_custom_op"
     ins(%arg0, %arg1, ... : memref<?xf32>, tensor<3x3xi64>, ...)
     outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>

// 内置 __builtin_index_select（SIMT 模式 + 自定义 extra_attr）
%1 = hivm.hir.custom
       {extra_attr = "srcStrideLength=3", hivm.vf_mode = #hivm.vf_mode<SIMT>}
       "__builtin_index_select"
       ins(%arg0, %arg1, ... : memref<?xf32>, tensor<16x400xi32>, ...)
       outs(%0 : tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
```

## 类方法

`getCoreType/setCoreType`、`getPipe/setPipe`、`getVFMode/setVFMode`、`getGMAddrArgsIndices/setGMAddrArgsIndices`、`isBuiltin()`、`getDpsInitsMutable()`（DPS 接口）。

## IR 层约束

1. `SinglePipeOpTrait`：单 Pipeline 执行，同步只涉及 Pipe 内（对比 [custom_macro](custom_custom_macro_op.md) 跨 Pipe）。
2. `AttrSizedOperandSegments`：inputs/outputs 数量由属性段记录。
3. 用户自定义实现的注册机制当前未完全开放（源码标记 TODO）；内置操作由编译器链接到模板库。
4. `__builtin_gather_load` 与独立的 `hir.gather_load`（见 [dma_gather_scatter.md](dma_gather_scatter.md)）功能相近，custom 版本可通过属性灵活配置。

相关：[custom_overview.md](custom_overview.md)、[custom_custom_macro_op.md](custom_custom_macro_op.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMOps.td#L1069-L1103](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L1069-L1103)
- [HIVMOps.td#L904-L967](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L904-L967)
- [custom-op.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/custom-op.mlir)
