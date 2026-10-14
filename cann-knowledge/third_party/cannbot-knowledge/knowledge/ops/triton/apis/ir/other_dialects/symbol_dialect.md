---
type: API
title: Symbol 方言（symbol.symbolic_int / bind_symbolic_shape）
description: Symbol 方言处理动态形状表示：symbol.symbolic_int 定义带范围约束的符号化整数（index），symbol.bind_symbolic_shape
  用仿射映射把形状表达式绑定到张量，配套 symbol-propagate / symbol-erase / symbol-to-encoding 等 Pass
  支持编译期动态形状推理。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.scalar, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Symbol 方言

Symbol 方言（方言名 `symbol`，C++ 命名空间 `::mlir::symbol`）处理动态形状表示：通过符号化整数和仿射映射描述张量动态维度之间的关系，使编译器能在编译期推理动态形状（范围约束、维度间算术关系），为 Tiling 和标量边界推导提供依据。

## 操作定义

### symbol.symbolic_int

表示一个带范围约束的符号化整数，返回 `index` 类型（`Pure`）。可通过仿射映射表达与其他符号的约束关系。

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `symbol_name` | `FlatSymbolRefAttr` | 符号名称 |
| `min_val` / `max_val` | `I64Attr` | 取值范围（含端点） |
| `int_symbols` | `Variadic<Index>` | 引用的其他符号整数 |
| `int_expressions` | `OptionalAttr<AffineMapAttr>` | 仿射映射表达式 |
| `result` | `Index` | 结果 |

```mlir
%0 = symbol.symbolic_int @s0 {min_val = 5, max_val = 10} : index
%1 = symbol.symbolic_int @s1 {min_val = 2, max_val = 20} : index
// 带仿射映射：result = s0 * s1
%2 = symbol.symbolic_int @s2 [%0, %1],
  affine_map<()[s1, s2] -> (s1 * s2)> {min_val = 2, max_val = 20} : index
```

`int_expressions` 存在时，`int_symbols` 按顺序对应仿射映射的局部符号，结果值由映射计算：`result = affine_map(int_symbols[0], int_symbols[1], ...)`。

### symbol.bind_symbolic_shape

把形状表达式绑定到张量，用仿射映射描述动态维度与符号的关系。仿射映射的每个结果对应张量的一个维度，局部符号与 `shape_symbols` 一一对应：`dim[i] = affine_map(shape_symbols...)[i]`。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `operand` | `AnyShaped` | 目标张量 |
| `shape_symbols` | `Variadic<Index>` | 形状符号（与仿射映射局部符号 1:1） |
| `shape_expressions` | `AffineMapAttr` | 形状表达式仿射映射 |

```mlir
symbol.bind_symbolic_shape %arg0, [%0, %1],
  affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>

symbol.bind_symbolic_shape %out0, [%0, %1, %2],
  affine_map<()[s0, s1, s2] -> (s0, s1 * 2 + s2, 3)> : tensor<?x?x3xf32>
```

## 变换 Pass

| Pass | 构造函数 | 功能 |
|------|----------|------|
| `symbol-propagate` | `createPropagateSymbolPass()` | 传播符号信息 |
| `symbol-erase` | `createEraseSymbolPass()` | 擦除符号信息 |
| `symbol-to-encoding` | `createSymbolToEncodingPass()` | bind_symbolic_shape → tensor encoding |
| `encoding-to-symbol` | `createEncodingToSymbolPass()` | tensor encoding → bind_symbolic_shape |
| `symbol-unfold-symbolic-int` | `createUnfoldSymbolicIntPass()` | symbol.symbolic_int → tensor.dim |

## 与其他方言的交互

- **HFusion**：`hfusion-fold-symbolic-dim` / `hfusion-unfold-symbolic-dim` 在 Symbol 与 HFusion 之间转换（HFusion 方言见 [overview](../hfusion/overview.md)）。
- **Tensor Encoding**：`symbol-to-encoding` / `encoding-to-symbol` 在 Symbol 操作与 tensor encoding 属性两种动态形状表示间互转。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [SymbolOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/Symbol/IR/SymbolOps.td)
- [Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/Symbol/Transforms/Passes.h)
