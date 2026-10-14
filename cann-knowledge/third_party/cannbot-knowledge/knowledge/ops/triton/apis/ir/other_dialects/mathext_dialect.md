---
type: API
title: MathExt 方言（math_ext.ilogb / ldep / divfhp）
description: MathExt 方言扩展标准 Math 方言，提供 NPU 特有的数学操作：math_ext.ilogb（取浮点指数部分）、math_ext.ldep（ldexp
  语义）、math_ext.divfhp（恢复最后 ULP 精度的高精度除法），均支持 fastmath 标志与常量折叠。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# MathExt 方言

MathExt 方言（方言名 `mathExt`，C++ 命名空间 `::mlir::mathExt`）扩展 MLIR 标准 Math 方言，提供 NPU 设备特有的数学操作。所有操作继承 `MathExt_Op` 基类，自动带 `Pure`、`VectorUnrollOpInterface`、`ElementwiseMappable` traits，均可向量化展开并参与 elementwise 映射。

## 操作基类

| 基类 | 说明 |
|------|------|
| `MathExt_FloatUnaryOp` | 浮点一元操作基类，`SameOperandsAndResultType`，支持 `ArithFastMathInterface`（`fastmath` 属性默认 none） |
| `MathExt_FloatBinaryOp` | 浮点二元操作基类，lhs/rhs/result 同类型，同样支持 fastmath |

## 操作定义

### math_ext.ilogb

计算浮点数的指数部分：`ilogb(x) = floor(log2(abs(x)))`。继承 `MathExt_FloatUnaryOp`，`hasFolder = 1`（支持常量折叠）。

```mlir
%result = math_ext.ilogb %val : f32
%result = math_ext.ilogb %val fastmath <fast> : f32
```

### math_ext.ldep

计算 `ldep(x) = x * (ilogb(x) + 1)^(-1)`，对应 ldexp 语义（TableGen mnemonic 为 `ldep`）。继承 `MathExt_FloatBinaryOp`，支持常量折叠。

```mlir
%result = math_ext.ldep %x, %y : f32
%result = math_ext.ldep %x, %y fastmath <fast> : f32
```

### math_ext.divfhp

高精度浮点除法，恢复最后 ULP（Unit in the Last Place）的精度。继承 `MathExt_FloatBinaryOp`，支持常量折叠。

```mlir
%result = math_ext.divfhp %x, %y : f32
%result = math_ext.divfhp %x, %y fastmath <fast> : f32
```

## 与 HFusion 方言的关系

HFusion 方言依赖 `mathExt::MathExtDialect`，其枚举函数与 MathExt 操作一一对应（HFusion 总览见 [overview](../hfusion/overview.md)）：

| MathExt 操作 | HFusion 枚举函数 | 说明 |
|-------------|------------------|------|
| `math_ext.ilogb` | `ilogb`（UnaryFn） | 指数部分 |
| `math_ext.ldep` | `ldexp`（BinaryFn） | ldexp 函数 |
| `math_ext.divfhp` | `divfhp`（BinaryFn） | 高精度除法 |

`hfusion-legalize-*` Pass 可将 HFusion 的枚举函数调用转换为对应 MathExt 操作。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [MathExtOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtOps.td)
- [MathExtBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/MathExt/IR/MathExtBase.td)
