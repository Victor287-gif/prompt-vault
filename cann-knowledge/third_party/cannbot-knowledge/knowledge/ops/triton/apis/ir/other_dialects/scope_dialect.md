---
type: API
title: Scope 方言（scope.scope / outline-scope / inline-scope）
description: Scope 方言提供区域作用域抽象：scope.scope 将操作分组为可携带 tcore_type（CUBE/VECTOR）等属性的逻辑作用域，支持
  outline-scope（抽为独立 func.func）与 inline-scope（提升到父区域）两种变换，no_inline 属性控制内联行为。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Scope 方言

Scope 方言（方言名 `scope`，C++ 命名空间 `::mlir::scope`）提供区域作用域抽象，把一组操作打包为逻辑作用域整体执行。作用域可携带 `tcore_type` 等属性标注计算核心类型（CUBE/VECTOR），是 VF（Vector Function）划分和 Cube/Vector 分区编译的基础设施。核心类型枚举见 [overview](../hivm/overview.md)。

## 操作定义

### scope.scope

区域作用域，内部操作作为一个整体执行；区域为单块（`SingleBlock`），隐式终止符为 `scope.return`，实现 `RegionBranchOpInterface`。

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `no_inline` | `UnitAttr` | 标记该作用域不可被 inline-scope 内联 |
| `results` | `Variadic<AnyType>` | 作用域返回值 |
| `$region` | `SizedRegion<1>` | 作用域区域 |

```mlir
scope.scope : () -> () {
  scope.return
} {tcore_type = #hivm.tcore_type<CUBE>}

scope.scope : () -> () {
  scope.return
} {no_inline}
```

### scope.return

`scope.scope` 区域的终止操作（`Terminator` + `ReturnLike` + `Pure`），返回作用域计算结果：

```mlir
scope.return
scope.return %val1, %val2 : f32, i64
```

## 变换 Pass

### outline-scope（ModuleOp）

将每个 `scope.scope` 抽为独立的 `func.func`，作用域属性（如 `tcore_type`）转移到新函数上，原位置替换为 `call`：

```mlir
// 输入：func.func @test() 内含两个带 tcore_type 的 scope.scope
// 输出：
func.func @test_scope_0() attributes {tcore_type = #hivm.tcore_type<CUBE>} { ... }
func.func @test_scope_1() attributes {tcore_type = #hivm.tcore_type<VECTOR>} { ... }
func.func @test() {
  call @test_scope_0() : () -> ()
  call @test_scope_1() : () -> ()
}
```

### inline-scope（ModuleOp）

将 `scope.scope` 区域内操作提升到父区域；标记了 `no_inline` 的作用域保持原样。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `force-inline` | bool | false | 强制内联，忽略 `no_inline` |

## 典型使用场景

- **计算核心类型标注**：`tcore_type` 属性标记作用域在 CUBE 或 VECTOR 核心上执行，是 Mix 模式分区编译的输入。
- **VF 划分**：计算图划分为多个 Scope，每个 Scope 对应一个向量化函数。
- **编译流程控制**：`no_inline` 确保关键作用域在编译流程中保持独立边界，不被提前内联打散。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [ScopeOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/Scope/IR/ScopeOps.td)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/Scope/Transforms/Passes.td)
