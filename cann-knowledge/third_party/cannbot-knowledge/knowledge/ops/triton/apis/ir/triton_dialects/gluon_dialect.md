---
type: API
title: Gluon 实验性方言（gluon.set_auto_layout）
description: Gluon 方言是 Triton 编译器的实验性布局推断方言，仅含 gluon.set_auto_layout 一个操作——把张量的 auto
  布局编码替换为具体编码，将布局决策延迟到编译流程更晚阶段。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Gluon 实验性方言

Gluon 方言（方言名 `gluon`，C++ 命名空间 `::mlir::triton::gluon`）是 Triton 编译器中的实验性方言，核心目的是提供布局推断基础设施：允许编译器在后续 Pass 中自动把 `auto` 编码替换为最优的具体编码类型。方言当前仅含一个操作 `gluon.set_auto_layout`。

## gluon.set_auto_layout

将张量的 `auto` 布局编码替换为具体编码类型。输入输出形状与元素类型必须相同（`SameOperandsAndResultShape` / `SameOperandsAndResultElementType`），带验证器。

```mlir
%result = gluon.set_auto_layout %src : tensor<128xf32, #auto> -> tensor<128xf32, #blocked>
```

验证规则：输入编码必须是 `auto` 或可推断类型；输出编码必须是具体编码属性；形状与元素类型保持不变。

## 在编译流程中的位置

```
Triton IR (tt)                张量可能带 auto 编码（布局未定）
    │  布局推断
    ▼
Gluon IR (gluon)              set_auto_layout: auto → 具体编码
    ▼
TritonGPU IR (ttg)            布局已确定
    ▼
LLVM IR
```

## 设计动机

- **延迟布局决策**：布局确定推迟到编译流程更晚阶段，使优化 Pass 掌握更多信息。
- **布局推断分离**：把布局推断逻辑从 TritonToTritonGPU 转换中解耦。
- **实验平台**：为新的布局推断算法提供试验载体。

## 昇腾适配场景

- `auto` → `FractalSharedEncodingAttr`（Cube 操作数，编码语义见 [tritongpu_encodings.md](tritongpu_encodings.md)）。
- `auto` → `BlockedEncodingAttr`（Vector 操作数）。
- 在 SIMD / SIMT / MIX 三种执行模式下选择不同布局策略。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [GluonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Gluon/IR/GluonOps.td)
