---
type: API
title: Annotation 方言（annotation.mark）
description: Annotation 方言提供 IR 值标注机制：annotation.mark 为任意 IR 值附加静态属性或动态键值对，effects
  属性声明内存效果模式；标注信息驱动 Bufferization 决策、缓存 IO（hfusion-cache-io）与多缓冲标记（hivm.multi_buffer）等后续
  Pass。
tags: [task.lookup, topic.compilation, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Annotation 方言

Annotation 方言（方言名 `annotation`，C++ 命名空间 `::mlir::annotation`）提供 IR 值标注机制：为任意 IR 值附加键值对属性，标注信息指导后续编译 Pass 的行为（Bufferization 决策、缓存策略、多缓冲标记等）。

## annotation.mark

为 IR 值附加属性标注，值可以是静态属性或动态 IR 值。实现 `MemoryEffectsOpInterface`（自定义 getEffects）。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `AnyType` | 被标注的值 |
| `effects` | `StrArrayAttr`（默认 `["write"]`） | 内存效果模式 |
| `values` | `Variadic<AnyType>` | 动态属性值 |
| `keys` | `OptionalAttr<StrArrayAttr>` | 属性键名 |

```mlir
// 动态键值对标注
annotation.mark %target keys = ["key"] values = [%val] : f64

// 静态字典属性标注
annotation.mark %target {key : val} : f64
```

常用查询接口：`isAnnotatedBy(key)`、`isAnnotatedByStaticAttr(key)`、`isAnnotatedByDynamicAttr(key)`、`getStaticAttrValue(key)`、`getDynamicAttrValue(key)`、`getMixedAttrValue(key)`、`getAttrNum()` / `isAttrEmpty()`（后两者统计均不含 effects）。

## EffectMode 枚举

| 枚举值 | 值 | 助记符 | 说明 |
|--------|---|--------|------|
| `Write` | 0 | `write` | 写效果（默认） |
| `Read` | 1 | `read` | 读效果 |

`effects` 默认 `["write"]`，表示标注操作对被标注值有写效果；标注仅用于信息传递而不影响内存行为时设为 `["read"]`。

## 承载属性语义

`annotation.mark` 是编译 Pass 间传递意图的通用载体，两个最重要的承载场景：

1. **多缓冲标记**：`hivm-mark-multi-buffer` Pass 向缓冲区写入 `annotation.mark %buf {hivm.multi_buffer = N : i32}`（i32 即实例数，2=双缓冲、4=四缓冲），`hivm-enable-multi-buffer` 随后据此把单缓冲分配转换为多缓冲分配。完整机制与 PlanMemory 后的 `pointer_cast` 多偏移形态见 [multi_buffer](../memory_management/multi_buffer.md)。
2. **缓存 IO**：`hfusion-cache-io` Pass 识别需要缓存的 `annotation.mark` 标注，插入 GM↔UB 数据搬移操作，并更新标注的 `effects` 属性。

## 与 Bufferization 的交互

- **缓存标注**：标记需要缓存的输入/输出。
- **别名信息**：`effects` 属性告知 Bufferization 别名关系。
- **Buffer 决策**：标注信息影响 Buffer 分配与布局决策。

## 典型使用场景

```mlir
// 标记缓存需求
%cached = annotation.mark %input keys = ["cache"] values = [%input] : tensor<128x256xf32>

// 标记 Bufferization 约束（无动态值）
%marked = annotation.mark %val keys = ["no_alias"] : memref<?xf32>

// 标记计算核心类型
%marked = annotation.mark %result keys = ["tcore_type"] values = [%cube_val] : tensor<128x256xf32>
```

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [AnnotationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationOps.td)
- [AnnotationAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationAttrs.td)
- [AnnotationBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/Annotation/IR/AnnotationBase.td)
