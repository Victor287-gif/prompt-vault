---
type: API
title: HIVM 类型转换（hivm.hir.vcast）
description: hivm.hir.vcast 类型转换操作语义：round_mode 七种舍入模式与 cast 三种模式、浮点↔整数转换支持矩阵、odd
  仅 f32→f16、低→高精度只需 rint、hfusion.cast 降级。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 类型转换（hivm.hir.vcast）

`hir.vcast` 在 UB 内做逐元类型转换，对应 Triton 的张量 `.to(dtype)`。它是一元族中**唯一不要求输入输出元素类型相同**的操作（无 SameOperandsElementType，自带 hasVerifier）。最大 rank 2，输入 VectorOnly，需要 Extra Buffer（见 [extra_buffer](../memory_management/extra_buffer.md)），支持 OTF 广播。

## 语法

```mlir
%r = hivm.hir.vcast ins(%src : tensor<16x32xf32>) outs(%dst : tensor<16x32xf16>)
                    round_mode = #hivm.round_mode<rint> -> tensor<16x32xf16>
```

## 属性

### round_mode（RoundMode，默认 rint）

| 字面量 | 值 | 语义 |
|--------|---|------|
| `RINT` | 0 | 就近取偶（默认） |
| `ROUND` | 1 | 四舍五入（远离零） |
| `FLOOR` | 2 | 向下取整 |
| `CEIL` | 3 | 向上取整 |
| `TRUNC` | 4 | 向零截断 |
| `ODD` | 5 | Von Neumann 舍入 |
| `TRUNCWITHOVERFLOW` | 6 | 截断并保留溢出行为 |

### cast（CastMode，默认 cast_signed）

| 字面量 | 值 | 语义 |
|--------|---|------|
| `cast_signed` | 0 | 有符号语义转换（默认） |
| `cast_unsigned` | 1 | 无符号语义转换 |
| `bitcast` | 2 | 位模式重解释 |

相关 UnsignedMode 枚举：SI2SI / SI2UI / UI2SI / UI2UI。

## 转换支持矩阵（节选要点）

| 源→目标 | 支持的 round_mode |
|---------|------------------|
| f32→f16 / i64 / i32 / i16 / bf16 | rint, round, floor, ceil, trunc（f32→f16 另支持 odd） |
| f16→f32；bf16→f32 | 仅 rint |
| f16→i32 / i16 / i8 / ui8 / i4；bf16→i32 | 五种模式（rint/round/floor/ceil/trunc） |
| i8/ui8→f16；i8→i1；i16→f32；i32→i64/i16；i64→i32；i4→f16；i1→f16/f32 | 仅 rint |
| i16→f16；i32→f32；i64→f32 | 五种模式 |

规律：

- **odd 仅 f32→f16** 支持。
- **低精度→高精度（如 f16→f32）只需 rint**——无舍入歧义。
- 浮点→整数通常支持全部五种舍入。

## 降级与辨析

- vcast → `hfusion.cast {round_mode = ...}`。
- `vcast(cast = bitcast)` 与独立的 `hir.bitcast` 不同：vcast 的 bitcast 是元素级位重解释，独立 bitcast 处理 memref 视图级重解释。

相关：[vector_unary_ops.md](vector_unary_ops.md)、[vector_scalar_lowering.md](vector_scalar_lowering.md)、[../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L420-L500](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L420-L500)
- [HIVMAttrs.td#L378-L406](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L378-L406)
- [HIVMAttrs.td#L431-L446](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L431-L446)
- [HIVMAttrs.td#L408-L425](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L408-L425)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
