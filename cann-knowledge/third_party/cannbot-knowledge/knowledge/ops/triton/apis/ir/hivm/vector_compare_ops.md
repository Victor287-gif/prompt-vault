---
type: API
title: HIVM 比较运算（hivm.hir.vcmp）
description: hivm.hir.vcmp 比较操作语义：六种 compare_mode、输入 F16/F32/I8/I16/I32/I64 输出 I1/I8、整数比较标量降级矩阵（仅
  i32 EQ/NE 走向量路径）、i1→i8 分解预处理与 hfusion.compare 降级。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 比较运算（hivm.hir.vcmp）

`hir.vcmp`（二元族，rank ≤ 1）逐元比较两个操作数，输出布尔掩码，对应 Triton 的 `tl.x < tl.y` 等。输入输出类型不同（无二元族常规的 SameOperandsElementType，也不需要 temp_buffer）。

## 语法

```mlir
%r = hivm.hir.vcmp <lt> ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
                        outs(%dst : tensor<16xi8>) -> tensor<16xi8>
```

## 属性与类型

- `compare_mode`（CompareMode，默认 EQ）：`EQ(0) / NE(1) / LT(2) / GT(3) / GE(4) / LE(5)`。
- 输入：F16, F32, I8, I16, I32, I64（src[0] VectorOnly）。
- 输出：I1, I8。
- src[1] 可为标量（带 ImplByScalarOpInterface）。

## 标量降级矩阵（关键）

| 元素类型 | 向量路径 | 标量降级 |
|---------|---------|---------|
| f16 / f32 | 全部六种模式 | — |
| i32 | **仅 EQ / NE** | LT / GT / LE / GE |
| i8 / i16 / i64 | 无 | 全部模式 |

整数大小比较（含 i32 的 LT/GT/LE/GE）全部标量降级；整数比较仅 i32 EQ/NE 走向量路径。标量降级性能损失 10x-100x 量级，见 [vector_scalar_lowering.md](vector_scalar_lowering.md)。

## i1 输出的特殊处理（HIVMDecomposeOp）

Store/计算不支持 i1 元素，`HIVMDecomposeOp` 预处理 pass 会把 i1 输出的 vcmp 改写为：i8 临时缓冲 vcmp → VCastOp 转回目标类型。标量降级路径（LowerToLoops）则使用 `arith.CmpIOp` + `arith.ExtUIOp` 实现。

## 降级映射

vcmp → `hfusion.compare {compare_fn = veq/vne/vlt/vle/vgt/vge}`。

## 优化建议

1. **优先用浮点比较**：f16/f32 六种模式全向量。
2. **整数比较先转浮点**：`vcast(i32→f32)` 后比较，通常快于 i32 大小比较的标量降级；只有 i32 EQ/NE 可直接走整数向量路径。
3. vcmp→vsel 链（[vector_ternary_ops.md](vector_ternary_ops.md)）受 vcmp 降级连累，整条链一起变慢。

相关：[vector_ternary_ops.md](vector_ternary_ops.md)、[vector_scalar_lowering.md](vector_scalar_lowering.md)、[../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L944-L979](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L944-L979)
- [HIVMAttrs.td#L452-L473](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L452-L473)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
