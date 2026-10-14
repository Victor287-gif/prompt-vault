---
type: API
title: HIVM 归约运算（hivm.hir.vreduce）
description: hivm.hir.vreduce 归约操作语义：11 种归约类型（sum/prod/max/min/带索引/any/all/位归约）、reduce_dims
  属性、with_index 双输出、unsigned_src 与 tie_break_left、keepdims 式输出与各芯片降级差异。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---

# HIVM 归约运算（hivm.hir.vreduce）

`hir.vreduce`（HIVM_VectorOp，InferMaxRankTrait）沿指定维度归约，对应 Triton 的 `tl.sum / tl.max / tl.min / tl.argmax` 等。输出**保持输入 rank**，被归约维度 size 置 1（keepdims 式），与 `hir.vcumsum`（保形前缀和，[vector_cumulative_sort.md](vector_cumulative_sort.md)）区分。

## 语法

```mlir
%val, %idx = hivm.hir.vreduce <max_with_index>
             ins(%src : tensor<23x77xf32>)
             outs(%dst_val, %dst_idx : tensor<23x1xf32>, tensor<23x1xi32>)
             unsigned_src = false tie_break_left = true reduce_dims = [1]
             -> tensor<23x1xf32>, tensor<23x1xi32>
```

## 归约类型（HIVM_ReduceOpAttr，必选）

| 字面量 | 值 | 输出数 | 说明 |
|--------|---|--------|------|
| `sum` | 1 | 1 | 求和 |
| `prod` | 2 | 1 | 求积 |
| `max` / `min` | 3 / 4 | 1 | 极值 |
| `max_with_index` / `min_with_index` | 5 / 6 | **2** | 值 + I32 索引（对应 tl.argmax/argmin） |
| `any` / `all` | 7 / 8 | 1 | 逻辑或/与 |
| `xori` / `ori` / `andi` | 9 / 10 / 11 | 1 | 位归约 |
| `none` | 0 | 1 | 占位 |

## 属性与操作数

| 项 | 类型 | 说明 |
|----|------|------|
| `reduce_dims` | DenseI64ArrayAttr（必选，非空） | 归约维度，须在 [0, rank) 内 |
| `unsigned_src` | BoolAttr（必选） | 按无符号解释源（仅整数有意义） |
| `tie_break_left` | BoolAttr（可选） | 仅 with_index：true 取最左同值索引 |
| `temp_buffer` | Optional memref | 临时缓冲（[extra_buffer](../memory_management/extra_buffer.md)） |

元素类型：I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32。dst 为 Variadic（with_index 时两个：值张量 + I32 索引张量）。

## 降级与芯片差异

- → `hfusion.reduce` / `hfusion.reduce_with_index`。
- **Reg-based 实现**（A5：Ascend310B / 950）：仅 max/min_with_index 在访存对齐非法（缺 StridedLayoutAttr）时降级。
- **Mem-based 实现**（A2/A3：910B / 910_93）：sum/prod/max/min/xori 遇 i64 降级；max/min_with_index 遇 i64/i32/i16 恒降级、f16/f32/bf16 flatten rank>2 降级；any/all/ori/andi 从不降级。

完整降级规则与规避策略见 [vector_scalar_lowering.md](vector_scalar_lowering.md)。

相关：[vector_cumulative_sort.md](vector_cumulative_sort.md)、[vector_scalar_lowering.md](vector_scalar_lowering.md)、[multi_buffer](../memory_management/multi_buffer.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L1129-L1216](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1129-L1216)
- [HIVMAttrs.td#L596-L631](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L596-L631)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
