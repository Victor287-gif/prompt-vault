---
type: API
title: HIVM 累积与排序（hivm.hir.vcumsum / vcumprod / vsort）
description: hivm.hir.vcumsum/vcumprod 前缀和积与 vsort 排序语义：cum_dims 严格升序约束、reverse 方向、vsort
  仅尾轴且双输出（值+I32 索引）、与 vreduce 的保形差异。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 累积与排序

对应 Triton 的 `tl.cumsum / tl.cumprod / tl.sort`。本族操作**保形**（输出形状同输入），与压缩维度的 [vector_reduction_ops.md](vector_reduction_ops.md)（vreduce）互补。

## vcumsum / vcumprod（前缀和 / 前缀积）

```
正向（reverse=false）: dst[i] = Σ src[0 .. i]     （cumprod 为 Π）
反向（reverse=true）:  dst[i] = Σ src[i .. end]
```

| 项 | vcumsum | vcumprod |
|----|---------|----------|
| 类型 | I1, I8, I16, I32, I64, F16, F32, BF16 | 同左 |
| 最大 rank | 2 | **1** |
| 降级 | linalg.generic + arith.addi/addf | linalg.generic + arith.muli/mulf |

公共属性：

- `cum_dims`（ConfinedAttr DenseArrayStrictlySorted）：累积维度，**严格升序、无重复**。
- `reverse`（BoolAttr，必选）。

## vsort（排序）

- 类型：F16, F32, I32, I64；rank ≤ 1。
- **双输出**：排序后的值张量 + I32 索引张量（helper：getDstValue/getDstIndex/getSignedSortAxis）。
- `descending`（BoolAttr，默认 false）。
- `sort_axis`（默认 -1）：**仅支持尾轴**（硬件约束），其他轴需先 vtranspose（[vector_data_movement.md](vector_data_movement.md)）。
- 需要 Extra Buffer（[extra_buffer](../memory_management/extra_buffer.md)）。

## 与 vreduce 的区分

| | vcumsum / vcumprod | vreduce <sum/prod> |
|---|---|---|
| 输出形状 | 与输入相同（保形） | 归约维 size 置 1（压缩） |
| 语义 | 每个位置的前缀累积 | 整维归约为单值 |

## 降级预警

cum_dims 单维且（元素类型 i64 或该维是展平后最后一维）时 vcumsum/vcumprod 标量降级；规避：避免在尾轴做 i64 累积。详见 [vector_scalar_lowering.md](vector_scalar_lowering.md)。

相关：[vector_reduction_ops.md](vector_reduction_ops.md)、[vector_data_movement.md](vector_data_movement.md)、[vector_scalar_lowering.md](vector_scalar_lowering.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L1719-L1772](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1719-L1772)
- [HIVMVectorOps.td#L1660-L1713](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1660-L1713)
- [HIVMVectorOps.td#L1778-L1839](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1778-L1839)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
