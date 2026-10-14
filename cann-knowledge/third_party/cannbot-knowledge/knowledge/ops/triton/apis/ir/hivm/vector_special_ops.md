---
type: API
title: HIVM 特殊向量运算（hivm.hir.varange / vmulextended / vmulext）
description: hivm.hir.varange 序列生成、vmulextended 双输出乘扩展、vmulext 高 32 位乘积的语义与差异：varange
  的 offset/strides 语义、vmulextended 仅 I16 双输出、vmulext 恒标量降级。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 特殊向量运算

## varange（序列生成）

生成等差序列张量，对应 `tl.arange` 的多维推广。

```
value[i0, i1, ..., in] = offset + Σ_k (i_k * strides[k])
```

- `offset`（Optional<Index>，默认 0）；`strides`（Variadic<Index>，≥1，个数应等于 rank）。
- 类型：I16, I32, F16, F32, I64；rank ≤ 3。
- 降级 → `hfusion.arange`。

示例：`offset = 1, strides = [1, 2]`，shape 2x4 → `[[1,3,5,7],[2,4,6,8]]`。

## vmulextended（I16 乘法扩展，双输出）

- **仅 I16**，rank ≤ 1；2 输入 2 输出。
- 输出 1：32 位乘积的**高 16 位**；输出 2：**低 16 位**。
- HIVM_VectorOp 基类，非 ElementwiseBinaryOp。

## vmulext（I32 乘法取高位，恒降级）

- **仅 I32**，rank ≤ 3；ElementwiseBinaryOp（2 输入 1 输出）。
- 输出 64 位乘积的**高 32 位**。
- 虽带 ImplByScalarOpInterface，实践中**始终标量降级**（见 [vector_scalar_lowering.md](vector_scalar_lowering.md)），性能差，应避免在关键路径使用。

## 对比

| | vmulextended | vmulext |
|---|---|---|
| 元素类型 | I16 | I32 |
| 输出 | 2（高16 + 低16） | 1（高32） |
| 基类 | HIVM_VectorOp | ElementwiseBinaryOp |
| 最大 rank | 1 | 3 |
| 降级 | 正常向量路径 | 恒标量降级 |

优选 vmulextended；需要 I32 高位乘积时尽量改写算法规避 vmulext。

相关：[vector_binary_ops.md](vector_binary_ops.md)、[vector_scalar_lowering.md](vector_scalar_lowering.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L1278-L1330](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1278-L1330)
- [HIVMVectorOps.td#L1463-L1496](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1463-L1496)
- [HIVMVectorOps.td#L578-L594](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L578-L594)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
