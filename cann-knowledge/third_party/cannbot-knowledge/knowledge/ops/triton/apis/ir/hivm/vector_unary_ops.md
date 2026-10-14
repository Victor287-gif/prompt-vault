---
type: API
title: HIVM 一元向量运算（hir.vexp/vabs/vln/vrelu/vrsqrt/vsqrt/vtanh/vsin/vcos/verf/vrec/vnot）
description: HIVM 一元逐元向量操作族语义与约束：12 个 op 的元素类型白名单、最大 rank、OTF 广播/转置、temp_buffer 与
  NoLibraryFunctionTrait 差异，及 linalg/HFusion 降级映射。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 一元向量运算

一元操作继承 `HIVM_ElementwiseUnaryOp`（`ElementwiseNaryOpTrait<1>`：1 输入 1 输出、同 rank），对输入向量逐元素独立运算。分三类：数学函数（vexp/vln/vsqrt/vrsqrt/vrec/vsin/vcos/vtanh/verf）、数值操作（vabs/vrelu）、位操作（vnot）。对应 Triton 的 `tl.math.exp/log/rsqrt/sqrt/tanh/sin/cos/erf`、`tl.abs` 等。通用操作数结构与 OTF 规则见 [vector_overview.md](vector_overview.md)。

## 语法模式

```mlir
%r = hivm.hir.vexp ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
%r = hivm.hir.vabs ins(%src : tensor<1x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) broadcast = [0] -> tensor<5x?x10xf32>
%r = hivm.hir.vrelu ins(%src : tensor<?x5x10xf32>) outs(%dst : tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>
hivm.hir.vnot ins(%src : memref<5x1x10xi32>) outs(%dst : memref<5x?x10xi32>) broadcast = [1]
```

## 操作约束汇总

| 操作 | 元素类型 | 最大 rank | OTF 广播/转置 | Extra Buffer | 备注 |
|------|---------|-----------|--------------|--------------|------|
| `hir.vexp` | F16, F32 | 3 | 是/是 | 是 | — |
| `hir.vabs` | F16, F32, I8, I16, I32, I64 | 3 | 是/是 | 是 | 支持标量实现与向量化接口 |
| `hir.vln` | F16, F32 | 3 | 是/是 | 是 | — |
| `hir.vrelu` | F16, F32, I32 | 3 | 是/是 | 是 | — |
| `hir.vrsqrt` | F16, F32 | 3 | 是/是 | 是 | — |
| `hir.vsqrt` | F16, F32 | 3 | 是/是 | 是 | — |
| `hir.vtanh` | 浮点（SameOperandsElementType 隐式） | — | — | — | NoLibraryFunctionTrait，内联实现 |
| `hir.vsin` | 浮点 | — | — | — | 同上 |
| `hir.vcos` | 浮点 | — | — | — | 同上 |
| `hir.verf` | 浮点 | — | — | — | 同上 |
| `hir.vrec` | F16, F32 | 3 | 是/是 | 是 | 逐元倒数 |
| `hir.vnot` | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | 3 | 是/是 | 是 | 按位取反，对浮点取其位模式 |

vtanh/vsin/vcos/verf 无显式 `OperElemTypeConstraints`，类型约束由 `SameOperandsElementType` 保证，具体支持由硬件内联实现决定；它们不走预编译库函数（`NoLibraryFunctionTrait`）。

## 通用验证规则

1. 输入输出元素类型一致（`SameOperandsElementType`）、rank 一致。
2. `$src[0]` 必须为向量（`VectorOnlyTrait<0>`）。
3. OTF 广播：维度唯一、`0 <= d < rank(dst)`、src 在广播维 size 为 1 或等于 dst。
4. OTF 转置：必须是 range(rank) 的排列且最后一维不可转置。
5. `temp_buffer`：vexp/vln/vrsqrt 等需要额外临时空间保存中间结果（ExtraBufferOpInterface，机制见 [extra_buffer](../memory_management/extra_buffer.md)）。

## 降级映射

| HIVM | 目标 | HIVM | 目标 |
|------|------|------|------|
| vexp | linalg.exp | vsqrt | linalg.sqrt |
| vabs | linalg.abs | vtanh | linalg.tanh |
| vln | linalg.log | vrec | linalg.reciprocal |
| vrsqrt | linalg.rsqrt | verf | linalg.erf |
| vrelu | hfusion.elemwise_unary {fun = relu} | vnot | hfusion.elemwise_unary {fun = vnot} |

## 性能提示

- vabs 在 i64 或 SIMT VF 模式下会标量降级，见 [vector_scalar_lowering.md](vector_scalar_lowering.md)。
- 类型转换不属于本族：`hir.vcast` 见 [vector_cast_ops.md](vector_cast_ops.md)（唯一不要求输入输出同元素类型的一元操作，最大 rank 2）。

相关：[vector_binary_ops.md](vector_binary_ops.md)、[vector_overview.md](vector_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L101-L103](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L101-L103)
- [HIVMVectorOps.td#L105-L136](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L105-L136)
- [HIVMVectorOps.td#L138-L171](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L138-L171)
- [HIVMVectorOps.td#L173-L204](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L173-L204)
- [HIVMVectorOps.td#L206-L237](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L206-L237)
- [HIVMVectorOps.td#L239-L270](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L239-L270)
- [HIVMVectorOps.td#L272-L303](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L272-L303)
- [HIVMVectorOps.td#L305-L315](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L305-L315)
- [HIVMVectorOps.td#L317-L327](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L317-L327)
- [HIVMVectorOps.td#L329-L339](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L329-L339)
- [HIVMVectorOps.td#L341-L351](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L341-L351)
- [HIVMVectorOps.td#L353-L384](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L353-L384)
- [HIVMVectorOps.td#L386-L418](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L386-L418)
- [HIVMVectorOps.td#L105-L418](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L105-L418)
- [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
