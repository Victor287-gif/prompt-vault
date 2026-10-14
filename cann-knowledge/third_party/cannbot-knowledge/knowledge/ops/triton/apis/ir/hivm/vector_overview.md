---
type: API
title: HIVM 向量操作总览（PIPE_V、继承层次与操作分类）
description: HIVM 向量操作族总览：HIVM_VectorOp 基类与 Elementwise N-ary 模板、PIPE_V 归属、OTF 广播/转置
  traits、操作分类索引与 HFusion/linalg 降级关系。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 向量操作总览

HIVM 向量操作是 Ascend NPU 向量计算单元（AIV）的核心 IR 抽象，全部归属 **PIPE_V**（`HIVM_VectorOp` 基类以 `OpPipeTrait<"PIPE::PIPE_V">` 静态指定），实现 `DestinationStyleOpInterface`，同时支持 tensor 与 memref 语义。Triton 的 `tl` 算子经编译流水线降级为这些操作。方言定位见 [overview.md](overview.md)。

## 继承层次

```
HIVM_Op（前缀 hir.）
  └── HIVM_StructuredOp（HIVMStructuredOpInterface / MemoryEffects / Flatten / LibraryFunction）
        └── HIVM_VectorOp（PIPE_V, VectorCoreTypeTrait, AlwaysSpeculatable,
                            SinglePipeOpTrait, DestinationStyleOpInterface）
              ├── HIVM_ElementwiseNaryOp（AttrSizedOperandSegments, 同 rank,
              │     UniformReassociationFlatten, CollapsibleConsecutiveTargetDims, TransposableOTF）
              │     ├── HIVM_ElementwiseUnaryOp   （N=1）
              │     ├── HIVM_ElementwiseBinaryOp  （N=2）
              │     └── HIVM_ElementwiseTernaryOp （N=3）
              └── 独立向量操作（vbrc / vreduce / vtranspose / varange / vgather ...）
```

## 通用操作数结构（Elementwise N-ary）

| 参数 | 类型 | 说明 |
|------|------|------|
| `$src` | Variadic<AnyShaped/AnyType> | N 个输入；支持向量-向量或向量-标量（取决于是否带 ImplByScalarOpInterface） |
| `$dst` | Variadic<AnyShaped> | 输出（DestinationStyle） |
| `$temp_buffer` | Optional<AnyMemRef> | 临时缓冲区（ExtraBufferOpInterface，见 [extra_buffer](../memory_management/extra_buffer.md)） |
| `$transpose` | DenseI64ArrayAttr（默认 {}） | OTF 转置排列 |
| `$broadcast` | DenseI64ArrayAttr（默认 {}） | OTF 广播维度 |
| `$result` | Variadic<AnyRankedTensor> | tensor 语义结果 |

通用格式：`hir.<op> attr-dict ins(...) outs(...) [temp_buffer(...)] [broadcast = ...] [transpose = ...] [-> type]`

## 关键 Traits

| Trait | 作用 |
|-------|------|
| `ElementwiseNaryOpTrait<N>` | N 输入 1 输出、同 rank 的逐元约束 |
| `CommutativeOpTrait` | 输入可交换（vadd/vmul/vmax/vmin/vor/vand） |
| `OperElemTypeConstraints<[idx], [types]>` | 操作数元素类型白名单 |
| `VectorOnlyTrait<i>` / `ScalarOnlyTrait<i>` | 指定操作数必须为向量/标量 |
| `StaticMaxRankTrait<N>` / `InferMaxRankTrait` | 静态/推断最大 rank |
| `BroadcastableOTF` / `TransposableOTF` | 计算时随路广播/转置，不产生独立搬运 |

OTF 广播约束：broadcast 维度唯一、`0 <= d < rank(dst)`、src 在该维 size 为 1 或等于 dst。OTF 转置约束：必须是 range(rank) 的排列且**最后一维不可转置**。

## 操作分类索引

| 类别 | 操作 | 详解卡 |
|------|------|--------|
| 一元 | vexp/vabs/vln/vrelu/vrsqrt/vsqrt/vtanh/vsin/vcos/verf/vrec/vnot（+vcast 属转换类） | [vector_unary_ops.md](vector_unary_ops.md) |
| 二元 | vadd/vsub/vmul/vdiv/vmax/vmin/vor/vand/vxor/vmod/vmodui/vpow/vmulext | [vector_binary_ops.md](vector_binary_ops.md) |
| 三元 | vsel | [vector_ternary_ops.md](vector_ternary_ops.md) |
| 类型转换 | vcast | [vector_cast_ops.md](vector_cast_ops.md) |
| 比较 | vcmp | [vector_compare_ops.md](vector_compare_ops.md) |
| 移位 | vshl/vshr | [vector_shift_ops.md](vector_shift_ops.md) |
| 归约 | vreduce（sum/prod/max/min/any/all/xori/ori/andi/max\|min_with_index） | [vector_reduction_ops.md](vector_reduction_ops.md) |
| 数据搬移 | vbrc/vtranspose/vinterleave/vdeinterleave/vflip/vpad/vconcat/vgather | [vector_data_movement.md](vector_data_movement.md) |
| 累积与排序 | vcumsum/vcumprod/vsort | [vector_cumulative_sort.md](vector_cumulative_sort.md) |
| 特殊 | varange/vmulextended/vmulext | [vector_special_ops.md](vector_special_ops.md) |

## 与 HFusion / linalg 的降级关系

- 标准算术（vadd/vsub/vmul/vdiv/vabs/vexp/vln 等）→ `linalg` 对应操作（linalg.add/exp 等）。
- 特殊语义操作 → HFusion：vrelu/vnot → `hfusion.elemwise_unary`；vcast → `hfusion.cast`；vcmp → `hfusion.compare`；vreduce → `hfusion.reduce[_with_index]`；varange → `hfusion.arange`；vinterleave/vdeinterleave → `hfusion.interleave/deinterleave`。

## 标量降级预警

部分向量操作在硬件不支持的数据类型/操作组合下被降级为标量循环（`ImplByScalarOpInterface::shouldLowerToScalarLoops()`），性能损失 10x-100x 量级。最常见触发：i64 元素类型、SIMT VF 模式、整数大小比较、vmulext（恒降级）。降级条件全表与规避策略见 [vector_scalar_lowering.md](vector_scalar_lowering.md)；对应性能分析流程中的标量降级瓶颈条目见 [../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md)。

相关：[overview.md](overview.md)、[vector_binary_ops.md](vector_binary_ops.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMBase.td#L52-L58](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td#L52-L58)
- [HIVMBase.td#L60-L68](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td#L60-L68)
- [HIVMVectorOps.td#L24-L40](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L24-L40)
- [HIVMVectorOps.td#L46-L95](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L46-L95)
- [HIVMAttrs.td#L203-L236](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L236)
- [HIVMTraits.td#L53-L56](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L53-L56)
- [HIVMTraits.td#L76](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L76)
- [HIVMTraits.td#L90-L106](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L90-L106)
- [HIVMTraits.td#L80-L81](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L80-L81)
- [HIVMTraits.td#L87](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L87)
- [HIVMTraits.td#L64](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L64)
- [HIVMTraits.td#L67](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L67)
- [HIVMTraits.td#L130-L132](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L130-L132)
- [HIVMTraits.td#L153-L155](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L153-L155)
- [HIVMTraits.td#L183](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L183)
- [HIVMTraits.td#L191-L192](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L191-L192)
- [HIVMVectorOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td)
- [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HIVMBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
