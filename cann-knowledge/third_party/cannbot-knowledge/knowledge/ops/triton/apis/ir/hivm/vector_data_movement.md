---
type: API
title: HIVM 数据搬移向量运算（vbrc/vtranspose/vinterleave/vdeinterleave/vflip/vpad/vconcat/vgather）
description: HIVM UB 内数据搬移操作族：vbrc 广播、vtranspose 转置、vinterleave/vdeinterleave 交错、vflip
  翻转、vpad 填充、vconcat 拼接、vgather 索引收集的语义、类型白名单与降级映射。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 数据搬移向量运算

UB 内的数据重排/复制操作族，对应 Triton 的 `tl.broadcast`、`tl.trans`、reshape 类变换。与 DMA 搬移（[dma_overview.md](dma_overview.md)）不同，本族不产生跨地址空间传输。

## vbrc（广播）

标量或向量沿 `broadcast_dims` 复制扩展。

- 标量输入：`broadcast_dims` 为空；向量输入：非空，维度 ∈ [0, rank(src))，src 在这些维 size 为 1。
- 类型：I8~UI64, F16, BF16, F32, F8E4M3FN, F8E5M2, I1；**I1 输出尾轴需 16 对齐**。

## vtranspose（转置）

`permutation` 必须是 range(rank) 的排列；`dim(dst, i) = dim(src, permutation[i])`。类型：AnyI8~AnyI32, I64, UI64, F16, BF16, F32, F8E4M3FN, F8E5M2。

## vinterleave / vdeinterleave（交错/解交错）

- `vinterleave`：N 个张量在最后一维交错合并，`interleave_channel_nums`（默认 2）必须等于 src 个数；rank 1；→ `hfusion.interleave`。
- `vdeinterleave`：逆操作，`channel_num`（默认 2）+ `index_mode`：`CHANNEL_0(0) / CHANNEL_1(1) / ALL_CHANNELS(999)`（取指定通道或全部）；→ `hfusion.deinterleave`。
- 类型：I8/I16/UI16/I32/UI32/I64/UI64, F16, F32, BF16, F8E4M3FN, F8E5M2（interleave 无 I8）。

## vflip（翻转）

`flip_axis`（I64Attr，必选）指定翻转轴；rank 1；类型 I8~UI64, F16, F32, BF16。

## vpad（填充）

`pad_value` + `low`/`high`（Variadic<Index>）与 `static_low`/`static_high`（DenseI64ArrayAttr）指定各维两侧填充量；带 NoLibraryFunctionTrait（内联展开）。

## vconcat（拼接）

`dim`（I64Attr）+ 变长 src；SameOperandsElementType；NoLibraryFunctionTrait；→ `tensor.concat`。

## vgather（索引收集）

`dst[i] = src[indices[i]]`（最后一轴）；data 类型 I1, I8, I16, UI16, I32, UI32, F16, BF16, F32, F8E4M3FN, F8E5M2；indices 为 I32；rank 1。**indices 越界行为未定义**，使用前需保证在界内。

## 选型提示

- OTF 广播（见 [vector_overview.md](vector_overview.md)）可在计算时随路完成，优先于显式 vbrc。
- vinterleave/vdeinterleave 在 i64 或 SIMT VF 模式下降级，见 [vector_scalar_lowering.md](vector_scalar_lowering.md)。
- 跨地址空间的 gather/scatter 用 DMA 族 [dma_gather_scatter.md](dma_gather_scatter.md)。

相关：[vector_overview.md](vector_overview.md)、[vector_cumulative_sort.md](vector_cumulative_sort.md)、[dma_gather_scatter.md](dma_gather_scatter.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L1056-L1123](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1056-L1123)
- [HIVMVectorOps.td#L1222-L1272](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1222-L1272)
- [HIVMVectorOps.td#L1336-L1377](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1336-L1377)
- [HIVMVectorOps.td#L1383-L1423](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1383-L1423)
- [HIVMVectorOps.td#L1429-L1457](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1429-L1457)
- [HIVMVectorOps.td#L1502-L1559](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1502-L1559)
- [HIVMVectorOps.td#L1565-L1604](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1565-L1604)
- [HIVMVectorOps.td#L1610-L1654](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1610-L1654)
- [HIVMVectorOps.td#L1056-L1654](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1056-L1654)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
