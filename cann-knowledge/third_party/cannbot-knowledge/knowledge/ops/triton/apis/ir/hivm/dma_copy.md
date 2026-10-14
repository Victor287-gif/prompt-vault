---
type: API
title: HIVM DMA Copy（hivm.hir.copy）
description: hivm.hir.copy 操作语义：本地内存层级间拷贝（UB→UB / GM→L1 / UB→L1），Pipeline 按源目标地址空间动态确定，仅支持左侧
  Padding，UB→L1 限 Ascend950。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# HIVM DMA Copy（hivm.hir.copy）

`hivm.hir.copy` 在本地内存层级之间拷贝数据，Pipeline 归属**动态**取决于源/目标地址空间组合，由编译器在 bufferization 阶段自动生成，无直接 Python API 对应。支持随路 Padding（仅左侧）。

## MLIR 语法

```mlir
hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
              outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)

hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
              outs(%dst : memref<16x16xf16, #hivm.address_space<cbuf>>)   // UB→L1，仅 Ascend950
```

## 操作数与属性

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `src` / `dst` | TensorOrMemref | 是 | 源/目标缓冲区，地址空间须落在支持的通路组合内 |
| `pad_mode` | HIVM_PadModeAttr | 否 | `PadNull` / `PadFirstElem` / `PadValue` |
| `pad_value` | AnyType | 否 | 填充值，元素类型与 src/dst 相同 |
| `result_tensor` | Optional<AnyRankedTensor> | 结果 | Tensor 语义下的结果 |

元素类型约束：`i1/i8/ui8/i16/ui16/f16/bf16/i32/ui32/f32/ui64/i64/f8E4M3FN/f8E5M2`（比 load/store 多 `i1`）。最大 rank 3。

## 动态 Pipeline 与通路支持

| 源 | 目标 | Pipeline | 支持范围 |
|----|------|----------|---------|
| UB | UB | `PIPE_V` | 所有型号 |
| GM | L1 | `PIPE_MTE2` | 所有型号 |
| UB | L1 | `PIPE_MTE3` | 仅 Ascend950 系列（950 的 Vector 核心可直接写 L1） |
| L0C | GM | `PIPE_FIX` | 定义内组合，实际 L0C 写出通常用 fixpipe |

## 验证器约束

1. src 与 dst 元素类型、rank 一致；未设 pad_mode 时形状兼容。
2. `pad_mode = PadValue` 时 `pad_value` 必填且类型一致。
3. memref 语义下仅允许 UB→UB、GM→L1、UB→L1（Ascend950）三种通路。
4. **仅支持左侧 Padding**（load 支持左右双侧），属硬件 DMA 引擎限制。

## 与 load/store 对比

| 操作 | 通路 | Pipeline | Padding | 布局转换 |
|------|------|----------|---------|---------|
| [hir.load](dma_load.md) | GM→UB | MTE2 | 左+右 | 无 |
| [hir.store](dma_store.md) | UB→GM | MTE3 | 不支持 | 无 |
| hir.copy | 本地↔本地 | 动态 | 仅左 | 无 |

## Lowering

库函数命名 `copy_{src_space}_to_{dst_space}_{rank}d_{datatype}`。IR 中出现 UB→UB copy 通常意味着 bufferization 引入了中间缓冲，性能分析时可结合 [bufferization](../memory_management/bufferization.md) 判断是否冗余拷贝。

相关：[dma_load.md](dma_load.md)、[dma_store.md](dma_store.md)、[dma_padding_quantization.md](dma_padding_quantization.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMDMAOps.td#L192-L244](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L192-L244)
- [HIVMDMAOps.cpp#L607-L632](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L607-L632)
- [HIVMDMAOps.cpp#L552-L605](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L552-L605)
- [HIVMDMAOps.cpp#L440-L448](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L440-L448)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
