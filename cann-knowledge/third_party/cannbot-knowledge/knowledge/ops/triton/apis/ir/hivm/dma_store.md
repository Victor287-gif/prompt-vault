---
type: API
title: HIVM DMA Store（hivm.hir.store）
description: hivm.hir.store 操作语义：UB→GM 的 MTE3 数据写回，atomic_kind 原子属性（ADD/MAX/MIN 硬件实现，AND/OR/XOR/CAS/XCHG
  软件实现）、验证约束与 lowering 规则。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.implementation.atomic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM DMA Store（hivm.hir.store）

`hivm.hir.store` 把数据从统一缓冲区（UB）写回全局内存（GM），映射到 MTE3 Pipeline，是 kernel 数据写回的主要方式。当前仅支持 UB→GM 一条通路（L1→GM 用 [hivm.hir.nz2nd](dma_nz2nd.md)，L0C 写出用 [hivm.hir.fixpipe](dma_fixpipe.md)）。对应 Triton 前端的 `tl.store`；支持 `atomic_kind` 属性实现原子存储语义。

## MLIR 语法

```mlir
hivm.hir.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
               outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)

hivm.hir.store ins(%src : memref<16x16xf32, #hivm.address_space<ub>>)
               outs(%dst : memref<16x16xf32, #hivm.address_space<gm>>)
               atomic = #hivm.atomic_kind<add>
```

## 操作数与属性

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `src` | TensorOrMemref | 是 | 源缓冲区，memref 语义下必须为 UB 地址空间 |
| `dst` | TensorOrMemref | 是 | 目标缓冲区，memref 语义下必须为 GM 地址空间 |
| `atomic_kind` | HIVM_AtomicKindAttr | 否 | 原子操作类型（见下表） |
| `result_tensor` | Optional<AnyRankedTensor> | 结果 | Tensor 语义下的结果 |

元素类型约束同 `hir.load`：`i8/ui8/i16/ui16/f16/bf16/i32/ui32/f32/ui64/i64/f8E4M3FN/f8E5M2`，最大 rank 3。

## AtomicKind 枚举（#hivm.atomic_kind<...>）

| 字面量 | 值 | 语义 | 实现 |
|--------|---|------|------|
| `none` | 0 | 非原子 | — |
| `add` | 1 | 原子加 | 硬件（isHWAtomic） |
| `max` | 2 | 原子最大值 | 硬件 |
| `min` | 3 | 原子最小值 | 硬件 |
| `and` / `or` / `xor` | 4/5/6 | 原子位运算 | 软件锁（isSWAtomic） |
| `cas` / `xchg` | 7/8 | 比较交换 / 交换 | 软件锁 |

硬件原子（add/max/min）由 DMA 引擎直接支持，性能高；软件原子需锁机制，开销大，性能敏感场景应优先硬件原子。独立原子指令（不依赖 set_atomic、可返回旧值、硬件实现）见 [dma_atomic.md](dma_atomic.md)。

## 验证器约束

1. src 与 dst 元素类型、rank 必须一致。
2. memref 语义下 src 必须为 UB、dst 必须为 GM。
3. Tensor 语义下 result_tensor 元素类型、rank 与 dst 一致。

## Lowering 与映射

- `tt.store` → `hivm.hir.store`；原子 store → `hivm.hir.store atomic = #hivm.atomic_kind<...>`。
- 最终 lowering 为库函数 `store_ubuf_to_gm_{rank}d_{datatype}`。
- 可用 `hir.set_atomic kind=#hivm.atomic_kind<add>` 设置全局原子模式使后续 store 自动带原子语义，用 `kind<NONE>` 重置（`hir.set_atomic` 属 Custom 操作族）。

相关：[dma_load.md](dma_load.md)、[dma_atomic.md](dma_atomic.md)、[dma_overview.md](dma_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMDMAOps.td#L138-L190](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L138-L190)
- [HIVMAttrs.td#L637-L664](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L637-L664)
- [HIVMDMAOps.cpp#L348-L376](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L348-L376)
- [HIVMDMAOps.cpp#L378-L394](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L378-L394)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
