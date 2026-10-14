---
type: API
title: HIVM DMA 操作总览（内存层级数据通路与 Pipeline 归属）
description: HIVM DMA 操作族总览：GM/L1/UB/L0C 间的数据通路、hir.load/store/copy/fixpipe/nd2nz/nz2nd/l12ub
  的 Pipeline 归属表、copy 动态 Pipeline 规则与同步模式。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# HIVM DMA 操作总览

DMA 操作负责在 Ascend NPU 各内存层级之间搬运数据，直接映射到硬件 DMA 引擎并按 Pipeline 执行，是 HIVM IR 中最高频的操作类别。内存层级的容量与对齐数值见 [memory_hierarchy](../architecture/memory_hierarchy.md)。

## 数据通路

```
                 GM（片外全局内存）
                /  |            \
       MTE2 load   | MTE2 nd2nz  \
              v    v              v
              UB   L1 <-----------+
              |    ^  \ MTE3 nz2nd
   PIPE_V copy|    |MTE1 l12ub
              v    |
              UB --+-- MTE3 store --> GM

L0C（Cube 累加器）-- FIX fixpipe --> GM / UB / L1（支持随路量化/激活）
```

## Pipeline 归属表

| 操作 | 数据通路 | Pipeline | 布局转换 | 随路功能 | 详解 |
|------|---------|----------|---------|---------|------|
| `hir.load` | GM → UB | MTE2 | 无 | Padding（左+右）、Eviction、init_out_buffer | [dma_load.md](dma_load.md) |
| `hir.store` | UB → GM | MTE3 | 无 | 原子操作（atomic_kind） | [dma_store.md](dma_store.md) |
| `hir.copy` | UB→UB / GM→L1 / UB→L1 | 动态 | 无 | Padding（仅左侧） | [dma_copy.md](dma_copy.md) |
| `hir.fixpipe` | L0C → GM/UB/L1 | FIX | NZ2ND / NZ2DN / NZ2NZ | 量化、ReLU、双目标 | [dma_fixpipe.md](dma_fixpipe.md) |
| `hir.nd2nz` | GM → L1 | MTE2 | ND → NZ | init_out_buffer | [dma_nd2nz.md](dma_nd2nz.md) |
| `hir.nz2nd` | L1 → GM | MTE3 | NZ → ND | 无 | [dma_nz2nd.md](dma_nz2nd.md) |
| `hir.l12ub` | L1 → UB | MTE1 | NZ → ND | 无 | 本表 |
| `hir.atomic_cas` / `hir.atomic_xchg` | GM ↔ UB | 无固定 | 无 | 硬件原子指令 | [dma_atomic.md](dma_atomic.md) |

稀疏与间接访存（gather_load / scatter_store / indirect_load / indirect_store）见 [dma_gather_scatter.md](dma_gather_scatter.md) 与 [dma_indirect_access.md](dma_indirect_access.md)。

## hir.copy 的动态 Pipeline

`hir.copy` 的 Pipeline 由源/目标地址空间组合决定：

| 源 | 目标 | Pipeline |
|----|------|----------|
| UB | UB | `PIPE_V` |
| L0C | GM | `PIPE_FIX` |
| GM | L1 | `PIPE_MTE2` |
| UB | L1 | `PIPE_MTE3` |

## 通路支持矩阵（按芯片）

| 通路 | 操作 | 支持范围 |
|------|------|---------|
| GM→UB / GM→L1 / UB→GM / L1→GM / L1→UB / UB→UB / GM→L1(copy) / L0C→GM / L0C→L1 | 对应上表操作 | 所有 Ascend 型号 |
| UB→L1（copy） | `hir.copy` | 仅 Ascend950 系列 |
| L0C→UB | `hir.fixpipe` | 仅 Ascend950 系列 |

## 同步模式

跨 Pipeline 数据依赖靠同步原语保证（同步操作族详见 [sync_overview.md](sync_overview.md)）。典型生产者-消费者链：

```
hir.load (MTE2) → set_flag[MTE2→V] → hir.vadd (V) → set_flag[V→MTE3] → hir.store (MTE3)
                  wait_flag 消费方等待                                 wait_flag 消费方等待
```

- Pipeline 间：`hir.set_flag` / `hir.wait_flag`；Pipeline 内顺序：`hir.pipe_barrier`
- 块间：`hir.sync_block` / `hir.sync_block_set` / `hir.sync_block_wait`

随路功能（Padding / 量化 / 激活 / Eviction）的完整枚举与各操作支持矩阵见 [dma_padding_quantization.md](dma_padding_quantization.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMDMAOps.cpp#L607-L632](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L607-L632)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
- [dma-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/dma-ops.mlir)
