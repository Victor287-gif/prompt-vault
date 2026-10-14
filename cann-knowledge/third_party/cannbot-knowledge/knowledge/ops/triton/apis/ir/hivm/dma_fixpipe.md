---
type: API
title: HIVM DMA Fixpipe（hivm.hir.fixpipe）
description: hivm.hir.fixpipe 操作语义：L0C→GM/UB/L1 的 FIX Pipeline 搬运，随路支持布局转换（NZ2ND/NZ2DN/NZ2NZ）、量化（F322F16/S322I8/F322BF16/QF322F32_PRE）、激活（ReLU/LeakyReLU/PReLU）与双目标拆分。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# HIVM DMA Fixpipe（hivm.hir.fixpipe）

`hivm.hir.fixpipe` 把数据从 L0C（Cube 累加器缓存）搬运到 GM/UB/L1，走 FIX Pipeline，属 Cube 核心类型，是矩阵乘结果写出的核心操作，也是 HIVM 中最复杂的 DMA 操作。随路（on-the-fly）功能在搬运同时完成，省去额外向量计算：布局转换、量化、激活、双目标输出。对应扩展 Python API `al.fixpipe`。

## MLIR 语法

```mlir
hivm.hir.fixpipe ins(%l0c : memref<256x128xf16>)
                 outs(%dst : memref<256x128xf16, strided<[2048, 1], offset: 0>>)

hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(...) outs(...)

%r = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>,
                       pre_relu = #hivm.fixpipe_pre_relu_mode<NORMAL_RELU>}
                      ins(%l0c : tensor<256x128xf32>)
                      outs(%dst : tensor<256x128xf16>) -> tensor<256x128xf16>
```

## 操作数与核心参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | AnyShaped | 源缓冲区（L0C，地址空间 `cc`） |
| `dst` | AnyShaped | 目标（GM/UB/L1；dst=UB 仅 Ascend950） |
| `quant_scale` | AnyFloat | 量化缩放因子，仅 QF322F32_PRE 模式使用 |
| `unit_flag_cond` / `unit_flag_mode` | Variadic<I1>/UnitFlagArrayAttr | Unit-flag 模式，用于循环至少执行一次的依赖场景 |

## 属性枚举

### dma_mode（HIVM_FixpipeDMAModeAttr，默认 NZ2NZ）

| 字面量 | 值 | 说明 |
|--------|---|------|
| `nz2nd` | 0 | NZ→ND 布局转换后搬运 |
| `nz2dn` | 1 | NZ→DN 转换（仅 Ascend950） |
| `normal` | 2 | NZ2NZ 保持 NZ 直搬（默认） |

### pre_quant（HIVM_FixpipePreQuantModeAttr，默认 NO_QUANT）

| 字面量 | 值 | 源→目标 |
|--------|---|---------|
| `NO_QUANT` | 0 | 不量化 |
| `F322F16` | 1 | F32→F16 |
| `S322I8` | 9 | F32→I8 |
| `QF322F32_PRE` | 15 | F32→F32 带 scale（需 quant_scale） |
| `F322BF16` | 16 | F32→BF16 |

### pre_relu（HIVM_FixpipePreReluModeAttr，默认 NO_RELU）

| 字面量 | 值 | 公式 |
|--------|---|------|
| `NO_RELU` | 0 | — |
| `NORMAL_RELU` | 1 | max(0, x) |
| `LEAKY_RELU` | 2 | x>0 ? x : alpha·x |
| `P_RELU` | 3 | x>0 ? x : p·x |

### dual_dst_mode（HIVM_FixpipeDualDstModeAttr，默认 NO_DUAL，仅 Ascend950 且 dst=UB）

| 字面量 | 值 | 说明 |
|--------|---|------|
| `NO_DUAL` | 0 | 单目标 |
| `ROW_SPLIT` | 1 | 按 M 维拆到两个 UB（M 须为 2 的倍数） |
| `COLUMN_SPLIT` | 2 | 按 N 维拆（N 须为 32 的倍数） |

## 随路功能执行顺序与组合约束

```
L0C → [Pre-Quant] → [Pre-ReLU] → [DMA Mode 布局转换] → [Dual Dst 拆分] → GM/UB/L1
```

量化+激活可叠加（先量化后激活）；NZ2DN 与双目标互斥；双目标要求 dma_mode 为 NZ2ND 或 NZ2NZ 且 dst=UB。完整支持矩阵见 [dma_padding_quantization.md](dma_padding_quantization.md)。随路量化与显式向量类型转换（[vector_cast_ops.md](vector_cast_ops.md) 的 vcast）是两条路径：fixpipe 随路零额外指令开销，应优先。

## 验证器约束

1. NZ2DN、dst=UB、双目标模式均仅 Ascend950 支持。
2. 双目标时 dma_mode 不得为 NZ2DN，且通路必须 L0C→UB。
3. Pipeline 固定 FIX，Core Type 固定 CUBE，无 rank 上限（`NoMaxRankTrait`）。
4. `getFixpipeState()` 状态机：`Init`(-1) / `QuantOrActivation`(0，有随路量化或激活) / `End`(1，含存储已完成)。

## 流水线位置与选型

```
hir.nd2nz (GM→L1) ×2 → hir.mmadL1 (→L0C) → hir.fixpipe (L0C→GM/UB/L1)
```

- 数据源是 L0C → fixpipe（可利用随路量化/激活）；数据源是 L1 的 NZ 数据 → [hir.nz2nd](dma_nz2nd.md)。
- 库函数命名：`fixpipe_{dma_mode}_{dual?}{src_type}_to_{dst_type}_{src_rank}d_to_{dst_rank}d_{dst_space}`。
- 双目标模式服务于 Mix（Cube+Vector）混合核心下把 L0C 矩阵拆给两个 UB 做 Cube-Vector 协同。

相关：[dma_nd2nz.md](dma_nd2nz.md)、[dma_nz2nd.md](dma_nz2nd.md)、[dma_overview.md](dma_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMDMAOps.td#L246-L326](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L246-L326)
- [HIVMAttrs.td#L847-L859](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L847-L859)
- [HIVMAttrs.td#L821-L845](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L821-L845)
- [HIVMAttrs.td#L783-L801](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L783-L801)
- [HIVMAttrs.td#L803-L819](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L803-L819)
- [HIVMDMAOps.cpp#L848-L891](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L848-L891)
- [HIVMDMAOps.cpp#L808-L842](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L808-L842)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
- [dma-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/dma-ops.mlir)
