---
type: API
title: HIVM DMA 随路功能（Padding / 量化 / 激活 / Eviction）
description: HIVM DMA 随路功能汇总：PadMode 三值与左右填充规则、FixpipePreQuantMode 五种量化、FixpipePreReluMode
  三种激活、EvictionPolicy，以及随路功能组合顺序与各操作支持矩阵。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.alignment]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# HIVM DMA 随路功能（Padding / 量化 / 激活 / Eviction）

随路（on-the-fly）功能在 DMA 搬运过程中同时完成，不产生额外计算指令，是 Ascend NPU 硬件的重要特性，可显著减少计算开销与访存次数。本卡汇总全部随路枚举与组合规则；各宿主操作的完整语义见 [dma_load.md](dma_load.md)、[dma_copy.md](dma_copy.md)、[dma_fixpipe.md](dma_fixpipe.md)。

## Padding（填充）

Tiling 后边界 Tile 源数据小于目标缓冲区时，用 Padding 填充多余位置。

### PadMode（#hivm.padmode<...>，用于 hir.load / hir.copy）

| 字面量 | 值 | 说明 |
|--------|---|------|
| `PadNull` | 0 | 不填充，源目标形状必须相同 |
| `PadFirstElem` | 1 | 用源数据第一个元素填充（边界复制场景） |
| `PadValue` | 2 | 用指定 `pad_value` 填充；仅给 pad_value 时自动推断为此模式 |

支持矩阵：`hir.load` 支持左+右 Padding；`hir.copy` **仅左侧**（硬件引擎限制）。`init_out_buffer` 在 Padding 覆盖不到全部位置时先把整个目标缓冲区初始化，二者可叠加：

```mlir
hivm.hir.load ins(%src : memref<16x15xf16, #hivm.address_space<gm>>)
              outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
              init_out_buffer = true pad_value = %val : f16
```

### PaddingOption（gather_load 专用 padding 属性）

| 字面量 | 值 | 说明 |
|--------|---|------|
| `zero`（PAD_ZERO） | 1 | 越界位置填零 |
| `nan`（PAD_NAN） | 2 | 越界位置填 NaN |

## Quantization（fixpipe 随路量化）

FixpipePreQuantMode（#hivm.fixpipe_pre_quant_mode<...>），L0C 搬出时同步类型转换：

| 字面量 | 值 | 源→目标 | 备注 |
|--------|---|---------|------|
| `NO_QUANT` | 0 | — | 默认 |
| `F322F16` | 1 | F32→F16 | 混合精度场景 |
| `S322I8` | 9 | F32→I8 | 量化推理 |
| `QF322F32_PRE` | 15 | F32→F32 | 带 scale 预量化，需 `quant_scale` 参数 |
| `F322BF16` | 16 | F32→BF16 | — |

```mlir
%r = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F32_PRE>}
                      ins(%l0c : tensor<256x128xf32>) outs(%dst : tensor<256x128xf32>)
                      quant_scale = %scale : f32 -> tensor<256x128xf32>
```

`quant_scale` 在保持 F32 的同时模拟量化缩放，常用于量化感知训练。

## Activation（fixpipe 随路激活）

FixpipePreReluMode（#hivm.fixpipe_pre_relu_mode<...>）：

| 字面量 | 值 | 公式 |
|--------|---|------|
| `NO_RELU` | 0 | — |
| `NORMAL_RELU` | 1 | max(0, x) |
| `LEAKY_RELU` | 2 | x>0 ? x : alpha·x |
| `P_RELU` | 3 | x>0 ? x : p·x |

## 组合规则

执行顺序：`L0C → Pre-Quant → Pre-ReLU → DMA Mode（布局转换）→ Dual Dst（拆分）→ 目标`。

| 组合 | 支持 | 备注 |
|------|------|------|
| 量化+激活 | 是 | 先量化后激活 |
| 量化/激活 + NZ2ND | 是 | — |
| 量化/激活 + NZ2DN | 是 | 仅 Ascend950 |
| 量化/激活 + 双目标 | 是 | 仅 Ascend950 且 dst=UB |
| NZ2DN + 双目标 | **否** | 互斥 |

## Eviction Policy（#hivm.evictionpolicy<...>，仅 hir.load）

| 字面量 | 值 | 说明 |
|--------|---|------|
| `EvictFirst` | 0 | 优先驱逐，适合一次性数据 |
| `EvictLast` | 1 | 最后驱逐，适合反复复用数据 |

## 各操作随路功能支持矩阵

| 功能 | hir.load | hir.store | hir.copy | hir.fixpipe | hir.nd2nz |
|------|---------|----------|---------|------------|----------|
| PadValue / PadFirstElem / 左 Padding | ✓ | — | ✓ | — | — |
| 右 Padding | ✓ | — | — | — | — |
| init_out_buffer | ✓ | — | — | — | ✓ |
| Eviction Policy | ✓ | — | — | — | — |
| 量化 / 激活 / 布局转换 / 双目标 | — | — | — | ✓ | — |
| 原子操作 | — | ✓ | — | — | — |

相关：[dma_fixpipe.md](dma_fixpipe.md)、[dma_load.md](dma_load.md)、[dma_copy.md](dma_copy.md)、[dma_gather_scatter.md](dma_gather_scatter.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td#L329-L349](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L329-L349)
- [HIVMAttrs.td#L920-L928](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L920-L928)
- [HIVMAttrs.td#L783-L801](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L783-L801)
- [HIVMAttrs.td#L803-L819](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L803-L819)
- [HIVMAttrs.td#L356-L372](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L356-L372)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
