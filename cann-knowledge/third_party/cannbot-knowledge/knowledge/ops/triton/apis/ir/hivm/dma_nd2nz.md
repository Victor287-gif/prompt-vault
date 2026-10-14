---
type: API
title: HIVM DMA ND2NZ（hivm.hir.nd2nz）
description: hivm.hir.nd2nz 操作语义：GM→L1 的 MTE2 搬运并随路做 ND→NZ（Fractal）布局转换，是 Cube 矩阵乘数据通路的入口；dst_continuous/init_out_buffer/pad_value
  属性与验证约束。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM DMA ND2NZ（hivm.hir.nd2nz）

`hivm.hir.nd2nz` 把数据从 GM 搬运到 L1，同时执行 ND（Normal Data，行优先连续）到 NZ（Fractal/Z 型分形）的布局转换，走 MTE2 Pipeline，属 Cube 核心类型（`CubeCoreTypeTrait`）。Cube 核心的矩阵乘（`hir.mmadL1`，见 [macro_mmad_l1.md](macro_mmad_l1.md)）要求输入以 NZ 格式存于 L1，因此 nd2nz 是矩阵数据进入 Cube 流水线的必经之路，由编译器自动插入，无直接 Python API 对应。

## MLIR 语法

```mlir
hivm.hir.nd2nz ins(%src : memref<256x128xf16, strided<[2048, 1], offset: 0>>)
               outs(%dst : memref<256x128xf16>)

hivm.hir.nd2nz {dst_continuous}
               ins(%src : memref<?x?x?x?xf32, #hivm.address_space<gm>>)
               outs(%dst : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>)
               init_out_buffer = true pad_value = %cst : f32
```

## 操作数与属性

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `src` | AnyShaped | 是 | 源缓冲区，通常 GM 地址空间 |
| `dst` | AnyShaped | 是 | 目标缓冲区，通常 L1（cbuf）地址空间 |
| `dst_continuous` | UnitAttr | 否 | 目标数据连续存储标志；**设置后 op 才能转换为优化的库函数调用** |
| `init_out_buffer` | BoolAttr | 否 | 搬运前初始化输出缓冲区（默认 false）；为 true 时 `pad_value` 必填 |
| `pad_value` | AnyType | 否 | 初始化填充值，类型与元素类型一致 |
| `init_condition` | AnyType | 否 | 条件化初始化 |

无最大 rank 限制（`NoMaxRankTrait`），与 load/store 的 rank≤3 不同。

## ND 与 NZ 布局

```
ND（行优先连续）          NZ（按 Fractal Block 重排，适配 Cube 引擎）
0  1  2  3                [0,1] [2,3]
4  5  6  7       →        [4,5] [6,7]
8  9  10 11               [8,9] [10,11]
```

布局规则的硬件背景见 [data_layout](../architecture/data_layout.md)。

## 验证器约束

1. src 与 dst 元素类型必须一致。
2. `init_out_buffer = true` 时 `pad_value` 必须提供。
3. Pipeline 固定 MTE2；Core Type 固定 CUBE（转换产物仅供 Cube 的 MMAD 消费）。

## 典型流水线位置

```
hir.nd2nz (GM→L1, ND→NZ) → hir.mmadL1 (L1→L0A/L0B→L0C) → hir.fixpipe (L0C→GM/UB, 可选 NZ→ND)
```

库函数命名随下游调整：下游为 `MmadL1Op` 且本 op 提供 per-channel bias 时，函数名追加 `_forbias` 后缀。

## 选型要点

- 数据给 Cube 做矩阵乘 → `hir.nd2nz`；数据给 Vector 核心 → [hir.load](dma_load.md)（GM→UB，无布局转换）。
- 逆操作（L1→GM、NZ→ND）见 [dma_nz2nd.md](dma_nz2nd.md)；从 L0C 做 NZ→ND 见 [dma_fixpipe.md](dma_fixpipe.md)。
- IR 中见到缺 `dst_continuous` 的 nd2nz 意味着无法走优化库函数路径，是潜在的搬运性能信号。

相关：[dma_nz2nd.md](dma_nz2nd.md)、[dma_fixpipe.md](dma_fixpipe.md)、[dma_overview.md](dma_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMDMAOps.td#L328-L368](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L328-L368)
- [HIVMDMAOps.cpp#L660-L671](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L660-L671)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
- [dma-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/dma-ops.mlir)
