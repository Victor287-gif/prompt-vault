---
type: API
title: HIVM DMA NZ2ND（hivm.hir.nz2nd）
description: hivm.hir.nz2nd 操作语义：L1→GM 的 MTE3 搬运并随路做 NZ→ND 布局转换，是 nd2nz 的逆操作；rank≤2
  约束、验证规则与库函数命名。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM DMA NZ2ND（hivm.hir.nz2nd）

`hivm.hir.nz2nd` 把数据从 L1 搬运到 GM，同时执行 NZ（Fractal）到 ND（行优先连续）的布局转换，走 MTE3 Pipeline，属 Cube 核心类型。它是 [hivm.hir.nd2nz](dma_nd2nz.md) 的逆操作，常用于把 Cube 计算产生的 NZ 格式中间结果从 L1 写回 GM 并恢复 ND 格式，由编译器自动插入，无直接 Python API 对应。

## MLIR 语法

```mlir
hivm.hir.nz2nd ins(%src : memref<256x128xf16, #hivm.address_space<cbuf>>)
               outs(%dst : memref<256x128xf16, #hivm.address_space<gm>>)

%res = hivm.hir.nz2nd ins(%src : tensor<256x128xf16>)
                      outs(%dst : tensor<256x128xf16>)
                      -> tensor<256x128xf16>
```

## 操作数与属性

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `src` | TensorOrMemref | 是 | 源缓冲区，memref 语义下必须为 L1（cbuf）地址空间 |
| `dst` | TensorOrMemref | 是 | 目标缓冲区，memref 语义下必须为 GM 地址空间 |
| `result_tensor` | Optional<AnyRankedTensor> | 结果 | Tensor 语义下的结果 |

无任何额外属性，是最简洁的 DMA 操作之一。

## 验证器约束

1. src 与 dst 元素类型必须一致。
2. memref 语义下 src 必须为 L1、dst 必须为 GM。
3. **最大 rank 为 2**（`StaticMaxRankTrait<2>`）——NZ↔ND 转换在 2D 矩阵上进行，更高维数据需先 reshape 为 2D。
4. Pipeline 固定 MTE3；Core Type 固定 CUBE。

## 库函数命名

```
nz2nd_{src_rank}d_to_{dst_rank}d_{datatype}
```

## 与其他操作的关系

```
hir.nd2nz (GM→L1, ND→NZ)        hir.nz2nd (L1→GM, NZ→ND)
       ↓                                 ↑
   hir.mmadL1 → hir.fixpipe (L0C→L1, NZ) ┘
```

- **与 fixpipe NZ2ND 模式的区别**：`hir.nz2nd` 数据源是 L1，仅做布局转换；[hir.fixpipe](dma_fixpipe.md) `dma_mode = nz2nd` 数据源是 L0C，且可叠加随路量化/激活。数据源是 L0C 优先 fixpipe，已是 L1 中 NZ 数据才用 nz2nd。
- 典型场景：Cube 中间结果写回 GM；多轮矩阵乘之间把 NZ 部分和暂存 GM。

相关：[dma_nd2nz.md](dma_nd2nz.md)、[dma_fixpipe.md](dma_fixpipe.md)、[dma_overview.md](dma_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMDMAOps.td#L370-L386](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L370-L386)
- [HIVMDMAOps.cpp#L693-L714](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L693-L714)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
