---
type: API
title: HFusion 数据搬移操作（interleave / deinterleave / flip / gather / arange / broadcast
  等）
description: HFusion 数据重排操作族：interleave/deinterleave 末维交织、flip 翻转、gather 轴向聚集、arange
  等差序列，及 broadcast/transpose/concat/pad 的 Linalg 承载方式。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 数据搬移操作

在不同形状与布局的张量间做数据重排。`interleave` / `deinterleave` / `flip` / `gather` / `arange` 由 HFusion 自行定义；`broadcast` / `transpose` / `concat` / `pad` 直接由 Linalg 命名操作承载。lowering 后的 HIVM 搬移对应物见 [dma_overview](../hivm/dma_overview.md) 与 [vector_data_movement](../hivm/vector_data_movement.md)。

张量语义的逐元读写 `load` / `store` 在源文档中归入逐元操作族，见 [elementwise_ops.md](elementwise_ops.md)。

## interleave

将 n 个同形状输入张量沿最后一维交织成一个输出，当前仅支持 n=2。Traits：`Pure`、`Commutative`、`SameOperandsAndResultRank`、`ReifyRankedShapedTypeOpInterface`。

```mlir
%out = hfusion.interleave %a, %b : tensor<128x256xf32>, tensor<128x256xf32> -> tensor<128x512xf32>
```

## deinterleave

interleave 的逆操作：沿末维反交织，偶数索引元素进第一个输出、奇数索引进第二个。约束：输入末维大小必须是 2 的倍数。

| `channel_index` | 行为 |
|-----------------|------|
| -1 | 输出两个张量（偶数 + 奇数通道） |
| 0 | 仅输出偶数索引通道 |
| 1 | 仅输出奇数索引通道 |

```mlir
%even, %odd = hfusion.deinterleave %in channel_index = -1 : tensor<128x512xf32> -> tensor<128x256xf32>, tensor<128x256xf32>
%even_only = hfusion.deinterleave %in channel_index = 0 : tensor<128x512xf32> -> tensor<128x256xf32>
```

## flip

沿 `flip_axis` 翻转张量，当前仅支持最后一维。

```mlir
%out = hfusion.flip %in flip_axis = 1 : tensor<128x256xf32> -> tensor<128x256xf32>
```

## gather

沿指定轴从源张量聚集元素，对应 `triton.language.gather` 语义。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | AnyShaped | 源张量 |
| `index` | AnyShaped | 索引张量 |
| `init` | AnyShaped | 初始值（DPS） |
| `axis` | I64Attr | 聚集轴 |

Traits：`AllRanksMatch<[src, index, init]>`、`AllShapesMatch<[index, init]>`、`AllElementTypesMatch<[src, init]>`、`BiShengIRAggregatedOpInterface`（可 decompose）。

语义（src: 16x16，index: 16x4，axis=1）：`output[i][j] = (index[i][j] == k) ? src[i][k] : output[i][j]`。

```mlir
%r = hfusion.gather ins(%src, %index, %init : tensor<16x16xf32>, tensor<16x4xi32>, tensor<16x4xf32>) axis = 1 -> tensor<16x4xf32>
```

## arange

生成等差数列张量：`arange[i, j, k] = offset + stride[0]*i + stride[1]*j + stride[2]*k`。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `offset` | Optional\<Index\> | 偏移量（默认 0） |
| `strides` | Variadic\<Index\> | 各维步长 |
| `init` | AnyShaped | 决定输出形状的初始张量 |
| `result_tensor` | Optional\<AnyShaped\> | 输出 |

## Linalg 承载的搬移操作

| 操作 | 说明 | 实现方式 |
|------|------|----------|
| `broadcast` | 广播到目标形状 | Linalg 广播语义 |
| `transpose` | 维度转置 | Linalg 转置语义 |
| `concat` | 沿维拼接 | Linalg 拼接语义 |
| `pad` | 边界填充 | Linalg 填充语义 |

注意与内存操作区分：本卡操作都在张量语义层做重排；带 memref 基址与偏移张量的访存（gather_load / scatter_store / indirect_*）属于 [memory_ops.md](memory_ops.md)。HIVM 侧的连续搬移见 [dma_copy](../hivm/dma_copy.md)、[dma_load](../hivm/dma_load.md)、[dma_store](../hivm/dma_store.md)。

相关：[overview.md](overview.md)、[memory_ops.md](memory_ops.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HFusionOps.td#L128-L213](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L128-L213)
- [HFusionStructuredOps.td#L196-L260](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td#L196-L260)
