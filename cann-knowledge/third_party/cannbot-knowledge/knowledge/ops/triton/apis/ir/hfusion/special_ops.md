---
type: API
title: HFusion 特殊操作（print / assert / barrier / sort / histogram / embedding_gather
  / is_inf / is_nan / is_finite / mulext / symbolic_dim）
description: HFusion 十一个特殊操作：Device 端调试 print/assert、单 Core 全 Pipeline 同步 barrier、尾轴
  sort、整数直方图 histogram、嵌入查找 embedding_gather、浮点数值检查 is_*、扩展整数乘 mulext 与符号维引用 symbolic_dim。
tags: [task.lookup, topic.compilation, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 特殊操作

不归入前五类的十一个操作：调试、同步、排序/统计、嵌入查找、数值检查、扩展乘法与符号维。

## 调试操作

| 操作 | 签名要点 | 说明 |
|------|----------|------|
| `hfusion.print` | `prefix`(StrAttr), `hex`(BoolAttr), `arg`(整数/浮点/张量) | Device 端打印 |
| `hfusion.assert` | `msg`(StrAttr), `cond`(整数/张量) | Device 端断言 |

```mlir
hfusion.print "value:" hex = true %val : tensor<128xf32>
hfusion.assert "index out of range" %cond : i32
```

## barrier

`hfusion.barrier`：无操作数无结果，同步**一个 Core 上所有 Pipeline** 的执行。lowering 到 HIVM 的流水线同步体系，见 [sync_overview](../hivm/sync_overview.md) 与 [sync_pipe_sync](../hivm/sync_pipe_sync.md)。

## sort

沿 `sort_axis` 排序，输出排序值与对应索引；`descending`(BoolAttr，默认 false) 控制降序。约束：输入输出 rank 相同，当前仅支持尾轴排序。

```mlir
%r = hfusion.sort ins(%src : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
```

HIVM 侧排序/累积承载见 [vector_cumulative_sort](../hivm/vector_cumulative_sort.md)。

## histogram

计算整数张量直方图：输入 I8/UI8/I16/UI16/I32/UI32/I64/UI64，`num_bins`(I64Attr) 指定桶数，可选 `mask`(I1 tensor) 仅统计 mask[i]=true 的元素。输出为 1D 张量（I8/I16/I32/I64），长度等于 num_bins。

```mlir
%hist = hfusion.histogram %input, 256, %mask : tensor<1024xi32>, tensor<1024xi1> -> tensor<256xi64>
```

## embedding_gather

Gather 语义的嵌入查找：`result[b][i][d] = src[index[b][i]][d]`。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | AnyMemRef | 2D 嵌入表（GM） |
| `index` | AnyRankedTensor | 1D/2D 索引张量 |
| `dst` | AnyRankedTensor | 目标张量 |
| `bound` | I32/I64 | 词汇表大小（边界检查） |
| `offsets` / `numels` | Variadic\<I32/I64\> | 偏移量 / 元素数量 |

```mlir
%r = hfusion.embedding_gather
  ins(%src, %index, %bound, [], [] : memref<10000x768xf16>, tensor<128xi32>, i32)
  outs(%dst : tensor<128x768xf16>) -> tensor<128x768xf16>
```

## 浮点数值检查（is_inf / is_nan / is_finite）

输入 BF16/F16/F32，输出 I1 布尔张量。`isfinite` 带 `BiShengIRAggregatedOpInterface`（支持 decomposeOperation，具体分解形式以实现为准）。

```mlir
%a = hfusion.isinf %in : tensor<128xf32> -> tensor<128xi1>
%b = hfusion.isnan %in : tensor<128xf32> -> tensor<128xi1>
%c = hfusion.isfinite %in : tensor<128xf32> -> tensor<128xi1>
```

## mulext

扩展有符号整数乘法：返回 2N 位乘积的低半与高半。Traits：`Pure`、`Commutative`、`AllTypesMatch<[lhs, rhs, low, high]>`。

```mlir
%low, %high = hfusion.mulext %a, %b : i32
```

## symbolic_dim

引用符号维度并返回 index 类型值，配合 symbol 方言做符号化形状推导；`hfusion-fold-symbolic-dim` / `hfusion-unfold-symbolic-dim` Pass 在 tensor.dim 与该操作间互换（见 [transforms.md](transforms.md)）。

```mlir
%0 = hfusion.symbolic_dim @SymName : index
```

相关：[overview.md](overview.md)、[reduction_ops.md](reduction_ops.md)、[memory_ops.md](memory_ops.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HFusionOps.td#L48-L269](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L48-L269)
