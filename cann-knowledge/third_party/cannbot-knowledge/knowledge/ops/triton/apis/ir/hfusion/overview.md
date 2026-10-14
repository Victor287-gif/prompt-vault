---
type: API
title: HFusion 方言总览（张量级融合操作与 Linalg 结构化范式）
description: HFusion（Hybrid Fusion）方言定位、依赖方言、六大操作分类（逐元/归约/矩阵乘/数据搬移/内存/特殊共 42 个操作）、类型系统与操作基类。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 方言总览

HFusion（Hybrid Fusion，混合融合）是 BiShengIR 中的张量级融合操作方言，处于 Triton 前端语义与 HIVM 硬件映射之间：承接前端算子语义，完成融合、Tiling 与向量化，再 lowering 到 [overview](../hivm/overview.md) 的 `hivm.hir.*` 操作。操作基于 Linalg 结构化范式，可通过 OpDSL（YAML）声明式定义。方言自带 Canonicalizer。

- **方言名称**：`hfusion`，C++ 命名空间 `::mlir::hfusion`
- **依赖方言**：hacc（异构调用约定，见 [overview](../hacc/overview.md)）、hmap、linalg、mathExt、mesh、symbol
- **TableGen / 定义文件**：`bishengir/include/bishengir/Dialect/HFusion/IR/` 下的 HFusionBase.td（基类与枚举）、HFusionEnums.td、HFusionAttrs.td、HFusionOps.td（非结构化操作）、HFusionStructuredOps.td、HFusionNamedStructuredOps.yaml（OpDSL 声明式定义）

## 操作分类速查（42 个）

| 类别 | 代表操作 | 数量 | 详解卡 |
|------|----------|------|--------|
| 逐元操作 | elemwise_unary, elemwise_binary, compare, select, cast, bitcast | 6 | [elementwise_ops.md](elementwise_ops.md) |
| 归约操作 | reduce_with_index, cumsum, cumprod | 3 | [reduction_ops.md](reduction_ops.md) |
| 矩阵乘操作 | matmul_mx, group_matmul | 2 | [matmul_ops.md](matmul_ops.md) |
| 数据搬移 | load, store, broadcast, transpose, concat, pad, interleave, deinterleave, flip, gather, arange | 11 | [data_movement_ops.md](data_movement_ops.md) |
| 内存操作 | gather_load, scatter_store, indirect_load, indirect_store, gatherT, index_put, scatterT, atomic_cas, atomic_xchg | 9 | [memory_ops.md](memory_ops.md) |
| 特殊操作 | print, assert, barrier, sort, histogram, embedding_gather, is_inf, is_nan, is_finite, mulext, symbolic_dim | 11 | [special_ops.md](special_ops.md) |

函数选择、舍入、原子等控制枚举集中在 [attributes_enums.md](attributes_enums.md)；变换 Pass 清单见 [transforms.md](transforms.md)。

## 类型系统与操作基类

| 要素 | 定义 | 说明 |
|------|------|------|
| `TensorOrMemref` | `AnyTypeOf<[AnyMemRef, AnyRankedTensor]>` | 张量或 MemRef 均可 |
| `ShapedTypeOf<allowedTypes>` | 自定义 | 元素类型受限的 ShapedType |
| `HFusion_Op<mnemonic, traits>` | 非结构化操作基类 | print/assert/barrier 等 |
| `HFusionStructuredBase_Op<mnemonic, props>` | 结构化操作基类 | 实现 `LinalgStructuredInterface` + `DestinationStyleOpInterface`（ins/outs + DPS init 风格） |

## 典型 IR 形态

```mlir
%0 = hfusion.elemwise_unary ins(%in : tensor<128x256xf32>) outs(%init : tensor<128x256xf32>) unary_fn = <sqrt>

%v, %i = hfusion.reduce_with_index ins(%in, %vi, %ii : tensor<128x256xf32>, tensor<128xf32>, tensor<128xi64>)
  outs(%vo, %io : tensor<128xf32>, tensor<128xi64>) reduce_with_index_kind = <max>

%r = hfusion.matmul_mx ins(%a, %b, %sa, %sb : tensor<16x32xf8E4M3FN>, tensor<32x64xf8E4M3FN>, tensor<2x2xui8>, tensor<2x4xui8>)
  outs(%acc : tensor<16x64xf32>) -> tensor<16x64xf32>
```

读 IR 要点：HFusion 层的 `ins(...) outs(...)` 结构对应 Linalg DPS 风格，lowering 到 HIVM 后逐元操作落到 `hir.v*` 向量指令、矩阵乘落到 `hir.matmul`/`hir.mmadL1`、搬移落到 `hir.load/store/copy`，对应关系见各详解卡的互链。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HFusionBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionBase.td)
- [HFusionEnums.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td)
- [HFusionAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td)
- [HFusionOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td)
- [HFusionStructuredOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td)
- [HFusionNamedStructuredOps.yaml](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yaml)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td)
