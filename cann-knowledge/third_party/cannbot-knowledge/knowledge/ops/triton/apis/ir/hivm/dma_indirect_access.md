---
type: API
title: HIVM 间接访存（hivm.hir.indirect_load / hivm.hir.indirect_store）
description: hivm.hir.indirect_load 与 hivm.hir.indirect_store 操作语义：SIMT 模板、PIPE_V
  的按偏移张量间接访存，DestinationStyle 接口，支持 1D-5D 与 mask/other。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 间接访存（hivm.hir.indirect_load / hivm.hir.indirect_store）

按偏移张量对 GM 做间接加载/存储，使用 SIMT（Single Instruction Multiple Thread）模板执行——每个线程独立处理一个索引位置，因此归属 **PIPE_V**（Vector Pipeline）。实现 `DestinationStyleOpInterface`，语义结构化，支持 1D-5D（`StaticMaxRankTrait<5>`），由编译器在 lowering 阶段生成，无直接 Python API 对应。

## hir.indirect_load

逐位置语义（各维度同理，1D 为例）：

```
dst[i] = mask[i] ? src[offsets[i]] : other[i]
```

```mlir
%result = hivm.hir.indirect_load ins(%src : memref<?xf32>, %offsets : tensor<16xi32>,
                                     %mask : tensor<16xi1>, %other : tensor<16xf32>)
                                 outs(%dst : tensor<16xf32>) -> tensor<16xf32>
```

| 参数 | 必选 | 说明 |
|------|------|------|
| `src`（AnyMemRef） | 是 | 源缓冲区，GM |
| `offsets` | 是 | 偏移索引张量，指定加载位置 |
| `dst` | 是 | 目标张量，决定输出形状与类型 |
| `mask` / `other` | 否 | 掩码与回退值；other 元素类型与 dst 一致 |

## hir.indirect_store

```
if (mask[i]) dst[offsets[i]] = src[i]     // 无 mask 时全部写出
```

```mlir
hivm.hir.indirect_store ins(%src : tensor<16xf32>, %offsets : tensor<16xi32>,
                            %mask : tensor<16xi1>)
                        outs(%dst : memref<?xf32>)
```

| 参数 | 必选 | 说明 |
|------|------|------|
| `src` | 是 | 源数据张量（UB 中） |
| `offsets` | 是 | 偏移索引张量，指定存储位置 |
| `dst`（AnyMemRef） | 是 | 目标缓冲区，GM |
| `mask` | 否 | 掩码 |

## 验证约束

- 最大 rank 5；Pipeline 固定 `PIPE_V`；带 `LibraryFunctionOpInterface`（有库函数实现）。
- indirect_load：src/dst/other 元素类型一致；可变操作数段大小由 `AttrSizedOperandSegments` 控制。
- indirect_store：src 与 dst 元素类型一致。

## 与 gather/scatter 系列对比

| 特性 | indirect_load / indirect_store | [gather_load / scatter_store](dma_gather_scatter.md) |
|------|-------------------------------|------------------------------------------------------|
| Pipeline | PIPE_V（SIMT） | 无固定 |
| 接口 | DestinationStyle | 非 DestinationStyle，ins/outs 分离 |
| Rank | 1D-5D | 无限制 |
| 库函数 | 有 | 无 |
| cache/evict/boundaryCheck 属性 | 不支持 | 支持 |

典型转换：`hir.gather_load → hir.indirect_load`（分解）或 `→ hir.custom name="__builtin_gather_load"`（hir.custom 见 [custom_custom_op.md](custom_custom_op.md)）。5D 上限对常见深度学习维度（batch/seq/head/h/w）足够；更高维需先 reshape。

相关：[dma_gather_scatter.md](dma_gather_scatter.md)、[dma_overview.md](dma_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMOps.td#L524-L592](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L524-L592)
- [HIVMOps.td#L598-L660](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L598-L660)
- [HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td)
