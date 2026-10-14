---
type: API
title: HFusion 内存操作（gather_load / scatter_store / indirect_load / indirect_store /
  gatherT / index_put / scatterT / atomic_cas / atomic_xchg）
description: HFusion 九个 memref 级访存操作：带 burst_len 与掩码的聚集/散射、1D-5D 间接访存、SIMT 模式 gatherT/index_put/scatterT，及
  atomic_cas/atomic_xchg 原子操作语义。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.implementation.atomic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 内存操作

九个 memref 级访存操作，支持 GM 与 UB 间搬移及原子同步语义。与 [data_movement_ops.md](data_movement_ops.md) 的张量语义重排不同，本组操作直接以 memref 基址 + 偏移/索引张量访存。掩码修饰枚举（cache / evict / padding）见 [attributes_enums.md](attributes_enums.md)。

## gather_load

按偏移量从源缓冲区聚集加载，支持掩码与回退值。

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `base` | AnyMemRef | 源内存缓冲区 |
| `indices` | RankedTensorOf<[I32, I64]> | 偏移量张量 |
| `burst_len` | I32/I64 | 突发长度 |
| `mask` / `other` | Optional | 掩码 / 掩码位置回退值 |
| `boundaryCheck` / `padding` | OptionalAttr | 边界检查 / 填充选项（zero/nan） |
| `cache` / `evict` / `isVolatile` | OptionalAttr | 缓存修饰 / 驱逐策略 / volatile |

```mlir
%r = hfusion.gather_load ins(%base, %indices, %burst_len, %mask, %other :
  memref<?xf32>, tensor<128xi32>, i32, tensor<128xi1>, f32) -> tensor<128xf32>
```

## scatter_store

gather_load 的对偶：按偏移量把源张量散射写入目标缓冲区，操作数为 `base` / `indices` / `data` / `burst_len` / `mask`，修饰符为 `boundaryCheck` / `cache` / `evict`（比 gather_load 少 `padding` 与 `isVolatile`，无 other）。

## indirect_load / indirect_store

间接访存，支持 1D-5D 掩码：

```
indirect_load:  dst[i] = mask[i] ? src[offsets[i]] : other[i]
indirect_store: if (mask[i]) dst[offsets[i]] = src[i]   // SIMT 模板
```

`indirect_load` 操作数：`src`(memref) / `offsets`(I32/I64 tensor) / `dst`（决定输出形状类型）/ `mask`(I1/I8) / `other`；`indirect_store` 操作数：`dst`(memref) / `offsets` / `src` / 可选 `mask`。

## SIMT 模式三操作（gatherT / index_put / scatterT）

源/目标均为 GM 缓冲区，索引与步长以标量参数（I32/I64 及其 Variadic）显式给出：

| 操作 | 语义 | 关键参数 |
|------|------|----------|
| `gatherT` | 沿轴从 GM 按索引聚集 | `src` / `index` / `dst` / `bound`（gather 维大小）/ `dim` / `src_stride` / `index_shape` / `offsets` |
| `index_put` | 按索引写值到目标指定位置 | `dst` / `index` / `value` / `scatter_dim` / `bound` / `start_offset` / `end_offset` / `dst_stride` |
| `scatterT` | 按索引散射写 | `dst` / `value` / `index_tile` / `index_boundary` / `dim` / `dst_stride` / `index_shape` / `offsets` |

## atomic_cas / atomic_xchg

| 操作 | 输入 | 语义 |
|------|------|------|
| `hfusion.atomic_cas` | `src0`（期望旧值）、`src1`（新值）、`dst` | `if (V == A) V = B; return old_V` |
| `hfusion.atomic_xchg` | `src`（新值）、`dst` | 写入新值并返回旧值 |

两者均支持 memref（无副作用）与 tensor（返回值）两种形态：

```mlir
hfusion.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
%old = hfusion.atomic_xchg ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

## 与 HIVM 的对应关系

- 聚集/散射 → `hir.gather_load` / `hir.scatter_store`，见 [dma_gather_scatter](../hivm/dma_gather_scatter.md)
- 间接访存 → `hir.indirect_load` / `hir.indirect_store`，见 [dma_indirect_access](../hivm/dma_indirect_access.md)
- 原子操作 → `hir.atomic_cas` / `hir.atomic_xchg`，见 [dma_atomic](../hivm/dma_atomic.md)

相关：[overview.md](overview.md)、[data_movement_ops.md](data_movement_ops.md)、[attributes_enums.md](attributes_enums.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HFusionOps.td#L329-L928](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L329-L928)
