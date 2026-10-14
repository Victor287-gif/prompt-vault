---
type: API
title: HIVM 稀疏访存（hivm.hir.gather_load / hivm.hir.scatter_store）
description: hivm.hir.gather_load 与 hivm.hir.scatter_store 操作语义：按索引张量对 GM 做非连续收集/散列写，支持
  mask、burst_len、boundaryCheck、cache 与 evict 属性。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 稀疏访存（hivm.hir.gather_load / hivm.hir.scatter_store）

按索引张量对全局内存做**非连续**的按位置收集/散列写，适用于稀疏数据、嵌入查找等场景。与 [hir.load](dma_load.md)/[hir.store](dma_store.md) 的连续访问不同，这两个 op 以显式索引张量定位元素，并支持 mask 条件化。对应 Triton 带 index 的 `tl.load` / `tl.store`。注意二者定义于 HIVMOps.td（Custom/基础族），无固定 Pipeline。

## hir.gather_load — 稀疏加载

```mlir
%result = hivm.hir.gather_load ins(%base : memref<?xf32>, %indices : tensor<16xi32>,
                                   %burst_len : i32, %mask : tensor<16xi1>, %other : f32)
                               -> tensor<16xf32>
```

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `base` | AnyMemRef | 是 | 源缓冲区基地址，GM |
| `indices` | tensor<I32/I64> | 是 | 偏移索引张量 |
| `burst_len` | I32/I64 | 是 | 每次访问的突发传输长度，越大带宽利用率越高但缓冲需求越大 |
| `mask` | tensor<I1> | 否 | 掩码，控制哪些位置加载 |
| `other` | 整/浮点 | 否 | 掩码位置的回退值，类型与结果元素一致 |

属性：`boundaryCheck`（DenseI32ArrayAttr，边界检查维度）、`padding`（`PAD_ZERO`(1)/`PAD_NAN`(2)，越界填充）、`cache`（CacheModifier：`none/ca/cg/wb/cs/wt/cv`）、`evict`（EvictFirst/EvictLast）、`isVolatile`。

## hir.scatter_store — 稀疏存储

```mlir
hivm.hir.scatter_store ins(%base : memref<?xf32>, %indices : tensor<16xi32>,
                           %data : tensor<16xf32>, %burst_len : i32, %mask : tensor<16xi1>)
```

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `base` | AnyMemRef | 是 | 目标缓冲区基地址，GM |
| `indices` | tensor<I32/I64> | 是 | 偏移索引张量 |
| `data` | AnyRankedTensor | 是 | 源数据（UB 中） |
| `burst_len` | I32/I64 | 是 | 突发传输长度 |
| `mask` | tensor<I1> | 否 | 掩码，控制哪些位置写出 |

属性：`boundaryCheck` / `cache` / `evict`（无 padding、无 other）。

## 验证约束

- `indices` 必须 I32/I64 张量；`mask` 必须 I1 张量；`other` 类型与输出元素一致。
- `SameVariadicOperandSize`：可变操作数大小一致。

## 与 load/store、indirect 系列对比

| 特性 | hir.load / hir.store | gather_load / scatter_store | [indirect_load / indirect_store](dma_indirect_access.md) |
|------|---------------------|----------------------------|----------------------------------------------------------|
| 访问模式 | 连续 | 按索引非连续 | 按索引非连续 |
| 掩码 | 经 Padding | mask 张量 | mask 张量 |
| Pipeline | MTE2/MTE3 | 无固定 | PIPE_V（SIMT 模板） |
| 缓存/边界属性 | eviction_policy | cache/evict/boundaryCheck | 不支持 |
| 接口风格 | ins/outs | ins/outs 分离 | DestinationStyle，1D-5D |

编译流程中 `gather_load` 可能被分解为 `indirect_load` 或 `hir.custom name="__builtin_gather_load"`（hir.custom 见 [custom_custom_op.md](custom_custom_op.md)）。访问连续时用 load/store 更高效。

相关：[dma_indirect_access.md](dma_indirect_access.md)、[dma_load.md](dma_load.md)、[dma_overview.md](dma_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMOps.td#L389-L426](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L389-L426)
- [HIVMAttrs.td#L920-L928](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L920-L928)
- [HIVMAttrs.td#L930-L942](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L930-L942)
- [HIVMOps.td#L459-L492](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L459-L492)
- [HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
