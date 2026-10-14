---
type: API
title: HIVM 原子操作（hivm.hir.atomic_cas / hivm.hir.atomic_xchg）
description: hivm.hir.atomic_cas（比较并交换）与 hivm.hir.atomic_xchg（原子交换）操作语义：GM 上不可分割读-改-写、返回旧值、直接映射硬件指令；与
  hir.store 原子模式的对比。
tags: [task.lookup, topic.compilation, topic.implementation.atomic, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 原子操作（hivm.hir.atomic_cas / hivm.hir.atomic_xchg）

两个独立原子操作，在 GM 上执行不可分割的读-改-写，是并发同步原语的基础。与 [hivm.hir.store](dma_store.md) 的 `atomic_kind` 模式不同，它们**不依赖** `hir.set_atomic` 全局原子模式，带 `NoLibraryFunctionTrait` 直接映射硬件指令，且**返回内存原值**。对应 Triton 的 `tl.atomic_cas` / `tl.atomic_xchg`。

## hir.atomic_cas — 比较并交换

语义：读内存位置 V；若 `V == src0`（期望旧值）则写入 `src1`（新值）；无论是否更新都返回 V 原值（Tensor 语义）。返回值可用于判定 CAS 是否成功（返回值 == 期望旧值即成功）。

```mlir
hivm.hir.atomic_cas ins(%src0, %src1 : memref<16xf32, #hivm.address_space<ub>>,
                                      memref<16xf32, #hivm.address_space<ub>>)
                    outs(%dst : memref<16xf32, #hivm.address_space<gm>>)

%result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<16xi32>, tensor<16xi32>)
                              outs(%dst : tensor<16xi32>) -> tensor<16xi32>
```

| 参数 | 说明 |
|------|------|
| `src`（恰好 2 个） | src0 = 期望旧值，src1 = 新值 |
| `dst` | 目标内存位置，GM 地址空间 |
| `result_tensor` | 返回 V 的原始值 |

## hir.atomic_xchg — 原子交换

语义：读内存位置当前值，写入新值，返回旧值，全程原子。

```mlir
hivm.hir.atomic_xchg ins(%src : memref<16xf32, #hivm.address_space<ub>>)
                     outs(%dst : memref<16xf32, #hivm.address_space<gm>>)
```

| 参数 | 说明 |
|------|------|
| `src`（恰好 1 个） | 新值 |
| `dst` | 目标内存位置，GM 地址空间 |
| `result_tensor` | 返回旧值 |

## 共同约束

- 元素类型：`i8/i16/i32/i64/f16/f32`（**不支持 bf16 与 f8**，硬件原子指令限制）。
- 输入与输出元素类型、rank 必须一致。
- 无固定 Pipeline 归属。

## 与 hir.store 原子模式对比

| 特性 | hir.store + atomic | hir.atomic_cas | hir.atomic_xchg |
|------|-------------------|----------------|-----------------|
| 操作类型 | ADD/MAX/MIN（硬件）；AND/OR/XOR/CAS/XCHG（软件锁） | CAS | XCHG |
| 依赖 set_atomic | 是（或 atomic_kind 属性） | 否 | 否 |
| 返回旧值 | 否 | 是 | 是 |
| 实现 | 硬件/软件混合 | 硬件指令 | 硬件指令 |
| 元素类型 | i8~f8E5M2（宽） | i8/i16/i32/i64/f16/f32 | 同左 |

选型：需要 CAS/XCHG 语义或需要旧值时用本卡操作（硬件指令、性能优于 store 的软件 CAS/XCHG）；只需 add/max/min 时 store 硬件原子更直接。

相关：[dma_store.md](dma_store.md)、[dma_overview.md](dma_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMDMAOps.td#L406-L443](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L406-L443)
- [HIVMDMAOps.td#L445-L481](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L445-L481)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
