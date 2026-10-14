---
type: API
title: AscendDPX 方言（ascend_dpx SIMT 虚拟 ISA）
description: AscendDPX 方言定义 Ascend NPU 的 SIMT 虚拟 ISA 操作集：线程/Block/Grid 索引查询、带掩码与 L2
  缓存策略的 load/store、Warp Shuffle 与 Warp 归约、12 种通用原子操作加 CAS、类型转换与数学函数，是 SIMT 执行模式 lowering
  到 LLVM 前的最后 IR 层。
tags: [task.lookup, topic.compilation, topic.implementation.atomic, topic.data_movement.memory_access, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# AscendDPX 方言

AscendDPX 方言（方言名 `ascend_dpx`，C++ 命名空间 `::mlir::ascend_dpx`）定义 Ascend NPU 的 SIMT（Single Instruction Multiple Thread）虚拟 ISA 操作集，操作数/结果基于 LLVM 类型（`!llvm.ptr` 等），处于 SIMT 模式 lowering 到 LLVM 指令之前的抽象层。昇腾适配中 `ConvertTritonAscendGPUToLLVM` 将 TritonGPU 操作经本方言降到 LLVM（Triton 方言体系见 [overview](../triton_dialects/overview.md)）。

## 程序信息操作

均继承 `AscendDPX_InfoOp`，无操作数，返回 i32（clock64 返回 i64）。

| 类别 | 操作 |
|------|------|
| 线程索引 | `ascend_dpx.thread_id_x/y/z` |
| Block 索引 | `ascend_dpx.block_idx_x/y/z`、`ascend_dpx.block_idx`（线性） |
| Block 维度 | `ascend_dpx.block_dim_x/y/z` |
| Grid 维度 | `ascend_dpx.grid_dim_x/y/z` |
| 其他 | `ascend_dpx.clock32`、`ascend_dpx.clock64`（i64）、`ascend_dpx.core_id` |

```mlir
%tid = ascend_dpx.thread_id_x : i32
%bid = ascend_dpx.block_idx_y : i32
```

## 内存操作

### ascend_dpx.load / ascend_dpx.store

SIMT 模式加载/存储，支持掩码与 L2 缓存策略。

| 操作数 | load | store |
|--------|------|-------|
| `ptr` | `LLVM_AnyPointer` 源地址 | 目标地址 |
| `value` | — | 待存储值 |
| `mask` / `falseVal` | 可选掩码与回退值 | 可选掩码 |
| `cache` | LoadCachePolicy（默认 `L2_CACHE_HINT_NORMAL_FV`） | StoreCachePolicy（同默认） |

```mlir
%val = ascend_dpx.load %ptr : !llvm.ptr<f32> -> f32
%val = ascend_dpx.load %ptr, %mask, %false cacheModifier = <L2_CACHE_HINT_NORMAL_CG> : !llvm.ptr<f32>, i1, f32 -> f32
ascend_dpx.store %ptr, %val : !llvm.ptr<f32>, f32
```

## Warp 操作

- **同步**：`ascend_dpx.sync_threads`（线程间屏障）。
- **Warp Shuffle**（统一签名 `(src, lane_mask, clamp, index) -> res`）：`shfl.up`（从低通道取值）、`shfl.down`（从高通道取值）、`shfl.bfly`（按位异或蝶形）、`shfl.idx`（按索引取值）。
- **Warp 归约**（单操作数）：`reduce.add`、`reduce.max/min`（有符号）、`reduce.umax/umin`（无符号）。

```mlir
%result = ascend_dpx.shfl.up %src, %mask, %clamp, %idx : (f32, i32, i32, i32) -> f32
```

## 原子操作

通用原子操作统一签名 `(src: ptr, data, cache) -> res`（返回原始值）：

| 类别 | 操作 |
|------|------|
| 位运算 | `atomic_and` / `atomic_or` / `atomic_xor` |
| 算术 | `atomic_add` / `atomic_sub` / `atomic_inc` / `atomic_dec` |
| 最值 | `atomic_max` / `atomic_min`（有符号）、`atomic_umax` / `atomic_umin`（无符号） |
| 交换 | `atomic_exchange` |
| CAS | `atomic_cas(src, expected, new_val, cache) -> old` |

```mlir
%old = ascend_dpx.atomic_cas %ptr, %expected, %new_val : (!llvm.ptr<i32>, i32, i32) -> i32
```

## 类型转换

`ascend_dpx.cast %in kind <...>`，CastKind 枚举：`si_to_fp`（有符号转浮点）、`ui_to_fp`（无符号转浮点）、`fp_to_si`、`fp_to_ui`。

```mlir
%out = ascend_dpx.cast %in kind <si_to_fp> : i32 to f32
```

## 数学函数

- **一元**（单操作数）：`atan`、`ceil`、`cos`、`erf`、`exp`、`exp2`、`floor`、`ilogb`、`log`、`log1p`、`log2`、`recip`、`relu`、`rint`、`round`、`rsqrt`、`sin`、`sqrt`、`tan`、`tanh`。
- **二元**：`div`、`udiv`（无符号整数除）、`pow`、`umulhi`（无符号乘高位）、`ldexp`。
- **数值检查**（返回 i1）：`isfinite`、`isinf`、`isnan`。

## L2 缓存策略（LoadCachePolicy / StoreCachePolicy）

两组枚举按五族组织：加载 15 个有效值、存储 16 个有效值，各加 `INVALID`：

| 族 | 加载后缀 | 存储后缀 | 含义 |
|----|----------|----------|------|
| NORMAL | `_FV/_LV/_PERS/_PREF` | `_FV/_LV/_PERS/_RED` | 正常访问（首次/末次有效、持久化、预取/归约） |
| NOTALLOC | `_KEEP/_CLEAN/_DROP` | `_CI/_PW/_PI/_RED` | 不分配缓存行 |
| IDS | `_FV/_LV/_PERS/_PREF` | — | IDS 策略（仅加载） |
| EXCLUSIV | `_FV/_LV/_PERS/_PREF` | — | 独占策略（仅加载） |
| WBH / WTS | — | `_FV/_LV/_PERS/_RED` | 写回 / 写穿透提示（仅存储） |

## 地址空间

| 枚举值 | 值 | 说明 |
|--------|---|------|
| `GLOBAL_MEM` | 1 | 全局内存 |
| `SHARED_MEM` | 3 | 共享内存 |

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [AscendDPXOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXOps.td)
- [AscendDPXBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXBase.td)
- [AscendDPXAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXAttrs.td)
