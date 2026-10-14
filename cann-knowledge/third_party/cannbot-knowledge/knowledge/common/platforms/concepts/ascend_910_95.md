---
type: Concept
title: Ascend 910_95（dav-c310）硬件速查
description: Ascend 910_95/950PR/950DT 系列（dav-c310，Reg-based A5 架构）的核数、UB/L1/L0C 容量、对齐约束、专有数据通路与
  910B 差异速查。
tags: [task.lookup, topic.architecture, topic.implementation.memory_management, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/cv_autotune/hardware_consts.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---
# Ascend 910_95（dav-c310）硬件速查

适用：目标设备为 Ascend910_95 / 950PR / 950DT 系列（架构代号 `dav-c310`，**Reg-based A5**）时的内存分配、Pipeline 选择、SIMT VF 模式判断。**本卡所有数值均为 910_95 平台值，与 910B（dav-c220，Mem-based A2/A3）显式分列，禁止混用。**

## AI Core 架构

```text
每个 AI Core = 1 Cube (AIC) + 2 Vector (AIV) + Scalar
VectorCoreCount = 2 * CubeCoreCount = 2 * AiCoreCount
```

## 910_95 系列型号规格表

| 型号 | AI Core | Cube | Vector | UB | DCache | L1 | L0A | L0B | L0C | Arch |
|------|---------|------|--------|-----|--------|-----|-----|-----|-----|------|
| Ascend910_950z | 4 | 4 | 8 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9579/957b/957d | 28 | 28 | 56 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9581/9589/958a/958b | 32 | 32 | 64 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9599 | 36 | 36 | 72 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |

> UB 248KB = 256KB - 8KB（预留 8KB 给编译器），源码值 `UbSize=2031616 bits`。

## 内存层次速查（910_95 vs 910B）

| 层次 | IR 标识符 | 枚举值 | 910_95 | 910B | 对齐 | 所属单元 |
|------|-----------|--------|--------|------|------|----------|
| GM | `gm` | 1 | HBM/L2 | HBM/L2 | - | 共享 |
| L1 | `cbuf` | 2 | 512KB | 512KB | 32B | Cube |
| L0A | `ca` | 3 | 64KB | 64KB | - | Cube A 端 |
| L0B | `cb` | 4 | 64KB | 64KB | - | Cube B 端 |
| L0C | `cc` | 5 | **256KB** | 128KB | 512B | Cube C 端 |
| UB | `ub` | 6 | **248KB** | 192KB | 32B | Vector |
| DCache | - | - | **32~120KB** | 无 | - | SIMT Vector |

对齐要求（所有架构一致）：UB 32B（256 bits）、L1 32B、L0C 512B（4096 bits）。

专用缓冲区：BT Buffer（BiasTable，1KB，64B 对齐，`copy_cbuf_to_bt`）；FP Buffer（FixPipe，7KB，128B 对齐，`hivm.fixpipe` 隐式使用）。

## 910_95 与 910B 关键差异

| 特性 | 910B (dav-c220) | 910_95 (dav-c310) |
|------|-----------------|-------------------|
| 架构类型 | Mem-based (A2/A3) | **Reg-based (A5)** |
| SIMT VF 模式 | 不支持 | **支持**（InferVFMode 推断 SIMD/SIMT/MIX） |
| UB / L0C | 192KB / 128KB | 248KB / 256KB |
| DCache | 无 | 32~120KB |
| L0C -> UB 直通 | 不支持 | **支持**（FixPipe） |
| UB -> L1 通路 | 不支持 | **支持**（PIPE_MTE3） |
| 紧耦合缓冲区 | 不支持 | **支持** |
| Fixpipe Dual Dst / NZ2DN | 不支持 | 支持（ROW_SPLIT / COLUMN_SPLIT） |
| 同步方式 | FFTS（基于内存） | SetFlag/WaitFlag（基于寄存器） |
| Pipe Barrier | 对所有 Pipe 生成 barrier | **跳过 PIPE_V 的 barrier** |
| 归约标量降级 | i64 归约、整数 argmax/argmin 降级 | 基本归约不降级 |
| vcmp(NE) | 规范化为 vnot(vcmp(EQ)) | 不做规范化 |

关键通路差异（CV 融合算子性能分水岭）：

```text
910_95: GM -> L1 -> L0A/L0B -> L0C -> UB -> V -> UB -> GM
                                       L0C 直通 UB（省 GM 中转）
910B:   GM -> L1 -> L0A/L0B -> L0C -> GM -> UB -> V -> UB -> GM
                                       必须经 GM 中转
```

## Pipeline 与源-目标映射

| 枚举 | 数值 | 数据流 | 典型操作 |
|------|------|--------|----------|
| PIPE_S | 0 | 标量 | 循环控制 |
| PIPE_V | 1 | UB -> UB | `vadd`, `vmul`, `vcast`, `vreduce` |
| PIPE_M | 2 | L0A/L0B -> L0C | `mmadL1` 矩阵乘 |
| PIPE_MTE1 | 3 | L1 -> L0A/L0B/BT | `mmadL1` 加载, `l12ub` |
| PIPE_MTE2 | 4 | GM <-> L1/UB | `load`, `nd2nz` |
| PIPE_MTE3 | 5 | UB -> GM/L1 | `store`, `nz2nd`, UB->L1（仅 950） |
| PIPE_FIX | 10 | L0C -> GM/L1/UB | `fixpipe`（L0C->UB 仅 950） |

> 其余枚举：PIPE_ALL(6), PIPE_MTE4(7), PIPE_MTE5(8), PIPE_V2(9), VIRTUAL_PIPE_MTE2_L1A(11)/L1B(12), PIPE_NUM(13), PIPE_UNASSIGNED(99)。

## 紧耦合缓冲区（910_95 特有）

IR 表示 `#hivm.tightly_coupled_buffer<id : optional<i32>>`，两种模式：

| 模式 | 数据流向 | 说明 |
|------|---------|------|
| MoveToUb | L0C -> UB | Cube 结果直通 Vector 工作区 |
| MoveToL1 | UB -> L1 | Vector 结果回传 Cube 输入缓存 |

Pipeline 选择逻辑：950 且开启布局优化 → `InsertCVDataMovement`；950 未开启 → `InsertCVTightCoupledBuffer`；非 950 → `InsertLoadStoreForMixCV`（必须 GM 中转）。

## VFMode 与核心类型枚举（仅 Reg-based 使用 VFMode）

| VFMode | 说明 |
|--------|------|
| SIMD(0) | 传统 Vector 执行，基于 UB |
| SIMT(1) | 类 GPU 线程级并行，基于寄存器 |
| MIX(2) | 混合模式，`--enable-simd-simt-mix-compile` 启用 |

操作级 `TCoreType`：`CUBE` / `VECTOR` / `CUBE_OR_VECTOR` / `CUBE_AND_VECTOR`（需 CV 同时执行）。函数级 `TFuncCoreType`：`AIC` / `AIV` / `MIX`。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
