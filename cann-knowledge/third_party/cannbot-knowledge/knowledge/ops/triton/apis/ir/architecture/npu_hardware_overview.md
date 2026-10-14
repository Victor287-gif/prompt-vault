---
type: API
title: NPU 硬件架构总览（AI Core 组成、型号规格、核心类型枚举、Reg-based/Mem-based）
description: Ascend NPU 的 AI Core 组成（1 Cube + 2 Vector）、全型号规格表（910B/910_93/310B/910_95/950PR/950DT）、TCoreType/VFMode
  等核心类型枚举及 Reg-based 与 Mem-based 架构分类，是理解 HIVM IR 硬件映射的入口。
tags: [task.lookup, topic.architecture, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---

# NPU 硬件架构总览

华为昇腾 NPU 面向 AI 计算，通过多级存储层次 + 专用计算单元实现高吞吐矩阵与向量计算。AscendNPU-IR 用 HACC 和 HIVM 两个方言把硬件特征映射进 MLIR IR：型号规格由 `NPUTargetSpec.td` 的 TableGen 定义描述，编译时经 `hacc.DeviceSpec` 属性传入 IR。

## AI Core 组成

| 组件 | 功能 | 数量关系 |
|------|------|----------|
| AIC（AI Cube） | 含 Cube 单元，做矩阵乘法 | 每 AI Core 1 个 |
| AIV（AI Vector） | 含 Vector 单元，做向量计算 | 每 AI Core 2 个 |
| Scalar | 标量计算单元，带 DCache/ICache | 每 AIC/AIV 各 1 个 |

核心数量恒等式：`VectorCoreCount = 2 × CubeCoreCount = 2 × AiCoreCount`。

## 型号规格速查（按系列 BaseSpec）

| 系列 | Arch 代号 | UB | L1 | L0A/L0B | L0C | 特点 |
|------|-----------|-----|-----|---------|-----|------|
| Ascend910B（B1-B4） | dav-c220 | 192KB | 512KB | 64KB | 128KB | B1/B2 24 核，B3/B4 20 核 |
| Ascend910_93（9362-9392） | dav-c220 | 192KB | 512KB | 64KB | 128KB | 9362/9372 20 核，9381+ 24 核 |
| Ascend310B（B1-B4） | dav-m300 | 256KB | 1024KB | 64KB | 128KB | 单核（1/1/1），Reg-based |
| Ascend910_95 / 950PR / 950DT | dav-c310 | 248KB（预留 8KB 给编译器） | 512KB | 64KB | 256KB | 带 DCache 32~120KB，L0C→UB 直通 |

950 系列按尾缀分档核心数：950z=4 核、957x=28 核、958x=32 核、959x=36 核；950DT 另有 950x/950y=8 核。逐型号完整表见 [hardware_specs_ir.md](../reference/hardware_specs_ir.md) 与上游 resource。

## 核心类型枚举（HIVM 属性）

| 枚举 | IR 语法 | 取值 | 用途 |
|------|---------|------|------|
| TCoreType | `#hivm.tcore_type<CUBE>` | CUBE(1) / VECTOR(2) / CUBE_OR_VECTOR(3) / CUBE_AND_VECTOR(4) | 标注单个操作的执行核心 |
| TFuncCoreType | `#hivm.func_core_type<AIV>` | AIC(1) / AIV(2) / MIX(3) / AIC_OR_AIV(4) | 标注函数运行的核心 |
| TModuleCoreType | `#hivm.module_core_type<MIX>` | AIC / AIV / MIX | 由模块内各函数类型推断 |
| VFMode | `#hivm.vf_mode<SIMD>` | SIMD(0) / SIMT(1) / MIX(2) | Vector Function 执行模式 |

`CUBE_OR_VECTOR` = 任选一核执行；`CUBE_AND_VECTOR` = 两核同时参与。模块级推断规则：全 AIV→AIV，全 AIC→AIC，否则 MIX。

## Reg-based 与 Mem-based 架构分类

编译器按**是否支持 SIMT VF 模式**把硬件分两类（`isRegBasedArch` / `isMemBasedArch`，IR 未指定设备时默认按 910B 处理）：

| 特性 | Mem-based（A2/A3） | Reg-based（A5） |
|------|--------------------|------------------|
| 对应芯片 | 910B / 910_93（dav-c220） | 310B / 950（dav-m300 / dav-c310） |
| SIMT VF | 不支持，跳过 InferVFMode | 支持，SIMT 走寄存器 + Triton GPU 编译路径 |
| 数据访问 | 向量操作全部基于 UB | SIMT 基于寄存器（LocalLoad/LocalStore），SIMD 基于 UB |
| 核间同步 | 基于内存的 FFTS + SetCrossCoreInstrOp | 寄存器级 SetFlag/WaitFlag |
| Pipe Barrier | 对所有 Pipe 生成 barrier | 跳过 PIPE_V 的 barrier |
| 归约 | i64 归约、整数 argmax/argmin 会标量降级 | 基本归约不降级（argmax/argmin 对齐问题除外） |
| 其他 | vcmp(NE) 规范化为 vnot(vcmp(EQ))；VReduce 需额外 buffer | 无 CmpVne 规范化；stride 对齐更严格；SIMT 下动态调整 UB |

对 Triton 算子优化的直接影响：Mem-based 上 i64 归约和整数 argmax/argmin 标量降级损失大，需改 dtype 或换算法；SIMT 仅 310B/950 可用。

## 硬件概念 → IR 映射

| 硬件概念 | IR 表示 |
|----------|---------|
| NPU 型号规格 | `hacc.DeviceSpec` 属性（`#hacc.target_device_spec<...>`） |
| 内存空间 | `#hivm.address_space<gm/cbuf/ca/cb/cc/ub>`，详见 [memory_hierarchy.md](memory_hierarchy.md) |
| 执行流水线 | `#hivm.pipe<PIPE_V/PIPE_M/...>`，详见 [pipeline_execution_model.md](pipeline_execution_model.md) |
| 数据布局 | `#hivm.data_layout<ND/nZ/zN/...>`，详见 [data_layout.md](data_layout.md) |
| 紧耦合缓冲区 | `#hivm.tightly_coupled_buffer<id>`，详见 [tightly_coupled_buffer.md](../memory_management/tightly_coupled_buffer.md) |

950 UB 为 248KB 的原因：源码注释明确 `reserve 8KB for compiler`（2031616 bits）。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- [NPUTargetSpec.td#L64-L98](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L64-L98)
- [NPUTargetSpec.td#L104-L150](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L104-L150)
- [NPUTargetSpec.td#L156-L190](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L156-L190)
- [NPUTargetSpec.td#L196-L262](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L196-L262)
- [NPUTargetSpec.td#L264-L346](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L264-L346)
- [NPUTargetSpec.td#L348-L490](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L348-L490)
- [NPUTargetSpec.td#L64-L74](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L64-L74)
- [NPUTargetSpec.td#L196-L208](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L196-L208)
- [HIVMAttrs.td#L298-L317](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L298-L317)
- [HIVMAttrs.td#L250-L269](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L250-L269)
- [HIVMAttrs.td#L271-L292](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L271-L292)
- [HIVMAttrs.td#L948-L960](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L948-L960)
- [HIVMAttrs.td#L171-L197](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L171-L197)
- [HIVMAttrs.td#L203-L244](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L244)
- [HIVMAttrs.td#L1010-L1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017)
- [HIVMAttrs.td#L84-L165](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L84-L165)
- [Utils.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HACC/Utils/Utils.cpp)
- [Utility.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
