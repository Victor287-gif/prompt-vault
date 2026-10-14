---
type: API
title: 术语表（AscendNPU-IR 核心概念中英文对照）
description: AscendNPU-IR 项目核心术语的中英对照与一句话释义，覆盖硬件架构、内存体系、数据布局、编译器 IR 概念、方言名称、内存管理、流水线同步、编译管线与操作类型八个分组。
tags: [task.lookup, topic.compilation, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# 术语表

AscendNPU-IR 核心概念中英对照速查。深读入口：硬件规格见 [hardware_specs_ir.md](hardware_specs_ir.md)，内存管理术语见 [overview](../memory_management/overview.md)。

## 硬件架构

| 英文 | 中文 | 说明 |
|------|------|------|
| AI Core | AI 核心 | NPU 基本计算单元，含 Cube Core 和 Vector Core |
| Cube Core | Cube 核心（矩阵计算单元） | 执行矩阵乘法（MAD），对应 CUBE CoreType |
| Vector Core | Vector 核心（向量计算单元） | 执行元素级向量运算，对应 VECTOR CoreType |
| AI Vector (AIV) | 纯向量 AI Core | 仅含 Vector Core 的 AI Core |
| Mix | 混合模式 | Cube 与 Vector 协同工作模式，对应 MIX CoreType |
| SubBlock | 子块 | AI Core 内逻辑分区，Mix CV 场景并行用 |
| CoreDim | 核心维度 | 描述核心数量的维度参数 |

## 内存体系

| 英文 | 中文 | 说明 |
|------|------|------|
| Memory Space | 地址空间 | NPU 内存逻辑分区：GM/UB/L1/L0A/L0B/L0C |
| GM (Global Memory) | 全局内存 | 所有 AI Core 共享的外部存储（AddressSpace.GM） |
| UB (Unified Buffer) | 统一缓冲区 | Vector Core 的片上存储（AddressSpace.UB） |
| L1 (Cube Buffer / CBUF) | L1 缓冲区 | Cube Core 的片上存储（AddressSpace.L1） |
| L0A / L0B | L0A/L0B 缓冲区 | 矩阵乘输入 A/B 的缓冲区 |
| L0C | L0C 缓冲区 | 矩阵乘累加结果缓冲区 |
| DCache | 数据缓存 | SIMT VF 模式数据缓存，950 系列支持 |
| FFTS (Fast Thread Schedule System) | 快速线程调度系统 | 硬件级线程调度机制，经 set_ffts_base_addr 配置 |

## 数据布局

| 英文 | 中文 | 说明 |
|------|------|------|
| Data Layout | 数据布局 | 数据在内存中的排列方式：ND/nZ/zN/Fractal |
| ND Layout | ND 布局 | 标准 N 维行优先布局 |
| nZ / zN Layout | nZ / zN 布局 | NPU 特有分块布局（zN 为其转置排列） |
| Fractal Layout | Fractal 布局 | 矩阵乘法专用分形布局 |
| DOTA_ND / DOTB_ND / DOTC_ND | 矩阵乘输入/输出布局 | 操作数 A/B/C 的专用 ND 布局标记 |

## 编译器 IR

| 英文 | 中文 | 说明 |
|------|------|------|
| IR | 中间表示 | 编译器内部程序表示形式 |
| Dialect | 方言 | MLIR 中一组相关操作与属性的集合 |
| Operation (Op) | 操作 | IR 基本计算单元，如 hivm.hir.vadd |
| Attribute (Attr) | 属性 | 编译期已知常量信息，附加在操作或值上 |
| Trait | 特征 | 操作的编译期属性标记，如 SinglePipeOpTrait |
| Interface | 接口 | 操作必须实现的方法集合，如 BufferizableOpInterface |
| MemRef | 内存引用 | MLIR 缓冲区类型，表示内存区域引用 |
| Tensor | 张量 | MLIR 张量类型，值语义、不可变 |
| SSA | 静态单赋值 | 每个变量只赋值一次的 IR 形式 |

## 方言名称

| 英文 | 说明 |
|------|------|
| HIVM (Huawei Intermediate Virtual Machine) | NPU 底层虚拟机 IR，直接映射硬件操作 |
| HACC (Huawei Accelerator) | 主机-设备交互与硬件规格标注 |
| HFusion | 混合融合 IR，高层可融合操作表示 |
| Scope / Symbol | 作用域管理 / 符号化维度（动态形状） |
| MemRefExt / MathExt | MemRef 扩展（workspace）/ 数学扩展（ilogb/ldexp） |
| HMAP | 多处理器通信（all_to_all_v） |
| Annotation | 编译器标注操作（annotation.mark） |
| AscendDPX | SIMT 编程模型的设备操作 |
| Triton / TritonGPU / Gluon | Triton 语言 op / Triton 后端 op / 布局自动推导 |

## 内存管理

| 英文 | 中文 | 说明 |
|------|------|------|
| Bufferization | 缓冲区化 | Tensor IR → MemRef IR 的转换 |
| Memory Planning | 内存规划 | 为逻辑缓冲区分配物理内存偏移 |
| Inplace | 原地复用 | 多缓冲区共享同一物理内存的优化 |
| Multi-Buffer / Double Buffer | 多缓冲 / 双缓冲 | 同一逻辑缓冲区多个物理实例，实现流水线重叠 |
| Tightly Coupled Buffer | 紧耦合缓冲区 | Cube-Vector 直传缓冲区，避免 GM 中转 |
| Extra Buffer | 额外缓冲区 | 操作执行所需临时缓冲区（temp_buffer） |
| Workspace | 工作空间 | 跨函数共享的全局内存分配区域 |
| Memory / Stride Alignment | 内存 / 步幅对齐 | 使分配与 stride 满足硬件对齐约束 |

## 流水线与同步

| 英文 | 中文 | 说明 |
|------|------|------|
| Pipeline | 流水线 | 指令执行流水线：S/V/M/MTE1/MTE2/MTE3/FIX |
| Pipe Barrier | 流水线屏障 | 流水线级间同步操作 |
| Unit Flag | 单元标志 | 硬件级事件通知机制，用于流水线同步 |
| Event | 事件 | 同步事件标识符（EVENT_ID0-7） |
| Sync Block / Block Sync | 同步块 / 块同步 | 跨 AI Core 的同步机制 |
| CV Pipelining | Cube-Vector 流水线化 | Cube 与 Vector 核心的流水线调度 |

## 编译管线与操作类型

| 英文 | 中文 | 说明 |
|------|------|------|
| VFMode | 向量函数模式 | VF 执行模式：SIMD/SIMT/MIX |
| SIMD / SIMT | 单指令多数据 / 单指令多线程 | SIMT 类似 GPU 编程模型 |
| Tiling | 切分 | 把大计算任务分解为适合硬件的小任务 |
| Fixpipe | 定点管道 | L0C 到 UB/GM/L1 的数据搬运与类型转换 |
| ND2NZ / NZ2ND | 布局转换 | 标准 ND 与 NPU 分块 NZ 布局互转 |
| Pointer Cast | 指针转换 | PlanMemory 后 memref.alloc 的物理偏移表示 |
| Structured Op | 结构化操作 | 用 iterator_type 描述迭代语义的 HIVM 操作 |
| Macro Op | 宏操作 | 封装复杂计算模式的 HIVM 操作（mmadL1/matmul） |
| Elementwise / Reduction Op | 逐元素 / 归约操作 | 逐元素独立计算 / 沿维度聚合 |
| Broadcast | 广播 | 小维度数据扩展到大维度 |
| VCast | 向量类型转换 | Vector Core 上的 dtype 转换 |
| Descale | 反缩放 | 量化矩阵乘中的反量化操作 |
