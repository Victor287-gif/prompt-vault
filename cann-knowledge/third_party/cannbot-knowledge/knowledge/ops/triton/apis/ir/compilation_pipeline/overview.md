---
type: API
title: 编译流水线总览（TTIR → HFusion → HIVM → LLVM）
description: AscendNPU-IR 编译数据流全景：前端到 HFusion 的六个转换 Pass、HFusion/HIVM 两层变换、SIMD/SIMT/MIX
  三条后端路径与关键转换 Pass（ConvertHFusionToHIVM、ConvertHIVMToTritonGPU、SplitMixKernel）。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# 编译流水线总览

AscendNPU-IR 存在两条主编译路径：**Triton 路径**（Triton IR 直接编译到 LLVM，用于标准 Triton kernel）与 **BiShengIR 路径**（经 HFusion → HIVM 两级方言降到 LLVM，用于昇腾 NPU 优化 kernel）。两路径在 SIMD 路径中交汇：HIVM 可转回 TritonGPU 复用其布局编码系统。IR 提取与落盘流程见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)。

```
Frontend（Triton Python / Torch / Linalg / Arith / Math / GPU / Tensor）
   │ Triton 路径                     │ BiShengIR 路径
   ▼                                 ▼
Triton IR (tt)                   HFusion（算子融合与调度，40+ 变换 Pass）
   │ ConvertTritonToTritonGPU       │ ConvertHFusionToHIVM
   ▼                                 ▼
TritonGPU IR (ttg)               HIVM（硬件指令映射，70+ 变换 Pass）
   │ TritonGPUToLLVM                │ SIMD: HIVM→TritonGPU→LLVM
   ▼                                 │ SIMT: HIVM→LLVM (DPX)
LLVM IR                           │ MIX:  SplitMixKernel 分裂 Cube/Vector
                                    ▼
                                 LLVM / CCE 指令 → Ascend NPU 二进制
```

## 三条后端路径

| 路径 | 走向 | 适用场景 | 关键 Pass |
|------|------|----------|-----------|
| SIMD | HIVM → TritonGPU → LLVM | Cube 矩阵乘密集 | `ConvertHIVMToTritonGPU`，布局用 `FractalSharedEncodingAttr` 适配 Cube |
| SIMT | HIVM → LLVM (via DPX) | 逐元素、归约等 Vector 密集 | `ConvertTritonAscendGPUToLLVM`、`GPUToDPX` |
| MIX | HIVM → SplitMixKernel → Cube/Vector 双 kernel | Cube+Vector 混合 kernel | `SplitMixKernel` 分裂后分别走 SIMD/SIMT，共享 workspace 传数据 |

详见 [hivm_to_backend.md](hivm_to_backend.md)。

## 方言转换 Pass 速查

| 转换 | Pass（CLI 参数） | 详解卡 |
|------|------------------|--------|
| arith/math/linalg/gpu/tensor/torch → HFusion | `convert-{arith,math,linalg,gpu,tensor,torch}-to-hfusion` | [frontend_to_hfusion.md](frontend_to_hfusion.md) |
| HFusion → HIVM | `convert-hfusion-to-hivm`（`mm-map-mode`: `core_op`/`macro_instr`） | [hfusion_to_hivm.md](hfusion_to_hivm.md) |
| HIVM → TritonGPU | `convert-hivm-to-tritongpu`（SIMD） | [hivm_to_backend.md](hivm_to_backend.md) |
| arith → HIVM LLVM | `convert-arith-to-hivm-llvm`（SIMT，双缓冲支持） | [hivm_to_backend.md](hivm_to_backend.md) |
| memref_ext → memref | `lower-memref-ext` | [hivm_to_backend.md](hivm_to_backend.md) |
| Triton → TritonGPU | `convert-triton-to-tritongpu` | [triton_ir_compilation.md](triton_ir_compilation.md) |

方言本身：HFusion 见 [overview](../hfusion/overview.md)，HIVM 见 [overview](../hivm/overview.md)，Triton/TritonGPU 见 [overview](../triton_dialects/overview.md)。

## Pipeline 入口函数

```cpp
void buildHFusionPipelines(OpPassManager &pm, const HFusionPipelineOptions &options);
void buildConvertToHIVMPipeline(OpPassManager &pm, const ConvertToHIVMPipelineOptions &options);
void buildHIVMTensorOptimizations(OpPassManager &pm, const HIVMPipelineOptions &options);
void buildLowerHIVMPipelines(OpPassManager &pm, const HIVMPipelineOptions &options);
```

选项全集见 [pipeline_options.md](pipeline_options.md)，各入口内部 pass 顺序见 [pass_dependencies.md](pass_dependencies.md)。

## 两级方言的设计动机

- **HFusion 层**：算子融合减少内存带宽需求、自动调度选择执行策略、自动向量化、多 kernel 分裂支持。变换清单见 [hfusion_transforms.md](hfusion_transforms.md)。
- **HIVM 层**：高级算子到 Cube/Vector/DMA 指令的硬件抽象、内存分配/多缓冲/同步规划、ND/Fractal 数据布局推断、自动同步注入。变换清单见 [hivm_transforms.md](hivm_transforms.md)。

`last_pass.mlir`（提取流程见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)）对应 BiShengIR 路径的最末阶段：内容为 HIVM 同步注入之后到 LLVM lowering 之前的 IR，含 `hivm.hir.*` 或 `llvm.*` 操作。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Conversion/Passes.td)
- [Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/Pipelines/Passes.h)
- [Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Pipelines/Passes.h)
