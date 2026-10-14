---
type: API
title: HIVM → 后端转换（SIMD / SIMT / MIX 三条路径）
description: HIVM 到 LLVM/TritonGPU 的三条 lowering 路径：SIMD（ConvertHIVMToTritonGPU + ConvertTritonAscendGPUToLLVM）、SIMT（ArithToHIVMLLVM
  + GPUToDPX + FuncToLLVM）、MIX（SplitMixKernel 分裂 Cube/Vector 双 kernel 经共享 workspace
  通信）。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM → 后端转换

HIVM 变换阶段（见 [hivm_transforms.md](hivm_transforms.md)）完成后，按执行模式分三条路径 lowering 到 LLVM IR。路径选择由 `pure-simt`、`enable-mixed-cv`、`use-dpx` 等 pipeline 选项控制（见 [pipeline_options.md](pipeline_options.md)）。

```
                    HIVM Dialect
          ┌───────────┼────────────┐
       SIMD 路径    SIMT 路径    MIX 路径
   ConvertHIVMTo  ArithToHIVM   SplitMixKernel
   TritonGPU      LLVMConversion     │
          │            │        Cube 部分 + Vector 部分
   TritonGPU Passes  GPUToDPX      │           │
          │            │        SIMD 路径    SIMT 路径
   ConvertTritonAscendGPUToLLVM    │           │
          └────────────┴──────────┴───────────┘
                       LLVM IR
```

## SIMD 路径：HIVM → TritonGPU → LLVM

### ConvertHIVMToTritonGPU（`convert-hivm-to-tritongpu`）

复用 TritonGPU 的布局编码系统表达 Cube 矩阵运算。依赖 triton、triton_gpu、arith、scf、tensor、memref、hivm。TritonGPU 操作与编码见 [tritongpu_ops](../triton_dialects/tritongpu_ops.md) 与 [tritongpu_encodings](../triton_dialects/tritongpu_encodings.md)。

| HIVM 操作 | TritonGPU 操作 |
|-----------|---------------|
| Cube 矩阵乘（`hir.matmul` 族） | `tt.dot` + `ttg.convert_layout` |
| DMA 搬运（`hir.load/copy`） | `ttg.async_copy_global_to_local` |
| 向量操作 | `ttg.local_load/store` + arith |
| 同步（set/wait_flag） | `ttg.async_wait` |

布局编码映射：Fractal（zN/nZ）→ `FractalSharedEncodingAttr`（昇腾 Cube 格式）；ND → `BlockedEncodingAttr`。

### ConvertTritonAscendGPUToLLVM（`convert-triton-ascend-gpu-to-llvm`）

标准 `ConvertTritonGPUToLLVM` 的昇腾适配版，把 TritonGPU 操作映射到昇腾 LLVM 内联汇编。输出含 Cube 矩阵乘内联汇编、共享内存管理、Fractal↔ND 布局转换与同步代码。

## SIMT 路径：HIVM → LLVM (via DPX)

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| ArithToHIVMLLVMConversionPass | `convert-arith-to-hivm-llvm` | arith → HIVM LLVM，支持双缓冲 |
| GPUToDPX | `convert-gpu-to-dpx` | gpu → DPX（Data Processing eXtension）操作，面向昇腾 Vector 单元；DPX 方言见 [ascend_dpx_dialect](../other_dialects/ascend_dpx_dialect.md) |
| FuncToLLVM | `convert-func-to-llvm` | 函数 → LLVM |

SIMT 路径输出含 Vector 向量指令内联汇编、标量代码、内存访问与 DPX 指令。纯 SIMT 也可用 ConvertHFusionToVector（`convert-hfusion-to-vector`）跳过 HIVM 的 Cube 映射。

## MIX 路径：SplitMixKernel（`hivm-split-mix-kernel`，ModuleOp）

把 Mix kernel（`tcore_type = CUBE_OR_VECTOR`）分裂为独立的 AICube 与 AIVector 函数：

```mlir
// 分裂前：单函数混合 cube_op / vector_op，经 workspace 传中间结果
// 分裂后：
func @mix_kernel_cube(workspace)   {tcore_type = #hivm.tcore_type<CUBE>}   ; annotation.mark 防 DCE
func @mix_kernel_vector(workspace) {tcore_type = #hivm.tcore_type<VECTOR>}
```

Cube 函数走 SIMD 路径、Vector 函数走 SIMT 路径分别编译，运行时通过共享 workspace 传递数据。分裂前的 CV 数据搬运与紧耦合缓冲由 HIVM 变换阶段的 Mix Kernel Pass 组插入（见 [hivm_transforms.md](hivm_transforms.md)），workspace 管理见 [workspace_management](../memory_management/workspace_management.md)。

## 辅助转换 Pass

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| LowerMesh | `lower-mesh` | Mesh → HCCL/LCCL 通信操作 |
| LowerMemRefExt | `lower-memref-ext` | memref_ext → 标准 memref（memref_ext 方言见 [memrefext_dialect](../other_dialects/memrefext_dialect.md)） |

## 转换依赖关系

- SIMD：`ConvertHIVMToTritonGPU` → TritonGPU Transforms（coalesce、pipeline 等，见 [triton_ir_compilation.md](triton_ir_compilation.md)）→ `AllocateSharedMemory` → `ConvertTritonAscendGPUToLLVM` → LLVM IR。
- SIMT：`ArithToHIVMLLVMConversionPass` → `GPUToDPX` → `FuncToLLVM` → LLVM IR。
- MIX：`SplitMixKernel` 后 Cube 函数接 SIMD 链、Vector 函数接 SIMT 链。

路径与 Triton 原生编译链的对比见 [triton_ir_compilation.md](triton_ir_compilation.md)；各 Pass 的顺序约束见 [pass_dependencies.md](pass_dependencies.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Conversion/Passes.td)
