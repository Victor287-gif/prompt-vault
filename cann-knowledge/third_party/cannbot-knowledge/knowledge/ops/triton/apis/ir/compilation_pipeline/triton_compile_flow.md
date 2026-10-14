---
type: API
title: Triton-Ascend 编译流程主线（Python → TTIR → Linalg → NPU 二进制）
description: 面向新手的 Triton-Ascend 编译主线：AST→TTIR→TTIR 优化→Ascend Passes→Linalg IR→BiSheng
  Compiler 的数据流、三条后端路径（A2/A3、910_95、SIMT Only）、编译缓存机制与中间 IR dump 调试入口。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---

# Triton-Ascend 编译流程主线

本卡面向新手讲清编译主线；阶段划分、各阶段输入/输出格式与路径选择代码的分析视角版本见 [triton_pipeline_overview.md](triton_pipeline_overview.md)，BiShengIR 侧（HFusion/HIVM 内部）总览见 [overview.md](overview.md)。

核心路径：`Python AST → TTIR → TTIR 优化 → Ascend Passes → Linalg IR → BiSheng Compiler → NPU 二进制`。

## 关键概念

| 概念 | 说明 |
|------|------|
| TTIR | Triton Tensor IR，基于 `tt` 方言的高层 IR |
| Linalg IR | MLIR 线性代数方言，Ascend 编译路径的核心 IR（落盘为 `kernel.ttadapter.mlir`） |
| Ascend Passes | TTIR→Linalg 的转换与优化 Pass 集合 |
| HFusion / HIVM | 昇腾算子融合方言 / 异构指令虚拟机方言（BiSheng Compiler 内部使用） |
| BiSheng Compiler | 毕昇编译器（`bishengir-compile`），Linalg/HFusion IR → NPU 二进制 |

## 主流水线

```
Python Kernel (@triton.jit)
  → Python AST
  → TTIR 生成（tt 方言）                # 见 ttir_generation.md
  → TTIR 优化 make_ttir                  # Inliner/Combine/Canonicalizer/ReorderBroadcast/CSE/LICM/SymbolDCE/LoopUnroll，见 ttir_optimization.md
  → Ascend Passes ttir_to_linalg         # AutoBlockify → [DAG 调度] → TritonToStructured → DiscreteMaskAccessConversion
                                         #   → TritonToAnnotation → TritonToUnstructured → TritonToHIVM/HFusion/LLVM
                                         #   → BubbleUpOperation → TritonToStructured(二次) → TritonToLinalg，见 ascend_passes.md
  → Linalg IR（linalg/memref/arith/scf）
  → bishengir-compile（按平台三选一）
  → NPU Binary（kernel.o / kernel_reloc.o）
```

各 Ascend Pass 作用清单见 [ascend_passes.md](ascend_passes.md)，TritonToLinalg 转换器表见 [triton_to_linalg.md](triton_to_linalg.md)。

## 三条后端路径

| 路径 | 平台 | 入口函数 | 关键 bishengir-compile 参数 |
|------|------|----------|------------------------------|
| A: A2/A3（910B） | Ascend910B 等 | `linalg_to_bin_enable_npu_compile_A2_A3` | `--enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True` |
| B: 910_95 | Ascend910_95xx（不支持 FFTS） | `linalg_to_bin_enable_npu_compile_910_95` | `--enable-hfusion-compile=true --enable-triton-kernel-compile=true`（不启用 hivm-compile 开关） |
| C: SIMT Only | 通用 | `ttir_to_npubin`（跳过 Linalg，直接从 TTIR 编译） | `--enable-hivm-compile=false --enable-triton-ir-compile --pure-simt --num-warps=4 --threads-per-warp=32` |

路径选择逻辑（`add_stages`）：`force_simt_only` → 路径 C；否则走 `ttadapter` 阶段后按 `compile_on_910_95` 选路径 B/A。命令细节与元数据回调见 [linalg_to_binary.md](linalg_to_binary.md)。

## 编译缓存

- 默认缓存目录 `~/.triton/cache`，可用 `TRITON_CACHE_DIR` 自定义。
- 缓存键 = kernel 源码 + 参数签名 + NPUOptions 全部字段 + 目标架构 的哈希；任一变化即重编译。
- `TRITON_ALWAYS_COMPILE=1` 强制绕过缓存。

## 调试 dump 入口

`TRITON_DEBUG=1`（或 `debug=True`）后，各阶段 IR 落盘到缓存目录：

| 文件 | 内容 |
|------|------|
| `kernel.ttir.mlir` | 优化后的 TTIR |
| `kernel.ttadapter.mlir` | Ascend Passes 后的 Linalg IR |
| `kernel.npuir.mlir` | BiSheng Compiler 输出的 NPU IR（编译失败时也会保存） |
| `kernel.llir.mlir` / `kernel.ll` | LLVM MLIR / LLVM IR（仅 CPU 后端） |

完整的 IR 提取流程（含 `last_pass.mlir`）见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)。排查顺序：`kernel.ttir.mlir` → `kernel.ttadapter.mlir` → `kernel.npuir.mlir`，配合 debug 输出中的 `[DEBUG] cmd_list:` 查看完整 bishengir-compile 命令行。

## NPU 适配要点

1. 编译路径按 `is_compile_on_910_95` 自动选择，无需手动干预。
2. `_check_bishengir_api_change()` / `_check_bishengir_is_regbased()` 检测 BiSheng Compiler 版本并适配 API 差异。
3. UB 溢出由 BiSheng Compiler 编译期检测报错；`--enable-print-memory-allocated-size` 打印内存分配明细。
4. 编译成功后从 `libkernel.so` 回调提取 task_type、workspace_size、lock_num 等运行时元数据。
5. CANN 运行时限制 kernel 名 ≤ 49 字符，超长截断。

## 常用环境变量速查

| 变量 | 作用 |
|------|------|
| `ASCEND_HOME_PATH` | CANN 安装路径（必须） |
| `TRITON_ASCEND_ARCH` | 手动指定架构（Ascend910B1~B4 / 910_9362~9392 / 910_9579~9599 / Ascend310B1~B4） |
| `TRITON_ALL_BLOCKS_PARALLEL` | 启用 AutoBlockify 逻辑核→物理核调整 |
| `TRITON_COMPILE_ONLY` | 仅编译不执行 |
| `TRITON_DISABLE_FFTS` | 禁用 FFTS |
| `TRITON_ENABLE_LIBDEVICE` | 启用 libdevice 链接 |

编译选项全集见 [triton_compile_options.md](triton_compile_options.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [compiler.py#L72-L93](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L72-L93)
- [compiler.py#L95-L168](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L95-L168)
- [compiler.py#L607-L797](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L607-L797)
- [compiler.py#L403-L604](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L403-L604)
- [compiler.py#L947-L989](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L947-L989)
- [compiler.py#L806-L911](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L806-L911)
- [compiler.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py)
- [architecture_design_and_core_features.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/architecture_design_and_core_features.md)
- [triton_ascend.cc](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/triton_ascend.cc)
- [utils.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/utils.py)
