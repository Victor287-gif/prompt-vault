---
type: API
title: Triton-Ascend 编译流水线阶段划分（SIMD / SIMT / 混合三路径）
description: 面向分析者的流水线阶段划分：SIMD/SIMT/混合三条编译路径的适用场景与对比、各阶段输入/输出格式、add_stages 阶段注册、缓存键构成与调试
  dump 文件清单。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---

# Triton-Ascend 编译流水线阶段划分

本卡面向分析者：讲阶段划分、路径选择与 I/O 格式。新手向的流程主线与缓存/调试入门见 [triton_compile_flow.md](triton_compile_flow.md)；BiShengIR 侧（HFusion/HIVM 内部 Pass）总览见 [overview.md](overview.md)。

路径选择由 `compile_mode` 控制：`"simd"`（默认，经 Linalg）、`"simt_only"`（TTIR 直编）、`"unstructured_in_simt"`（混合）。

## 三条编译路径对比

### 路径 A：SIMD（默认）

`TTIR → Ascend Passes → Linalg IR → HFusion IR → HIVM IR → LLVM IR → 设备二进制`

| 维度 | 说明 |
|------|------|
| 执行模型 | SIMD，Cube/Vector 核心分工 |
| 适用场景 | 大部分 kernel，尤其向量计算密集 |
| mix_mode | `aiv`（Vector）或 `aic`（Cube） |
| 关键优化 | CV 流水线、算子融合、内存复用 |
| 编译选项 | `compile_mode="simd"`（默认） |

### 路径 B：SIMT

`TTIR → TTGIR → LLVM IR → 设备二进制`

| 维度 | 说明 |
|------|------|
| 执行模型 | SIMT 线程模型 |
| 适用场景 | 控制流密集、动态索引、离散访存 |
| 关键优化 | 布局传播、线程局部性、内存合并 |
| 编译选项 | `compile_mode="simt_only"`（置 `force_simt_only=True`） |

SIMT 路径内部走 Triton 原生 TTIR→TTGIR→LLVM 链，详见 [triton_ir_compilation.md](triton_ir_compilation.md)。

### 路径 C：混合（`unstructured_in_simt`）

SIMD 部分走 HFusion/HIVM；SIMT 部分经 `AutoScope` 识别区域 → `InsertMemSemanticForSimtVF` 插内存语义 → `OutlineScope` 外联 → `SplitSimtModule` 分离 → `HIVMToTritonGPU` 进入 TritonGPU 路径，两路在 LLVM IR 层汇合。

## 各阶段输入/输出格式

| 阶段 | 输入 | 输出 | 格式说明 |
|------|------|------|----------|
| AST → TTIR | Python AST | TTIR Module | MLIR Module（`tt.func`/`tt.load` 等） |
| TTIR 优化 | TTIR Module | 优化后 TTIR Module | 同上 |
| Ascend Passes | TTIR Module | **Linalg IR 字符串** | 含 linalg/memref/arith/scf 的 MLIR 文本 |
| Linalg → Binary | Linalg IR 字符串 | NPU 二进制 bytes | `.o`，由 bishengir-compile 生成 |
| TTIR → Binary（SIMT） | TTIR Module | NPU 二进制 bytes | 直接由 bishengir-compile 编译 |
| CPU 路径 | Linalg IR 字符串 | CPU 汇编文本 | mlir-opt → LLVM IR → llc |

## 阶段注册（add_stages）

```python
def add_stages(self, stages, options):
    if self.target.backend == "npu":
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
        if options.force_simt_only:
            stages["npubin"] = lambda src, metadata: ttir_to_npubin(src, metadata, options)
            return
        stages["ttadapter"] = lambda src, metadata: ttir_to_linalg(src, metadata, options, named_ops=True)
        if options.compile_on_910_95:
            stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile_910_95(...)
        else:
            stages["npubin"] = lambda src, metadata: linalg_to_bin_enable_npu_compile_A2_A3(...)
    else:  # CPU 后端
        stages["ttir"]     = ...  # make_ttir
        stages["ttadapter"] = ... # ttir_to_linalg
        stages["llir"]     = ...  # linalg_to_llir
        stages["cpuasm"]   = ...  # llir_to_cpuasm
```

| 后端 | 阶段序列 |
|------|----------|
| NPU SIMD | `ttir`(make_ttir) → `ttadapter`(ttir_to_linalg) → `npubin`(linalg_to_bin_*) |
| NPU SIMT | `ttir` → `npubin`(ttir_to_npubin) |
| CPU | `ttir` → `ttadapter` → `llir` → `cpuasm` |

编译主流程 `compile()`：选后端 → 解析 NPUOptions → 缓存查找 → 加载 Dialect → `src.make_ir()` 生成 TTIR → 按 stages 顺序逐阶段执行 → 写缓存 → 返回 `CompiledKernel`。

## 缓存键与缓存控制

```python
key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{backend_options.hash()}-{str(sorted(env_vars.items()))}"
hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
```

| 因素 | 说明 |
|------|------|
| `triton_key()` | Triton 版本 + 编译器/后端/语言模块 SHA256 |
| `src.hash()` | 源码哈希 |
| `backend.hash()` | 后端目标（如 NPU 架构） |
| `backend_options.hash()` | NPUOptions 全字段哈希 |
| `env_vars` | 缓存失效环境变量 |

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `TRITON_ALWAYS_COMPILE` | 0 | 1 = 强制重编译 |
| `TRITON_KERNEL_DUMP` | 0 | 1 = dump 各阶段 IR |
| `TRITON_KERNEL_OVERRIDE` | 0 | 1 = 启用 IR 覆盖 |

## 调试 dump 文件

`opt.debug=True` 时按阶段落盘到缓存目录：

| 文件 | 阶段 |
|------|------|
| `kernel.ttir.mlir` | TTIR 优化后 |
| `kernel.ttadapter.mlir` | Ascend Passes 后（Linalg IR） |
| `kernel.llir.mlir` / `kernel.ll` / `kernel_linked.ll` | LLVM-MLIR / LLVM IR / 链接 libdevice 后（CPU 路径） |
| `kernel.s` | CPU 汇编 |
| `kernel.npuir.mlir` | BiSheng Compiler 输出的 NPU IR |

调试三板斧：`TRITON_KERNEL_DUMP=1` + `MLIR_ENABLE_DUMP=1` + `TRITON_DEBUG=1`。完整提取流程见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)。

## 平台差异要点

- 910B/910_95 用 `linalg_to_bin_enable_npu_compile_910_95()`，A2/A3 用 `linalg_to_bin_enable_npu_compile_A2_A3()`。
- 910_95 特有选项：`enable_hivm_auto_cv_balance`、`sync_solver` 等 CV 流水线项；A2/A3 特有：`enable_ubuf_saving`、`tile_mix_vector_loop` 等。
- `_parse_linalg_metadata()` 从 Linalg IR 提取 `mix_mode`、`kernel_name`、`tensor_kinds`；BiSheng Compiler 输出 `required_ub_bits` 供 Inductor autotune。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [compiler.py#L1058](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L1058)
- [compiler.py#L224](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/compiler/compiler.py#L224)
