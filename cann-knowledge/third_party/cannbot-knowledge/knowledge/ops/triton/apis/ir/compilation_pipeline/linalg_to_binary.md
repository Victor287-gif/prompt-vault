---
type: API
title: Linalg IR → 设备二进制（bishengir-compile 与 CPU 工具链）
description: 编译末段：A2/A3 与 910_95 两条 bishengir-compile 路径的命令行与差异、SIMT 直编、CPU 后端 mlir-opt
  降链、kernel.o/libkernel.so 产物与元数据回调、UB 占用提取，及 BiSheng 编译选项分类速查。
tags: [task.lookup, topic.compilation, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---

# Linalg IR → 设备二进制

流水线末段。NPU 后端由 BiSheng Compiler（`bishengir-compile`）完成 `Linalg → HFusion → HIVM → LLVM → kernel.o`（内部阶段见 [overview.md](overview.md)）；CPU 后端走 mlir-opt + llvm-link + llc 标准工具链。

## NPU 编译命令

A2/A3（910B，`linalg_to_bin_enable_npu_compile_A2_A3`）：

```bash
bishengir-compile kernel.ttadapter.mlir \
  --target=Ascend910B3 \
  --enable-hfusion-compile=true --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True \
  -o kernel
```

910_95（`linalg_to_bin_enable_npu_compile_910_95`，不支持 FFTS）：

```bash
bishengir-compile kernel.ttadapter.mlir \
  --target=Ascend910_9589 \
  --enable-hfusion-compile=true --enable-triton-kernel-compile=true \
  -o kernel
```

SIMT Only（`ttir_to_npubin`，跳过 Linalg）：

```bash
bishengir-compile kernel.ttir.mlir \
  --target=Ascend910B3 \
  --enable-hivm-compile=false --enable-triton-ir-compile --pure-simt \
  --num-warps=4 --threads-per-warp=32 \
  -o kernel
```

两 NPU 函数共同流程：`_parse_linalg_metadata()` 解析元数据 → 写临时 `kernel.ttadapter.mlir` → 构建选项 → subprocess 调 bishengir-compile → 从 stdout 提取 UB 占用 → 从 `libkernel.so` 加载元数据回调。A2/A3 额外支持 `--enable-ubuf-saving`、`--tile-mix-vector-loop`、`--tile-mix-cube-loop`、`--reg-based=true`、`--enable-memory-display`。

## 产物与元数据

| 文件 | 说明 |
|------|------|
| `kernel.o` / `kernel_reloc.o` | NPU ELF 目标文件；名字由 `_check_bishengir_api_change()` 按编译器 API 版本决定（新 API → `kernel.o`） |
| `libkernel.so` | 共享库，含元数据回调（可选） |

回调函数（ctypes 从 libkernel.so 加载）：

| 回调 | 元数据字段 |
|------|-----------|
| `_infer_task_type_function` | `bs_task_type` |
| `_infer_workspace_shape_function` | `workspace_size` |
| `_infer_sync_block_lock_num_function` | `lock_num` |
| `_infer_sync_block_lock_init_function` | `lock_init_val` |

UB 占用从编译 stdout 正则 `UB\s+size\s*=\s*(\d+)\s*bits` 提取到 `metadata["required_ub_bits"]`，供 Inductor autotune；`ENABLE_PRINT_UB_BITS=1` 可打印。

运行时加载：`driver.active.utils.load_binary()` 加载 `.o` → 按 `workspace_size` 分配 GM workspace → 按 `lock_num`/`lock_init_val` 配同步 → launcher 启动。

链接：编译时经 `bisheng_options` 传 `-cce-link-aicore-ll-module`（libdevice，受 `TRITON_ENABLE_LIBDEVICE` 控制）；用户 bitcode 经 `--link-aicore-bitcode=<path>`（来自 Linalg IR 的 `bitcode` 属性）。

## CPU 后端降链

`linalg_to_llir()`：mlir-opt 顺序执行 `convert-linalg-to-affine-loops → eliminate-empty-tensors → empty-tensor-to-alloc-tensor → one-shot-bufferize(allow-return-allocs-from-loops=true) → lower-affine → convert-linalg-to-loops → convert-scf-to-cf → convert-cf-to-llvm → convert-{arith,math,complex,vector,index}-to-llvm → memref-expand → expand-strided-metadata → finalize-memref-to-llvm → convert-func-to-llvm → lower-affine → convert-arith-to-llvm → reconcile-unrealized-casts`，再 `mlir-translate --mlir-to-llvmir`。

`llir_to_cpuasm()`：`downgrade_llir()` → `llvm-link` 链 `libspirv-aarch64--.bc`（`--only-needed`）→ `llc` 生成汇编；kernel 名后缀 `" cpu"`。

## BiSheng Compiler 选项分类速查

通用：`--target=<arch>`、`--enable-hfusion-compile=true`、`--enable-triton-kernel-compile=true`、`-o`。

| 类别 | 选项（平台） |
|------|--------------|
| 内存优化 | `--enable-auto-multi-buffer`（通用）、`--enable-ubuf-saving`（A2/A3）、`--limit-auto-multi-buffer-only-for-local-buffer`、`--limit-auto-multi-buffer-of-local-buffer`、`--set-workspace-multibuffer`、`--disable-tightly-coupled-buffer-reuse`（910_95） |
| CV 流水线 | `--enable-auto-bind-sub-block`、`--enable-hivm-auto-cv-balance`、`--enable-hivm-graph-sync-solver`、`--enable-hivm-unit-flag-sync`、`--enable-hivm-inject-barrier-all-sync`、`--enable-hivm-inject-block-all-sync`、`--disable-auto-inject-block-sync`（均通用）；`--tile-mix-vector-loop` / `--tile-mix-cube-loop`（A2/A3） |
| 向量化 | `--enable-drop-unit-dims`、`--enable-flatten`、`--enable-auto-vectorize-v2`（通用）；`--enable-vf-fusion`、`--hfusion-max-fused-ops-in-auto-vectorize-v2`、`--hfusion-max-fused-elementwise-ops`、`--enable-vf-merge-level`、`--hfusion-enable-multiple-consumer-fusion`（910_95） |
| SIMT | `--enable-hivm-compile=false`、`--enable-triton-ir-compile`、`--pure-simt`、`--num-warps`、`--threads-per-warp`、`--enable-bishengir-simt-optimization`、`--simt-stack-limit`、`--shared-mem-dynamic-size`、`--enable-simt-reorder-instruction=true`、`--disable-fma` |
| 调试 | `--bishengir-print-ir-after=hivm-inject-sync`、`--enable-sanitizer=true`、`--enable-debug-info=true`、`--enable-print-memory-allocated-size`、`--enable-memory-display=true`、`--disable-ffts` |

Python 侧 NPUOptions 到这些选项的映射见 [triton_compile_options.md](triton_compile_options.md)；BiShengIR 内部 pipeline 选项见 [pipeline_options.md](pipeline_options.md)。

## 调试

编译失败时 NPU IR 保存为 `kernel.npuir.mlir`；`opt.debug=True` 打印完整命令行（输出中搜 `[DEBUG] cmd_list:`）；配合 `TRITON_KERNEL_DUMP=1`。提取细节见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [compiler.py#L403](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L403)
- [compiler.py#L607](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L607)
- [compiler.py#L171](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L171)
- [compiler.py#L229](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L229)
