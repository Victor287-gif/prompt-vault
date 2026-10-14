---
type: API
title: Triton-Ascend 编译选项与环境变量（NPUOptions 全集）
description: NPUOptions 全字段表（基础/线程/编译模式/Ascend 特定/精度/内存/CV 流水线/向量化/SIMT）、compile_mode
  三值语义、SIMT 优化位掩码解析、环境变量参考与常用配置组合。
tags: [task.lookup, topic.compilation, topic.tiling, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---

# Triton-Ascend 编译选项与环境变量

Python 侧 `NPUOptions` 字段经 `triton.Config(...)` / autotune / `triton.compile(options={...})` 传入，最终映射为 bishengir-compile 命令行参数（映射结果见 [linalg_to_binary.md](linalg_to_binary.md)；BiShengIR 内部 pipeline 选项见 [pipeline_options.md](pipeline_options.md)）。

## NPUOptions 字段全集

基础：`debug`(False，调试输出)、`sanitize_overflow`(True，溢出检查)、`llvm_version`(15)、`kernel_name`("triton_"，名称前缀)、`arch`(""，自动填充)。

线程与并行：

| 选项 | 默认 | 语义 |
|------|------|------|
| `cluster_dims` | (1,1,1) | 集群维度 |
| `num_warps` | 4 | warp 数；SIMD 下影响 HFusion 向量化策略，SIMT 下控制线程数（4/8/16） |
| `num_ctas` | 1 | CTA 数 |
| `num_stages` | 1 | 流水线阶段数 |
| `warp_size` | 32 | 每 warp 线程数 |
| `num_buffers_warp_spec` / `num_consumer_groups` | 0 | warp 特化缓冲区数 / 消费者组数 |
| `reg_dec_producer` / `reg_inc_consumer` | 0 | 生产者/消费者寄存器调整 |

编译模式：

| 选项 | 默认 | 语义 |
|------|------|------|
| `compile_mode` | "simd" | `"simd"`：parallel_mode=simd，走 Linalg 路径；`"unstructured_in_simt"`：置 force_simt_template=True，混合路径；`"simt_only"`：置 force_simt_only=True 且 parallel_mode=simt，TTIR 直编 |
| `parallel_mode` | "simd" | `simd` / `simt` |
| `force_simt_only` | False | 强制纯 SIMT |
| `force_simt_template` | False | 强制 SIMT 模板 |
| `mix_mode` | "" | 通常自动确定 |

`__post_init__` 副作用：`force_simt_only` 时 `shared_mem_dynamic_size` 默认 122880，否则 221184。

Ascend 特定：

| 选项 | 默认 | 语义 |
|------|------|------|
| `auto_blockify_size` | 1 | AutoBlockify 分块大小（需配合 `TRITON_ALL_BLOCKS_PARALLEL`） |
| `compile_on_910_95` | 自动检测 | 910_95 平台编译 |
| `optimize_dynamic_offset` | False | 优化动态偏移 |
| `enable_mask_fallback_conversion` | False | mask 回退转换 |
| `enable_warp_specialization` | False | warp 特化 |
| `enable_nd2nz_on_vector` | False | Vector 核 ND→NZ 布局转换 |
| `enable_persistent` | False | 持久化模式 |
| `optimize_epilogue` | False | Epilogue 优化 |
| `enable_fp_fusion` | True | 浮点融合 |
| `allow_fp8e4nv` | False | 允许 FP8 E4NV |
| `auto_tile_and_bind_subblock` | True | 自动分块绑定子块 |
| `vf_merge_level` | 1 | VF 合并级别 |
| `enable_select_analysis` | True | select 分析 |
| `add_auto_scheduling` | False | 自动调度（DAG 亲和性优化，见 [ascend_passes.md](ascend_passes.md)） |

精度：`default_dot_input_precision`("ieee")、`allowed_dot_input_precisions`(("ieee","hf32"))、`max_num_imprecise_acc_default`(0)、`supported_fp8_dtypes`(("fp8e5","fp8e4b15",...))。

内存与缓冲：

| 选项 | 默认 | 语义 |
|------|------|------|
| `multibuffer` | 非 910_95 时 True | ping-pong 流水线（存算并行），可用 UB 减半；910_95 默认关 |
| `enable_ubuf_saving` | None | UB 节省（A2/A3） |
| `enable_auto_bind_sub_block` | None | 自动绑定子块（CV 融合） |
| `disable_tightly_coupled_buffer_reuse` | False | 禁用紧耦合缓冲区复用 |
| `shared_mem_dynamic_size` | 自动 | 共享内存动态大小 |
| `stream` | None | NPU stream |

CV 流水线：`enable_hivm_auto_cv_balance`(None，CV 自动均衡)、`sync_solver`(None，同步求解器)、`unit_flag`(None，单元标志同步)、`inject_barrier_all`(None，全屏障注入)、`inject_block_all`(None，全块同步注入)、`tile_mix_vector_loop`/`tile_mix_cube_loop`(None，混合循环 tiling，A2/A3)、`disable_auto_inject_block_sync`(None)、`enable_mixed_cv`(None)、`enable_vf_fusion`(False)。

向量化：`enable_drop_unit_dims`(None)、`enable_flatten`(None)、`enable_auto_vectorize_v2`(None)、`auto_vectorize_v2_max_fused_ops_num`(None)、`prevec_max_fused_ops_num`(None)、`hfusion_enable_multiple_consumer_fusion`(False)。

SIMT：`enable_bishengir_simt_optimization`(位掩码，见下)、`simt_stack_limit`(None)、`enable_simt_reorder_instruction`(False)、`disable_fma`(False，禁 FMA 提高精度)。

其他：`extern_libs`(None)、`bisheng_options`(默认含 libdevice 链接)、`limit_auto_multi_buffer_only_for_local_buffer`(None)、`limit_auto_multi_buffer_of_local_buffer`(None)、`set_workspace_multibuffer`(None)、`enable_cce_vf_auto_sync`(None)、`enable_cce_vf_remove_membar`(None)、`disable_size_align_for_cast`(None)。

## SIMT 优化位掩码（enable_bishengir_simt_optimization）

整数十进制按位解析（`getPassColumnDigit`），默认 `900101`：

| 位 | Pass | 默认位值 |
|----|------|----------|
| 个位 | `decompose-reduction` | 1（启用） |
| 十位 | `optimize-layouts` | 0 |
| 百位 | `convert-triton-gpu-to-llvm` | 1 |
| 千位 | `reduce-op` | 0 |
| 万位 | `optimize-loads` | 0 |
| 十万位 | `loop-restructure-arange-optimization` | 9（贪心平衡分组） |

十万位特殊值：0=不执行，1=单组，2~8=指定分组数，9=贪心自动分组。

## 环境变量

| 类别 | 变量（默认 → 语义） |
|------|---------------------|
| 调试 | `TRITON_DEBUG`(0→1 调试输出)、`MLIR_ENABLE_DUMP`(0/1/kernelName dump MLIR)、`LLVM_IR_ENABLE_DUMP`、`TRITON_KERNEL_DUMP`(1=dump 各阶段 IR)、`TRITON_DUMP_DIR`、`MLIR_ENABLE_REMARK`、`USE_IR_LOC`(IR 带位置)、`TRITON_FRONT_END_DEBUGGING`(保留完整 traceback)、`TRITON_REPRODUCER_PATH`(MLIR 复现文件)、`TRITON_INTERPRET`(解释器模式)、`TRITON_DEVICE_PRINT`、`TRITON_MEMORY_DISPLAY`(内存 JSON)、`TRITON_PRINT_AUTOTUNING`、`TRITON_ENABLE_LLVM_DEBUG` / `TRITON_LLVM_DEBUG_ONLY` |
| 编译控制 | `TRITON_ALWAYS_COMPILE`(1=绕缓存)、`DISABLE_LLVM_OPT`(1 或 "disable-lsr")、`MLIR_ENABLE_TIMING` / `LLVM_ENABLE_TIMING`、`TRITON_DEFAULT_FP_FUSION`(默认 1)、`TRITON_KERNEL_OVERRIDE` / `TRITON_OVERRIDE_DIR`、`TRITON_ASCEND_COMPILE_SPEED_OPT`(失败后跳过后续阶段)、`TRITON_COMPILE_ONLY`、`TRITON_DISABLE_FFTS`、`TRITON_DISABLE_PRECOMPILE`、`TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` |
| 运行调度 | `TRITON_ALL_BLOCKS_PARALLEL`(1=AutoBlockify，允许 grid>65535)、`TRITON_ENABLE_TASKQUEUE`、`TRITON_ENABLE_SANITIZER`、`ENABLE_PRINT_UB_BITS`、`TRITON_ENABLE_LIBDEVICE` |
| 其他 | `TRITON_BENCH_METHOD`("npu"=do_bench_npu)、`TRITON_REMOTE_RUN_CONFIG_PATH` |

环境变量优先级高于代码配置（如 `TRITON_ALL_BLOCKS_PARALLEL` 覆盖 `auto_blockify_size`）。

## 常用配置组合

```python
# SIMT 纯线程
triton.Config({"BLOCK_SIZE": 1024}, compile_mode="simt_only", num_warps=8,
              enable_bishengir_simt_optimization=900101)
# 混合路径
triton.Config({"BLOCK_SIZE": 1024}, compile_mode="unstructured_in_simt", add_auto_scheduling=True)
# CV 流水线（matmul）
triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2,
              multibuffer=True, enable_hivm_auto_cv_balance=True, sync_solver=True)
# AutoBlockify（需 os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"）
triton.Config({"BLOCK_SIZE": 1024}, auto_blockify_size=4)
# 调试
os.environ.update(TRITON_KERNEL_DUMP="1", MLIR_ENABLE_DUMP="1", TRITON_ALWAYS_COMPILE="1")
```

`compile_mode` 是最关键选项（决定整条路径）；SIMT 位掩码需谨慎（错误组合可能编译失败）；部分选项仅特定平台生效；推荐用 autotune 搜索组合。

