---
type: Optimization
title: NPU 编译参数速查（含 910_95 差异）
description: Triton-Ascend NPUOptions 编译参数全表（核心/辅助/编译模式/精度）、按算子类型的配置模板、精度与 UB 溢出回退策略及
  910_95 平台差异。
tags: [task.optimization, topic.compilation, topic.programming_model.execution_unit.cube, topic.implementation.pipeline, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/triton/autotune.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/triton/jit.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# NPU 编译参数速查（含 910_95 差异）

> 触发条件：为 kernel 配置编译参数时查阅。参数经 `kernel[grid](...)` 关键字传入，映射到 `NPUOptions` 字段。

## 核心参数（按算子类型选择）

| 参数 | 类型 | 默认值 | 语义 | 适用场景 |
|------|------|--------|------|----------|
| `enable_flatten` | bool | None | IR 展平优化：多维循环展平为一维，提升向量化效率 | 纯 Vector: True；CV 融合: False |
| `multibuffer` | bool | 非 910_95 为 True；**910_95 为 False** | ping-pong 多缓冲流水，计算当前数据时预取下一批 | 通用；910_95 需显式设 True |
| `enable_auto_bind_sub_block` | bool | None | Vector 核心划分子块并行执行 | CV 融合: True；纯 Vector: False |
| `sync_solver` | bool | None | 自动求解 Cube-Vector 同步点位置 | CV 融合: True |
| `set_workspace_multibuffer` | int | None | workspace 多缓冲数量（中间结果多份缓冲） | CV 融合: 2 |
| `limit_auto_multi_buffer_of_local_buffer` | str | None | 限制 UB 多缓冲策略；`"no-limit"` 不限制 | CV 融合: "no-limit" |
| `enable_mixed_cv` | bool | None | 混合 CV 模式，Cube/Vector 同 kernel 交替执行 | CV 融合: True |

## 辅助参数

| 参数 | 语义 | 副作用 |
|------|------|--------|
| `enable_ubuf_saving` | UB 节省优化（A2/A3） | 可能降低并行度 |
| `enable_hivm_auto_cv_balance` | HIVM 自动 CV 负载均衡 | 编译时间增加 |
| `inject_barrier_all` / `inject_block_all` | 所有操作/块间注入屏障 | 性能下降，调试用 |
| `disable_auto_inject_block_sync` | 禁用自动块同步 | 可能数据竞争 |
| `enable_vf_fusion` | VF（Vector Function）融合，默认 False | UB 占用增加 |
| `limit_auto_multi_buffer_only_for_local_buffer` | 多缓冲仅用于 UB | 减少全局缓冲开销 |
| `enable_cce_vf_auto_sync` / `enable_cce_vf_remove_membar` | CCE VF 自动同步 / 移除内存屏障 | 编译时间增加 / 同步问题 |
| `disable_size_align_for_cast` | 禁用 cast 大小对齐 | 可能非对齐访问 |
| `tile_mix_vector_loop` / `tile_mix_cube_loop` | 混合 Vector/Cube 循环 tiling（A2/A3） | 影响 UB / L1 占用 |
| `unit_flag` | 单元标志同步 | 调试用 |

## 编译模式参数

| 参数 | 默认值 | 语义 |
|------|--------|------|
| `compile_mode` | "simd" | 编译路径选择（见下） |
| `num_warps` | 4 | SIMD 模式影响 HFusion 向量化策略；SIMT 模式控制线程数（非 GPU 线程概念） |
| `num_stages` | 1 | NPU 无软件流水线概念，保持 1 |
| `auto_blockify_size` | 1 | AutoBlockify 分块大小，配合 `TRITON_ALL_BLOCKS_PARALLEL` |
| `add_auto_scheduling` | False | 自动调度（DAG 亲和性），SIMT/混合模式 |
| `enable_bishengir_simt_optimization` | 000 | SIMT 优化控制位（位模式），SIMT 模式 |

`compile_mode` 取值：

| 值 | 行为 | 编译路径 |
|----|------|----------|
| `"simd"` | 默认 | Linalg → HFusion → HIVM → Binary |
| `"unstructured_in_simt"` | 非结构化转 SIMT，自动 `force_simt_template=True` | SIMD + SIMT 混合 |
| `"simt_only"` | 纯 SIMT，自动 `force_simt_only=True`、`parallel_mode="simt"` | TTIR → TTGIR → LLVM → Binary |

## 精度参数

| 参数 | 默认值 | 语义 |
|------|--------|------|
| `default_dot_input_precision` | "ieee" | dot 默认输入精度 |
| `allowed_dot_input_precisions` | ("ieee", "hf32") | 允许的 dot 输入精度 |
| `enable_fp_fusion` | True | 浮点融合（FMA） |
| `disable_fma` | False | 禁用 FMA：提高精度、降低性能 |

## 按算子类型的配置模板

**纯 Vector**（add、LayerNorm、Softmax）：`enable_flatten=True, multibuffer=True`，无需 CV 参数。

**纯 Cube**（MatMul 前向，epilogue 在 Vector）：
`enable_auto_bind_sub_block=True, enable_flatten=True, set_workspace_multibuffer=2, sync_solver=True, limit_auto_multi_buffer_of_local_buffer="no-limit", multibuffer=True`

**CV 融合**（Flash Attention、MatMul+后处理）：
`enable_auto_bind_sub_block=True, enable_flatten=False, set_workspace_multibuffer=2, sync_solver=True, limit_auto_multi_buffer_of_local_buffer="no-limit", multibuffer=True, enable_mixed_cv=True`

```python
fwd_kernel[grid](q, k, v, o, ..., BLOCK_M=128, BLOCK_N=32,
                 enable_auto_bind_sub_block=True, enable_flatten=False,
                 set_workspace_multibuffer=2, sync_solver=True,
                 limit_auto_multi_buffer_of_local_buffer="no-limit",
                 multibuffer=True, enable_mixed_cv=True)
```

决策树：

```
kernel 含 tl.dot()？
├─ 否 → 纯 Vector：enable_flatten=True, multibuffer=True
└─ 是 → Cube 后有 Vector 处理（softmax/mask/激活）？
    ├─ 否 → 纯 Cube：sub_block=True, flatten=True, workspace_mb=2, sync_solver, no-limit, multibuffer
    └─ 是 → CV 融合：上述 + flatten=False + enable_mixed_cv=True
```

参数也可经 `triton.Config({...}, multibuffer=True, ...)` 在 autotune 中传入，或 `triton.compile(kernel, options={...})`。

## 回退策略

编译失败逐步回退（高性能 → 最大兼容）：

```
1. 完整 CV 参数 → 2. 去 enable_mixed_cv → 3. 去 enable_auto_bind_sub_block
→ 4. 去 sync_solver → 5. 最小集 multibuffer=True
```

精度问题：`disable_fma=True` → `enable_fp_fusion=False` → `default_dot_input_precision="ieee"`。

UB 溢出：减 BLOCK_M/N → `set_workspace_multibuffer=2→1` → 限制 `limit_auto_multi_buffer_of_local_buffer` → `enable_flatten=False` → `enable_ubuf_saving=True`（A2/A3）。

## 910_95（dav-c310）差异

| 特性 | A2/A3 (910B) | 910_95 |
|------|-------------|--------|
| `multibuffer` 默认 | True | **False（需显式开启）** |
| UB | 192 KB | **256 KB** |
| RF | 无 | 128 KB |
| FFTS | 支持 | 不支持（自动禁用） |
| fixpipe L0C→UB | 不支持 | 支持 |
| copy (UB→UB/L1) | 不支持 | 支持 |
| Vector 核数 | = Cube 核数 | = Cube × 2 |
| `shared_mem_dynamic_size`（SIMT） | 221184 | 122880 |

平台自动检测（`NPUOptions.compile_on_910_95`）：PCI ID 含 `0xd806`，或 `npu-smi info` 输出含 `ascend910_95`/`ascend950`/`910_958b`。

## GPU 参数对照

| 维度 | GPU | NPU |
|------|-----|-----|
| 流水线 | `num_stages=2~4` | `multibuffer=True` |
| 并行度 | `num_warps=4/8/16` 线程 | `num_warps=4` 影响向量化策略 |
| 线程级并行 | 天然 SIMT | `compile_mode="simt_only"` + num_warps=8/16 |
| 双核协调 | 无 | `enable_mixed_cv` + `sync_solver` + `enable_auto_bind_sub_block` |
| 精度 | tf32/ieee/tf32x3 | ieee/hf32 |


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [get_ascend_devices.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/tools/get_ascend_devices.py)
- [compiler.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py)
- [utils.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/utils.py)
