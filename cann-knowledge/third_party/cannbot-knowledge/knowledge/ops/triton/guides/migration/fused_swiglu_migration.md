---
type: Guide
title: Fused SwiGLU GPU→NPU 迁移对照
description: Fused SwiGLU（fwd/bwd_b/bwd_x/bwd_w 四 kernel）GPU→NPU 迁移 Diff：libdevice
  替换、autotune 搜索空间重构、tl.sum 轴变换、SPLIT_K 原子操作移除与编译参数配置。
tags: [task.migration, topic.compute.composite, topic.quality.compatibility.version_migration, topic.compute.elementwise.activation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/sigmoid.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/exp.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# Fused SwiGLU GPU→NPU 迁移对照

基于 `fused_swiglu.py`（GPU）与 `fused_swiglu_npu.py`（NPU）对比。适用：SiLU 门控线性单元融合算子（fwd / bwd_b / bwd_x / bwd_w 四子 kernel）迁移。

## Diff 1：导入与设备

```python
# GPU
from maybe_triton_jit import maybe_triton_jit
from triton.language.extra import libdevice
from .utils import is_hopper, is_ampere
# NPU
import torch_npu
# from maybe_triton_jit import maybe_triton_jit   # 注释，改用标准 @triton.autotune
# from triton.language.extra import libdevice     # 注释，改用 tl.fdiv / tl.exp
from utils import is_hopper, is_ampere
```

## Diff 2：fast_sigmoid / fast_silu 数学函数替换

```python
# GPU
def fast_sigmoid(x):
    return libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-x))
def fast_silu(x):
    dtype = x.type.element_ty
    x = x.to(tl.float32)
    return libdevice.fast_dividef(x, 1.0 + libdevice.fast_expf(-x)).to(dtype)
# NPU
def fast_sigmoid(x):
    return tl.fdiv(1.0, 1.0 + tl.exp(-x))
def fast_silu(x):
    dtype = x.type.element_ty
    x = x.to(tl.float32)
    return tl.fdiv(x, 1.0 + tl.exp(-x)).to(dtype)
```

映射规则：`libdevice.fast_dividef(a, b)` → `tl.fdiv(a, b)`；`libdevice.fast_expf(x)` → `tl.exp(x)`。Triton 内建函数无需额外导入（NPU 侧 CANN libdevice 路径为 `triton.language.extra.cann.libdevice`）。

## Diff 3：Autotune 装饰器与配置（核心差异）

- 装饰器：`@maybe_triton_jit` → `@triton.autotune`（顺序保持在 `@triton.jit` 上方，`key` 不变）。
- GPU `else` 分支返回单个固定 Config；NPU 改为列表推导式搜索空间：

| Kernel | 搜索维度 | 候选值 | 组合数 |
|--------|---------|--------|--------|
| fwd | BM×BN×BK | [128,256,512]×[64,128,256,512]×[32,64,128,512] | 48 |
| bwd_b | BM×BN | range(16,129,16)×range(16,65,16) | 56 |
| bwd_x | BM×BN×BK | 同 fwd | 48 |
| bwd_w | BM×BN×BK | 同 fwd | 48 |

**避坑**：源 NPU 版仍保留 `num_stages`/`num_warps`（无效遗留，应移除）；48 个配置首次 autotune 耗时长，建议控制在 20 个以内。

## Diff 4：前向 kernel 指针算术

GPU：循环外预计算 offset（含 `% M` 取余）+ 循环内指针递增 `x_ptrs += BLOCK_SIZE_K`。
NPU：移除取余，循环内每次重算 offset 与完整 mask：

```python
for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    offset_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
    x_mask = (offset_xm[:, None] < M) & (offset_k[None, :] < K)
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    g_fc_offs = (offset_k[:, None] * N + offset_wn[None, :])
    g_fc_mask = (offset_k[:, None] < K) & (offset_wn[None, :] < N)
    w_g = tl.load(w_g_ptr + g_fc_offs, mask=g_fc_mask, other=0.0)
```

原因：指针递增在 NPU 编译后可能产生不可预期行为；w_g/w_fc 共享同一组 `g_fc_offs`/`g_fc_mask`。

## Diff 5：bwd_b 的 tl.sum 轴变换

```python
# GPU：accumulator (BLOCK_SIZE_N, BLOCK_SIZE_M)，沿 axis=1 归约
tl.store(db_g_ptr + col_off, tl.sum(sum_b_g, 1), mask=col_off < N)
# NPU：accumulator (BLOCK_SIZE_M, BLOCK_SIZE_N)，沿 axis=0 归约
tl.store(db_g_ptr + col_off, tl.sum(sum_b_g, 0), mask=col_off < N)
```

配套指针布局同步调整：GPU `dy_ptr + (row_off[None,:] * N + col_off[:,None])` → NPU `dy_ptr + (row_off[:,None] * N + col_off[None,:])`。目的：适配 NPU 内存访问连续性。

## Diff 6：bwd_w 循环方向与 SPLIT_K 移除（含负面结论）

```python
# GPU：反向遍历 + SPLIT_K 原子写
for k in range(K, 0, -BLOCK_SIZE_K * SPLIT_K): ...
if SPLIT_K == 1: tl.store(...)
else: atomic_store(dw_g_ptrs, dw_g, dw_mask, LOCK_G, SPLIT_K)
# NPU：正向遍历 + 直接 store（SPLIT_K 分支被注释掉）
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    ...
tl.store(dw_g_ptrs, dw_g, mask=dw_mask)
```

**注意（避坑）**：当前 NPU 版 SPLIT_K 固定为 1，**不支持多 SPLIT_K 并行归约**；如需 SPLIT_K > 1 须重新实现原子操作（NPU 的 `tl.atomic_cas`/`tl.atomic_xchg` 行为可能与 GPU 不同，需实测）。

## Diff 7：编译参数（NPU 特有）

| Kernel | 类型 | enable_auto_bind_sub_block | enable_flatten | multibuffer | 其他 |
|--------|------|---------------------------|----------------|-------------|------|
| fwd | CV 融合 | True | False | True | set_workspace_multibuffer=2, sync_solver=True, limit_auto_multi_buffer_of_local_buffer="no-limit" |
| bwd_b | 纯 Vector | False | True | True | - |
| bwd_x | CV 融合 | True | False | True | 同 fwd |
| bwd_w | CV 融合 | True | False | True | 同 fwd |

选择逻辑：含 `tl.dot` → CV 融合需 CV 协同参数且 `enable_flatten=False`（Cube/Vector 需各自独立循环结构）；纯 Vector 仅 `enable_flatten=True` + `multibuffer=True`。

## Diff 8：测试代码

`device="cuda"` → `torch.device("npu")`；加 `torch.manual_seed(1024)` 保证可复现；用 `torch.load` 读 GPU dump 数据做三方对比（NPU vs GPU vs CPU FP32）；加 `torch_npu.profiler`；`torch.npu.synchronize()` 替代 cuda 版；打印 `kernel.best_config` 查看 autotune 结果。

## 910_95 特别注意

| 特性 | 说明 |
|------|------|
| multibuffer 默认值 | 910_95 默认 `False`（910B 默认 True），必须显式 `=True` |
| UB 容量 | 910_95 为 256KB，允许更大 BLOCK_SIZE；910B 应限制搜索空间 |
| 搜索空间 | 48 配置首次 autotune 耗时长，控制在 20 个以内 |
| SPLIT_K | 仅支持 SPLIT_K=1 |