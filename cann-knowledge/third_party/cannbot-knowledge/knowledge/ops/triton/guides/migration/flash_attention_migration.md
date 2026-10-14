---
type: Guide
title: Flash Attention GPU→NPU 迁移对照
description: Flash Attention（含前向/反向）从 GPU Triton 迁移到 Ascend NPU 的 12 个关键 Diff 点：Grid
  收缩、mask 预计算、CV 流水线、propagate_nan、编译参数与 bwd 融合。
tags: [task.migration, topic.attention, topic.quality.compatibility.version_migration, topic.compute.normalization, topic.implementation.masking]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---
# Flash Attention GPU→NPU 迁移对照

基于 `flash_attention.py`（GPU）与 `flash_attention_npu_v8.py`（NPU）逐行对比的 12 个 Diff 点。适用：含 `tl.dot` + softmax + mask 的注意力算子迁移（前向+反向）。

## Diff 1：Autotune 配置（移除 num_stages/num_warps）

NPU 无 warp / 软件流水线 stage 概念，`triton.Config` 中移除 `num_stages`/`num_warps`，改用 `multibuffer` 等编译参数；NPU 新增 `get_bwd_qkv_configs()` 供融合反向 QKV kernel autotune。`@maybe_triton_jit` 替换为标准 `@triton.autotune`。

```python
# NPU：else 分支（非 Hopper/Ampere）不带 num_stages/num_warps
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
    for BM in [32, 64, 128] for BN in [32, 64, 128]
]
```

## Diff 2：Grid 收缩 3D → 1D + 核内循环（核心差异）

GPU：`grid = (cdiv(max_seqlen_q, BLOCK_M), q_head, batch_size)`，kernel 内用 `tl.program_id(0/1/2)`。
NPU：Grid 收缩到 `(AICORE_NUM,)`，interleaved 分配实现负载均衡：

```python
pid = tl.program_id(0)
NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)
NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE * q_head
start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM
for block_idx in range(start_block, end_block, step):
    task_hz_idx = block_idx // NUM_BLOCKS_M
    start_m = block_idx % NUM_BLOCKS_M
    start_b = task_hz_idx // q_head
    start_qh = task_hz_idx % q_head
    start_kvh = start_qh * kv_head // q_head  # 避免整除精度问题
    # 原 kernel 逻辑放在 if 守卫内
# 调用处
fwd_kernel[(AICORE_NUM,)](..., AICORE_NUM=NUM_CORES, MAX_Q_LEN=..., MAX_K_LEN=..., BATCH_SIZE=...)
```

核数获取：`driver.active.utils.get_device_properties(torch.npu.current_device())["num_aicore"] / ["num_vectorcore"]`。

## Diff 3：for 循环内 return/continue → if 守卫

Triton 不允许 `for`/`while` 内 `return`/`continue`。核内循环包裹后必须改为正向守卫：

```python
# GPU:  if start_m * BLOCK_M >= q_len: return
# NPU:
if start_m * BLOCK_M < q_len:
    ...  # 原 kernel 逻辑
```

## Diff 4：Block Pointer 差异

- `q_attn_arg_block_ptr` / `k_attn_arg_block_ptr` 被注释，替换为 `mask_block_ptr`（指向预计算 mask_tensor），base 按 batch 偏移 `mask_tensor_ptr + start_b * MAX_Q_LEN * MAX_K_LEN`，shape `(q_len, k_len)`，strides `(MAX_K_LEN, 1)`。
- `tl.multiple_of(start_n, BLOCK_N)` 对齐提示在 NPU 版被注释掉（性能影响不大，可保留可移除）。

## Diff 5：mask 动态计算 → 预计算 mask_tensor

GPU 在 kernel 内调 `mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, TYPE)` 动态算 mask；NPU 在 kernel 外（CPU 上）用 `generate_mask_fn_vectorized` 预计算 `[batch, max_q_len, max_k_len]` bool tensor，kernel 内直接 `load_if(mask_block_ptr, False, False)`。**收益**：减少 kernel 内计算量；**代价**：增加 `batch * max_q_len * max_k_len` bytes 显存。mask_tensor 需传入 `forward()` 并存 ctx 供反向复用。

## Diff 6：Softmax exp2 → exp（仅前向）

```python
# GPU
qk_scale = scale * log2e  # log2e = 1.4426950408889634
alpha = tl.math.exp2((m - m_new) * qk_scale)
p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
# NPU（前向）
qk_scale = scale
s = s * qk_scale + tl.where(mask, 0.0, -2.0 ** 30)   # scale 与 mask 合并为一步
p = tl.math.exp(s - m_new[:, None])
# NPU 反向（bwd_kv / bwd_qkv / bwd_q）仍保留 exp2 + log2e
p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
```

l 值计算也随之变化：GPU `l = m * scale + tl.log(l)`；NPU `m = m + tl.log(l)`（scale 已乘入 s）。

## Diff 7：CV 流水线优化（tl.dot(p,v) 提前）

将 Cube 操作提前、Vector 操作延后，使二者并行：

```python
# GPU
alpha = tl.math.exp2((m - m_new) * qk_scale); p = tl.math.exp2(...); p_sum = tl.sum(p, 1)
acc *= alpha[:, None]; v = load_if(v_block_ptr, ...); acc += tl.dot(p.to(dtype), v)
# NPU
p = tl.math.exp(s - m_new[:, None])
v = load_if(v_block_ptr, False, True)     # v 加载提前
pv = tl.dot(p.to(dtype), v)               # Cube 操作提前
p_sum = tl.sum(p, 1)                      # Vector 延后
alpha = tl.math.exp(m - m_new)            # Vector 延后
acc = acc * alpha[:, None] + pv           # 合并 acc 更新
```

## Diff 8：propagate_nan 参数（NPU 特有，必加）

online softmax 中全 mask 列会产生 `-inf`，默认行为可能吞没 NaN：

```python
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
```

归约用 `propagate_nan=True`，逐元素用 `tl.PropagateNan.ALL`（默认 `NONE`）。GPU 版 Triton 不支持该参数。

## Diff 9：编译参数配置（性能核心）

| 参数 | CV 融合 kernel | 纯 Vector kernel |
|------|---------------|-----------------|
| `multibuffer=True` | ✓ | ✓ |
| `limit_auto_multi_buffer_of_local_buffer="no-limit"` | ✓ | ✓ |
| `enable_mixed_cv=True` | ✓ | 不需要 |
| `enable_auto_bind_sub_block=True` | ✓ | 不需要 |
| `sync_solver=True` | ✓ | 不需要 |
| `enable_flatten=False` | ✓（CV 不能展平） | 不需要 |
| `set_workspace_multibuffer=2` | ✓ | 不需要 |

判断 kernel 类型：含 `tl.dot` → CV 融合（7 参数全配）；纯向量 → 仅 `multibuffer` + `no-limit`。**910_95 上 `multibuffer` 默认 False，必须显式置 True。**

## Diff 10：bwd_preprocess_ifmn 特化

用 3D block pointer `(TASK_SIZE, Q_HEAD_NUM, V_DIM)`（`order=(2,1,0)`）一次加载所有 head，按 seq 段分配任务，归约轴从 axis=1 变 axis=2；使用 `VECTOR_NUM` 而非 `AICORE_NUM`（纯 Vector 不需要 Cube）；`TASK_SIZE = cdiv(max_seqlen_q * batch_size, VECTOR_NUM)`；`BLOCK_SIZE` 按 `V_DIM > 64 ? 16 : 32` 选择。

## Diff 11：bwd_qkv 融合 kernel + atomic_add

NPU 将 `bwd_kv_kernel` 与 `bwd_q_kernel` 融合为 `bwd_qkv_kernel`，一次遍历同时算 dq/dk/dv：

```python
@triton.autotune(list(filter(keep, get_bwd_qkv_configs())),
    key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
    reset_to_zero=["dq_ptr"])      # autotune 每次运行前清零 dq
# kernel 内：
extension.compile_hint(ds, "break_vf")
dq = tl.dot(ds, tl.trans(k)); dq *= scale
tl.atomic_add(dq_ptr + (...), dq, mask=row_mask[:, None])  # 多 K 块对同一 Q 块贡献梯度
dk += tl.dot(tl.trans(ds), q); dv += tl.dot(tl.trans(p), do)
```

要点：`dq` 必须 `torch.zeros(..., dtype=torch.float32)` 初始化（atomic_add 需零初值 + fp32 保精度），算完 `dq.to(q.dtype)`；`use_fused_bwd_qkv` 参数控制是否启用，否则回退分离 kernel。

## Diff 12：extension.compile_hint(ds, "break_vf")

在 `ds` 计算完成、`tl.dot(ds, k)` 之前插入，打断过长的 VF 融合链防 UB 溢出：

```python
import triton.language.extra.cann.extension as extension
ds = tl.where(mask, ds, 0.0)
extension.compile_hint(ds, "break_vf")
dq += tl.dot(ds, k)
```

## 910_95 特别注意

1. `multibuffer` 默认 `False`，必须显式 `=True`。
2. `enable_mixed_cv` 仅在 910_95（Ascend950）平台生效。
3. UB 溢出回退顺序：减小 `BLOCK_M`/`BLOCK_N` → `set_workspace_multibuffer=1` → `limit_auto_multi_buffer_of_local_buffer="no-limit"` → 关键位置插 `break_vf`。
4. NaN 传播行为与 GPU 可能不同，online softmax 必须加 `propagate_nan`。

## 迁移检查清单

Config 移除 `num_stages`/`num_warps` → Grid 3D→1D+核内循环 → `return` 改 `if` 守卫 → mask_fn 改预计算 mask_tensor → 前向 softmax exp2→exp → `tl.dot(p,v)` 提前 → 加 `propagate_nan` → 按 CV/Vector 类型配编译参数 → `bwd_preprocess_ifmn` 替换 → `bwd_qkv` 融合+atomic_add → 插 `break_vf` → 910_95 显式 `multibuffer=True`。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Config.h#L351-L363](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Tools/bishengir-compile/Config.h#L351-L363)
- [core.py#L22](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L22)
- [core.py#L1957](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1957)
- [standard.py#L174](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/standard.py#L174)
- [aux_ops.py#L135-L151](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/aux_ops.py#L135-L151)
- [compiler.py#L846](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L846)
