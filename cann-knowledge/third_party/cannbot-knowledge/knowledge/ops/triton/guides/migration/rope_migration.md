---
type: Guide
title: RoPE GPU→NPU 迁移对照
description: RoPE 旋转位置编码 GPU→NPU 迁移 Diff：1D Grid 核内循环、Block Pointer 2D→3D、head 转 constexpr、libdevice
  路径变更与 UB 压力评估。
tags: [task.migration, topic.quality.compatibility.version_migration, topic.compute.elementwise.arithmetic, topic.compute.composite, topic.programming_model.execution_unit.vector, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/sin.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---
# RoPE GPU→NPU 迁移对照

基于 `rope.py`（GPU）与 `rope_npu.py`（NPU）对比。适用：`tl.make_block_ptr` + `libdevice.pow` 的多维 Block Pointer RoPE kernel 迁移。

## Diff 1：导入与 libdevice 路径

```python
# GPU
from triton.language.extra import libdevice
from .utils import is_hopper, is_ampere
# NPU
import torch_npu
from triton.language.extra.cann import libdevice   # CANN 后端路径
# is_hopper/is_ampere 注释掉（不再需要 GPU 架构分支）
```

## Diff 2：Autotune 配置

GPU 按架构返回 1 个 Config（`BLOCK_M: 32, num_stages=2, num_warps=4`）；NPU 改搜索空间、移除 `num_stages`/`num_warps`、引入 `multibuffer`：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "multibuffer": MB})
        for BM in [100, 75, 50, 25, 10, 5, 2]
        for MB in [True, False]
    ],
    key=["DIM", "REVERSE"]
)
```

**避坑**：`BM` 候选含 5/25/75/100 等非 2 的幂次值——910_95 要求 32B 对齐（FP16 元素数须为 16 的倍数），BLOCK_M=5 会性能下降甚至编译错误。建议 `[2, 4, 8, 16, 32, 64, 128]`。

## Diff 3：Kernel 签名

- `head` 从普通参数变为 `head: tl.constexpr`——Block Pointer 的 `block_shape` 依赖 head，必须编译期确定（**副作用**：每个不同 head 值触发一次重编译；head 固定的模型可接受）。
- 新增 `max_seq_len` 普通参数，用于核内 `tasks = tl.cdiv(max_seq_len, BLOCK_M)`。

## Diff 4：Grid 3D → 1D + 核内循环（核心差异）

```python
# GPU
grid = lambda META: (triton.cdiv(max_len, META["BLOCK_M"]), bs, head)
# NPU
grid = (bs,)
# kernel 内：
start_b = tl.program_id(0)
begin = tl.load(cu_seqlens + start_b)
len = tl.load(cu_seqlens + start_b + 1) - begin
tasks = tl.cdiv(max_seq_len, BLOCK_M)
for start_m in range(tasks):
    if start_m * BLOCK_M < len:   # if 守卫替代 return
        ...
```

每个 program 处理一个 batch 的全部 seq block 和全部 head，避免 coreDim 超限。**代价**：batch 小（1~4）时大量 AI Core 空闲，可扩展 Grid 为 `(bs * head,)` 提升并行度（需相应调整 program_id 计算）。

## Diff 5：Block Pointer 2D → 3D（核心差异）

```python
# GPU：3D Grid 的 start_h 维度在 base 中偏移
x0_block_ptr = tl.make_block_ptr(
    base = in_ptr + begin * head * DIM + start_h * DIM,
    shape = (len, DIM), strides = (head * DIM, 1),
    offsets = (start_m * BLOCK_M, 0),
    block_shape = (BLOCK_M, DIM // 2), order = (1, 0))
# NPU：head 维度纳入 block_ptr，一个 program 加载所有 head
x0_block_ptr = tl.make_block_ptr(
    base = in_ptr + begin * head * DIM,
    shape = (len, head, DIM), strides = (head * DIM, DIM, 1),
    offsets = (start_m * BLOCK_M, 0, 0),
    block_shape = (BLOCK_M, head, DIM // 2), order = (2, 1, 0))
```

`order=(2, 1, 0)` 表示最内维（DIM）连续，符合行优先布局。

## Diff 6：旋转计算广播 2D → 3D

```python
# GPU：x0 (BLOCK_M, DIM//2)，cos 同形，2D 广播自动对齐
x1 = x0 * cos - y0 * sin
# NPU：x0 (BLOCK_M, head, DIM//2)，cos (BLOCK_M, DIM//2) 需显式扩展
x1 = x0 * cos[:, None, :] - y0 * sin[:, None, :]
y1 = x0 * sin[:, None, :] + y0 * cos[:, None, :]
```

sin/cos 不依赖 head 维度，经 `[:, None, :]` 广播。`inv_freq = libdevice.pow(base, -2.0 / DIM * offset_n)` 逻辑不变，仅 libdevice 路径变更。

## Diff 7：测试代码

输入 `.to(DEVICE)`；加 `torch_npu.profiler` 与 `torch.npu.synchronize()`；容差放宽为 `rtol=1e-3, atol=1e-3`（BF16 合理）。

## 910_95 特别注意

1. **3D Block Pointer 的 UB 压力**：`block_shape=(BLOCK_M, head, DIM//2)` 一次加载全部 head。FP16 估算 `BLOCK_M=100, head=32, DIM//2=32` → 约 200KB，接近 910_95 UB（256KB）上限。head 更大时须减小 BLOCK_M 或将 head 维也做循环切分。
2. **multibuffer 与 UB**：910_95 默认 `multibuffer=False`；autotune 选中 `True` 时 UB 可用减半（128KB）。大 head 场景优先 `multibuffer=False` 或减小 BLOCK_M。
3. **libdevice.pow 精度**：pow 在 Vector 单元执行（Cube 不支持），与 GPU 精度可能有微小差异。用 `rtol=1e-3` 验证；不满足则预计算 inv_freq 表传入 kernel。