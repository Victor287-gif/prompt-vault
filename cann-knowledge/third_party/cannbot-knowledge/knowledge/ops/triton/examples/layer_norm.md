---
type: Example
title: LayerNorm 示例（前向融合 + 反向自旋锁分组归约）
description: 教学示例：LayerNorm 前向单 kernel 融合（分块累加 mean/var + rstd 预计算），反向用 atomic_cas
  自旋锁分组累加 dw/db。
tags: [task.implementation, topic.compute.normalization, topic.compute.reduction, topic.compute.composite, topic.implementation.atomic, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/05-layer-norm.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# LayerNorm 示例（前向融合 + 反向自旋锁分组归约）

## 前向 kernel（完整）

三次遍历行数据（mean → var → normalize），全部片上完成；`rstd = 1/sqrt(var+eps)` 预计算倒数避免重复除法。

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
```

要点：方差直接算 `(x - mean)^2` 的均值（不用 `E[x^2]-E[x]^2`，后者数值上可能得负方差）；分块累加器 `_mean/_var` 用 fp32；`tl.where(cols < N, x - mean, 0.)` 防止 padding 污染方差。

## 反向 kernel（完整）

dx 每行独立；dw/db 是全行归约，用 `GROUP_SIZE_M` 组缓冲区 + `atomic_cas` 自旋锁减少竞争，第二个 kernel 再汇总各组。

```python
@triton.jit
def _layer_norm_bwd_dx_fused(
    DX, DY, DW, DB, X, W, Mean, Rstd, Lock,
    stride, N,
    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd

    tl.store(DX + cols, dx, mask=mask)

    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)

@triton.jit
def _layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB,
    M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)
```

Host 侧关键参数（封装于 `torch.autograd.Function`）：

```python
MAX_FUSED_SIZE = 65536 // x.element_size()
BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
# backward：GROUP_SIZE_M 按 N 取 256(<=1024) / 128(<=4096) / 96(<=8192) / 64(>8192)
# locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32)  # 锁 + 计数器
```

## NPU 注意

- **atomic 限制**：NPU 上 `atomic_add` 不支持多核 add+保存中间结果，dw/db 累加需改为本例的普通 add + 自旋锁方式。`atomic_cas` 自旋锁可用，但获取/释放必须在同一执行路径，条件分支导致锁未释放会死锁。
- **BLOCK_SIZE 上限**：`MAX_FUSED_SIZE = 65536 // element_size`，N 更大需列分块。
- 精度：kernel 内部全程 fp32，写回再转原 dtype；核对 eps 与 PyTorch 一致。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [05-layer-norm.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/tutorials/05-layer-norm.py)
