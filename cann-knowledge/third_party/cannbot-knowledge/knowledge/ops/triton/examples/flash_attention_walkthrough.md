---
type: Example
title: Flash Attention 教学 Walkthrough（block_ptr + online softmax）
description: 逐步教学：Flash Attention 前向 kernel 的 STAGE 分块、online softmax（exp2/log2 域）、make_block_ptr
  用法及 NPU 注意事项。
tags: [task.implementation, topic.attention, topic.compute.normalization, topic.data_movement.memory_access, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---
# Flash Attention 教学 Walkthrough

> 定位：upstream tutorial 风格的逐步讲解（block_ptr 版，含 STAGE 机制教学）。经验沉淀的生产骨架（GQA/MQA、causal 变体）见 [flash_attention_causal.md](./flash_attention_causal.md)。

## 内循环 kernel（online softmax 核心）

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i
```

## 主 kernel（block_ptr 构建 + STAGE 分派）

```python
@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q = tl.load(Q_block_ptr)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX,
                                        V.dtype.element_ty == tl.float8e5)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                                        start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX,
                                        V.dtype.element_ty == tl.float8e5)

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
```

## 要点讲解

- **2D grid**：`(N_CTX // BLOCK_M, Z * H)`——axis0 为 Q 行块，axis1 为 batch×head 组合（`off_z = off_hz // H`, `off_h = off_hz % H`）。
- **STAGE 机制**：1=非因果（[0, N_CTX)）；2=因果对角块（[start_m*BLOCK_M, (start_m+1)*BLOCK_M)，块内做 causal mask）；3=因果完整（先 off-band 无 mask 循环再 on-band 带 mask 循环）。主 kernel 用 `STAGE & 1` / `STAGE & 2` 组合调用。
- **online softmax 状态**：`m_i`（running max，初值 -inf）、`l_i`（running sum，初值 1.0）、`acc`（fp32 累加器）。每块：`m_ij = max(m_i, rowmax(qk))`，`alpha = exp2(m_i - m_ij)`，`l_i = l_i*alpha + rowsum(p)`，`acc = acc*alpha + p@v`。收尾 `acc /= l_i`。
- **log2 域优化**：`qk_scale *= 1.44269504`（即 1/ln2），配合 `exp2/log2` 替代 `exp/log`，硬件上更高效；结果不一致时首先检查此转换。
- **推荐块大小**：BLOCK_M ∈ {64, 128}，BLOCK_N ∈ {32, 64}，HEAD_DIM ∈ {64, 128}。

## NPU 注意

- **`make_block_ptr` 的 `order`**：行优先 `order=(1, 0)`，列优先 `order=(0, 1)`（K 的转置靠 `shape=(HEAD_DIM, N_CTX)` + `order=(0, 1)` 表达）。NPU **只允许**用 `order` 表达转置，不能靠调整 stride 顺序。block_ptr 与复杂循环/分支搭配可能编译失败，此时改手动指针算术。
- **fp8 不支持**：NPU 当前无 `tl.float8e5`，V 用 fp16/bf16。
- **num_stages=1**：NPU 软件流水线策略与 GPU 不同，autotune 时固定 `num_stages=1`，`num_warps ∈ {4, 8}`。
- **进一步优化**：`al.compile_hint(x, "dot_pad_only_k")`、`al.multibuffer(buf, 2)` 存算并行、`al.sync_block_set/wait` 做 Cube/Vector 同步（CV 融合）。
- causal 结果错误时先查 STAGE 参数：causal=True 应传 STAGE=3。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [06-fused-attention.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/tutorials/06-fused-attention.py)
