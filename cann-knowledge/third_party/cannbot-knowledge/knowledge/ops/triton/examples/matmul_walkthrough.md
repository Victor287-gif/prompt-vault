---
type: Example
title: Matmul 教学 Walkthrough（GPU 基线 → NPU 对角线分核）
description: 逐步教学：从标准 GPU matmul kernel 出发，讲解分块/fp32 累加，再给出 NPU 对角线分核 + dot_pad_only_k
  + multibuffer 优化版。
tags: [task.implementation, topic.compute.composite, topic.programming_model.execution_unit.cube, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---
# Matmul 教学 Walkthrough

> 定位：教学向逐步讲解（GPU 基线 → NPU 优化）。直接可套用的生产骨架见 [gemm_basic.md](./gemm_basic.md)（block_ptr + 列优先 PID 分组版本）。

## GPU 基线 kernel（教学起点，不针对 NPU 优化）

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)
```

基线要点：GROUP_SIZE_M 分组的 pid→(pid_m, pid_n) 映射提升 L2 复用；`tl.dot(a, b, acc)` fp32 累加，K 循环推进指针；写回前可做激活融合。

## NPU 优化版：对角线分核 + compile_hint

水平分核在大 shape 下有两个问题：大量核心同时读同一块左矩阵产生 Bank 冲突；右矩阵超 L2 容量导致 Cache Miss。对角线分核按 N×N 方块沿对角线分配任务缓解两者。

```python
import triton.language.extra.cann.extension as al

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
    ],
    key=["M", "N", "K"]
)
@triton.jit
def matmul_kernel(
        mat_a, mat_b, mat_c,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    if NUM_BLOCKS_M >= BLOCK_TRESHHOLD and NUM_BLOCKS_N >= BLOCK_TRESHHOLD:
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            curThresholdM = BLOCK_TRESHHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_TRESHHOLD
            curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
            curThresholdN = BLOCK_TRESHHOLD if block_idx % (NUM_BLOCKS_N * BLOCK_TRESHHOLD) < (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN else NUM_BLOCKS_N % BLOCK_TRESHHOLD
            localRelativeBlock = block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) % (BLOCK_TRESHHOLD * curThresholdM)
            task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_TRESHHOLD * NUM_BLOCKS_N) * BLOCK_TRESHHOLD
            x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
            while y != 0:
                x, y = y, x % y
            lcm = curThresholdM * curThresholdN // x
            task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_TRESHHOLD

            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N

            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (k_start + tl.arange(0, BLOCK_K))[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
                al.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
                al.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
    else:
        # 小矩阵：传统顺序分核即可（block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N），
        # 内层 K 循环与上一分支完全相同
        for block_idx in range(pid, NUM_BLOCKS, num_cores):
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            # ... 与对角线分支相同的 K 循环加载/dot/store ...
```

Host 端以实际 AICore 数为 grid：

```python
from triton.runtime import driver

def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

def triton_matmul(mat_a, mat_b):
    m, k = mat_a.shape
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    num_cores = get_npu_properties()["num_aicore"]
    matmul_kernel[(num_cores,)](mat_a, mat_b, mat_c, m, n, k, num_cores)
    return mat_c
```

## NPU 要点

- **推荐分块**（512B 对齐亲和）：`BLOCK_M=128, BLOCK_N=256, BLOCK_K=256`。
- **UB 估算**（bf16 输入 + fp32 累加，128×256×256）：`128*256*2 + 256*256*2 + 128*256*4 ≈ 320KB`，需开 double buffer（A2/A3 96KB×2=192KB）或减小分块；910_95 UB 256KB 更充裕。
- **`al.compile_hint(x, "dot_pad_only_k")`**：NPU 上 `tl.dot` 要求 M/N 维对齐；该 hint 告知编译器 M/N 已合法，仅 K 维补零，减少 padding 开销。所有 matmul kernel 通用。
- **`al.multibuffer(buf, 2)`**：多缓冲区实现存算并行（Cube+Vector 混合 kernel）。
- **对角线分核适用条件**：M、N 方向块数均 ≥ 阈值（如 8）才有优势；小矩阵用顺序分核。
- **输出 dtype**：NPU 上 bf16 优于 fp16；fp32 累加下 bf16/fp16 输入的 matmul 与 PyTorch 差 1e-2 内属正常。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [03-matrix-multiplication.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/tutorials/03-matrix-multiplication.py)
- [09-persistent-matmul.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/09-persistent-matmul.py)
