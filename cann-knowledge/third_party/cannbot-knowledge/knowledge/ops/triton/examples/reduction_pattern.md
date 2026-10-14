---
type: Example
title: 归约操作模式集（行/列/全局归约与 NPU atomic 限制）
description: 教学示例集：行归约、大列分块累加、atomic 列归约、两阶段全局归约的完整 kernel，附 NPU 归约 dtype 支持矩阵与 atomic
  限制。
tags: [task.implementation, topic.compute.reduction, topic.implementation.atomic, topic.programming_model.execution_unit.vector, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/sum.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/reduce.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 归约操作模式集（Reduction Patterns）

归约是 Softmax / LayerNorm / Flash Attention 的公共子模式。本卡按场景给出完整 kernel。

## 行归约（最常用）

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def row_sum_kernel(input_ptr, output_ptr,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * n_cols
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(row, axis=0)
    tl.store(output_ptr + row_idx, row_sum)

def row_sum(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    row_sum_kernel[(n_rows,)](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

- 行最大值变体：`other=-float('inf')` + `tl.max(row, axis=0)`。
- **mask 的 other 取值必须匹配归约幺元**：sum 用 0.0，max 用 -inf，min 用 +inf，否则 padding 污染结果。

### 大 n_cols：分块累加

```python
@triton.jit
def row_sum_large_kernel(input_ptr, output_ptr,
                         n_rows, n_cols,
                         BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * n_cols
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        row = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
        partial_sum += row
    result = tl.sum(partial_sum, axis=0)
    tl.store(output_ptr + row_idx, result)
```

## 列归约（atomic 累加）

```python
@triton.jit
def col_sum_atomic_kernel(input_ptr, output_ptr,
                          n_rows, n_cols,
                          BLOCK_M: tl.constexpr):
    row_start = tl.program_id(0) * BLOCK_M
    cols = tl.arange(0, n_cols)
    rows = row_start + tl.arange(0, BLOCK_M)
    mask = rows[:, None] < n_rows
    data = tl.load(input_ptr + rows[:, None] * n_cols + cols[None, :], mask=mask, other=0.0)
    partial_sum = tl.sum(data, axis=0)
    tl.atomic_add(output_ptr + cols, partial_sum, mask=cols < n_cols)
```

> **NPU 注意**：`atomic_add` 不支持"多核 add + 保存中间结果"的用法；需要中间结果时改普通 add + 自旋锁（见 layer_norm 反向）。列归约访存不连续、带宽利用率低，可考虑转置后走行归约。

## 全局归约：两阶段（避免 atomic 精度问题）

```python
@triton.jit
def global_sum_stage1_kernel(input_ptr, partial_sums,
                             n_elements, num_blocks,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_sum = tl.sum(data, axis=0)
    tl.store(partial_sums + pid, block_sum)

@triton.jit
def global_sum_stage2_kernel(partial_sums, output_ptr,
                             num_blocks,
                             BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    data = tl.load(partial_sums + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(data, axis=0)
    tl.store(output_ptr, total)
```

## 自定义归约（tl.reduce）

```python
@triton.jit
def my_reduce(a, b):
    return tl.maximum(a, b)

result = tl.reduce(data, axis=0, combine_fn=my_reduce)
```

## NPU 归约 dtype 支持矩阵

| 归约操作 | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|---------|------|-------|-------|-------|------|------|------|------|
| `tl.sum` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| `tl.max` / `tl.min` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| `tl.argmax` / `tl.argmin` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| `tl.reduce` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |
| `tl.xor_sum` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓* |

✓* = Triton 内部将 bool 转 int8 运算。

## NPU atomic 限制汇总

- `atomic_add`：不支持多核 add+保存中间结果。
- `atomic_or / atomic_xor / atomic_and / atomic_xchg / atomic_cas`：暂不支持在 loop 中使用。
- `sem` 只支持默认 `"acq_rel"`，`scope` 只支持默认 `"gpu"`。
- 精度：归约内部用 fp32 累加（fp16 直接 sum 有精度损失），写回时再转目标 dtype。