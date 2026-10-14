---
type: Example
title: 融合 Softmax 示例（Fused Softmax on NPU）
description: 教学示例：单 kernel 完成行级 softmax（减最大值数值稳定 + 行归约），含 grid-stride 多行循环与 NPU UB
  估算。
tags: [task.implementation, topic.compute.normalization, topic.compute.composite, topic.compute.reduction, topic.precision, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/02-fused-softmax.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---
# 融合 Softmax 示例（Fused Softmax on NPU）

kernel fusion 的经典教学案例：原生 PyTorch softmax 需多次读写 Global Memory，融合后单行仅 1 次读 + 1 次写，max/exp/sum/div 全部在片上完成。

## 完整代码

```python
import torch
import torch_npu

import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_programs = 32
    y = torch.empty_like(x)
    num_programs = min(num_programs, n_rows)
    softmax_kernel[(num_programs, 1, 1)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE
    )
    return y

torch.manual_seed(0)
x = torch.randn(1823, 781, device='npu')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```

## 要点讲解

- **数值稳定性**：利用平移不变性 `softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`，减最大值后 exp 参数 ≤ 0，避免溢出。
- **2 的幂填充 + mask**：`BLOCK_SIZE = triton.next_power_of_2(n_cols)`；`n_cols` 非 2 的幂时 `other=-float('inf')` 使 padding 位置 exp 值为 0，不污染 sum。
- **grid-stride 行循环**：`row_start = program_id(0)`，`row_step = num_programs(0)`，`tl.range` 交错分配行，固定 program 数即可覆盖任意行数，利于负载均衡。
- **归约轴**：`row` shape 为 `[BLOCK_SIZE]`，`tl.max/tl.sum(row, axis=0)` 归约为标量（每行一个）。
- **前提**：本实现要求整行一次装入片上（`BLOCK_SIZE >= n_cols`）；`n_cols` 超 UB 时需改列分块 + 增量 max/sum（online softmax，见 flash_attention_walkthrough）。

## NPU 适配要点

1. **UB 估算**（fp32）：`UB 占用 ≈ BLOCK_SIZE * 4B * 3`（row / numerator / output）。要求 `3 * BLOCK_SIZE * 4 <= 96KB`（A2/A3，double buffer 开）或 `<= 128KB`（910_95），即 BLOCK_SIZE ≤ 8192（A2/A3）/ ≤ 10922（910_95）。
2. **512B 对齐**：fp32 推荐 BLOCK_SIZE 为 128 的倍数（128/256/512/1024 按 n_cols 取）。
3. **program 数**：`num_programs = min(32, n_rows)`；可用 `driver.active.utils.get_device_properties(device)["num_aicore"]` 取实际核心数。
4. **精度**：NPU 上 `tl.exp` 为近似计算；fp16 输入建议 kernel 内部转 fp32 计算、写回时再转回。
5. causal 变体：load 时叠加 `causal_mask = col_offsets <= row_idx`，padding 值仍为 `-inf`。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [02-fused-softmax.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/tutorials/02-fused-softmax.py)
- [02_fused_softmax_example.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/examples/02_fused_softmax_example.md)
