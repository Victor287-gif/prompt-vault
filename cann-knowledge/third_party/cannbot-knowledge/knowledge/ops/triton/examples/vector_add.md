---
type: Example
title: 向量加法示例（Vector Add on NPU）
description: Triton-Ascend 入门教学示例：逐行讲解 SPMD 向量加法 kernel 的 grid 计算、mask 边界保护与 NPU 适配要点。
tags: [task.implementation, topic.compute.elementwise.arithmetic, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/01-vector-add.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---
# 向量加法示例（Vector Add on NPU）

最基础的 SPMD 模式教学：每个 program 处理 `BLOCK_SIZE` 个元素的分片，展示 `program_id` / `arange` / `mask` / `cdiv` 的最小用法。向量加法本身是 memory-bound，单独写 kernel 不会比 PyTorch 快，价值在于作为融合与更复杂 kernel 的骨架。

## 完整代码

```python
import torch
import torch_npu

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

## 要点讲解

- **grid 计算**：1D grid，大小 `triton.cdiv(n_elements, BLOCK_SIZE)`，cdiv 向上取整保证覆盖尾块。
- **偏移构造**：`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`，即本 program 负责的全局元素下标。
- **mask 边界保护**：`n_elements` 非 BLOCK_SIZE 整数倍时尾块越界，`tl.load(..., mask=mask)` 越界位置填 0，`tl.store` 越界位置不写。未 mask 的越界 load 会读到垃圾值（NaN 的常见来源）。
- **BLOCK_SIZE**：编译期 `tl.constexpr`，取 2 的幂；NPU 上建议 256 的倍数。默认 1024，大向量可试 2048/4096。
- **超大向量（> 2^31 元素）**：偏移和 `n_elements` 需用 int64。

## NPU 适配要点

1. `import torch_npu` 注册后端，张量用 `device='npu'`；无需 `assert x.is_cuda`。
2. **512B 对齐**：BLOCK_SIZE 按 dtype 取对齐倍数——fp32 取 128 的倍数，fp16/bf16 取 256 的倍数。
3. **UB 空间限制**：片上同时存在 x/y/output 三个 tensor，需满足 `3 * BLOCK_SIZE * element_size` 不超 UB——A2/A3 为 96KB（double buffer 开）/ 192KB（关）；910_95 为 128KB（开）/ 256KB（关）。
4. **benchmark**：NPU 上用 `do_bench_npu`，或设 `export TRITON_BENCH_METHOD=npu`。
5. 简单 elementwise 手动选 BLOCK_SIZE 即可；如需 autotune，`key=['n_elements']`，configs 扫 `{256, 512, 1024, 2048}`。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [01-vector-add.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/01-vector-add.py)
- [01-vector-add.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/tutorials/01-vector-add.py)
