---
type: Optimization
title: grid 超 65535 的两种分派改写
description: 对于输入 shape 较大的算子，直接按照 BLOCK_SIZE 切分得到的 grid 总数可能超过 65535。
tags: [task.optimization, topic.tiling, topic.implementation.masking, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# grid 超 65535 的两种分派改写

### 3.2 大 Shape 处理

#### 问题描述
对于输入 shape 较大的算子，直接按照 `BLOCK_SIZE` 切分得到的 grid 总数可能超过 65535。

**示例**：
- 输入 shape: `(327680, 1024)`
- BLOCK_SIZE: 1024
- Grid 需求: `327680 / 1024 = 320` (每行一个 block)
- 如果有多行：`327680 > 65535` 错误：超限！

---

当 `grid = (triton.cdiv(n, BLOCK_SIZE),)` 超过 65535 时，需采用以下策略之一。

#### 方法一：交错循环（Interleaved Loop）

#### 适用场景
- 按行/按块独立处理的算子
- Element-wise、Reduce、Normalization 等

#### 核心思想
- **固定 Grid 为核心数**
- **每个核心以步长方式交错处理数据**
- **负载均衡最好，代码最简洁**

每个核心循环处理多个块：

```python
@triton.jit
def kernel(ptr, n_elements, num_cores: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, num_blocks, num_cores):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(ptr + offsets, mask=mask, other=0.0)
        # compute...
        tl.store(ptr + offsets, result, mask=mask)
```

#### 方法二：连续分块法（Continuous Block Method）

##### 适用场景
- 连续内存访问优化

将一维 grid 映射为多维，或使用 `(grid_x, grid_y)` 形式：

```python
MAX_GRID = 65535
grid_x = min(triton.cdiv(n, BLOCK_SIZE), MAX_GRID)
grid_y = triton.cdiv(triton.cdiv(n, BLOCK_SIZE), MAX_GRID)
grid = (grid_x, grid_y)

# kernel 内: pid = tl.program_id(0) * grid_y + tl.program_id(1)
# 示例：处理大shape的向量操作
@triton.jit
def large_vector_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_GRID: tl.constexpr,  # 每个grid负责的元素总数
):
    pid = tl.program_id(0)

    # 计算当前grid负责的数据范围
    grid_start = pid * ELEMENTS_PER_GRID
    grid_end = min(grid_start + ELEMENTS_PER_GRID, n_elements)

    # 分块处理当前grid负责的数据
    for block_start in range(grid_start, grid_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < grid_end

        # 加载、计算、存储
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute_function(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# 启动方式
import torch_npu
import triton.runtime.driver as driver

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在__init__中获取核心数（新 API，无需设备 init）
        properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        self.VEC_CORE_NUM = properties["num_vectorcore"]

    def forward(self, input_tensor):
        n_elements = input_tensor.numel()
        BLOCK_SIZE = *  # 设置为尽可能大的合适的每次处理的值，充分利用ub
        MAX_GRID_SIZE = self.VEC_CORE_NUM  # 使用初始化时获取的核心数

        # 计算每个grid需要处理的元素数
        ELEMENTS_PER_GRID = triton.cdiv(n_elements, MAX_GRID_SIZE)
        # 向上取整到BLOCK_SIZE的倍数，确保循环能完整处理
        ELEMENTS_PER_GRID = triton.cdiv(ELEMENTS_PER_GRID, BLOCK_SIZE) * BLOCK_SIZE

        # 计算实际需要的grid数量
        grid_size = triton.cdiv(n_elements, ELEMENTS_PER_GRID)

        output_tensor = torch.empty_like(input_tensor)
        large_vector_kernel[grid_size,](
            input_tensor, output_tensor, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            ELEMENTS_PER_GRID=ELEMENTS_PER_GRID,
        )
        return output_tensor
```

## 失效边界与不触发场景

- **不触发**：`cdiv(n, BLOCK_SIZE)` 未超过 65535 时无需改写。
- **失效**：两种方法都要求块间无依赖；存在跨块归约依赖时不适用。
