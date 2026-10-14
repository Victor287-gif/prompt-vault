---
type: Guide
title: Grid 启动限制与核心数查询
description: import triton.runtime.driver as driver。
tags: [task.implementation, topic.tiling, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.cube, topic.compute.reduction, topic.compute.composite]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/README.md
  role: supporting
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Grid 启动限制与核心数查询

### 3.1 Grid 维度限制

- **Grid 必须是 tuple 类型**，最多 3 维
- **支持的格式**：`(x,)`, `(x, y)`, `(x, y, z)`
- **Grid 各维度最大值**：65535
- **BLOCK_SIZE 限制**：单块元素数 < 65536

### 3.3 动态核心数获取

根据算子类型选择核心类型：

| 算子类型 | 核心类型 | 获取方式 |
|----------|----------|----------|
| Element-wise、Reduce | **VEC_CORE_NUM**（向量核心） | `driver.active.utils.get_device_properties(device)["num_vectorcore"]` |
| MatMul | **CUBE_CORE_NUM**（矩阵核心） | `driver.active.utils.get_device_properties(device)["num_aicore"]` |

**关键规则**：必须在 `__init__` 中获取核心数，**不能在 `forward` 中获取**，否则每次前向都会查询，影响性能。

```python
import torch_npu
import triton.runtime.driver as driver

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        self.VEC_CORE_NUM = properties["num_vectorcore"]
        self.CUBE_CORE_NUM = properties["num_aicore"]

    def forward(self, x):
        grid = (self.VEC_CORE_NUM,)
        kernel[grid](...)
```
