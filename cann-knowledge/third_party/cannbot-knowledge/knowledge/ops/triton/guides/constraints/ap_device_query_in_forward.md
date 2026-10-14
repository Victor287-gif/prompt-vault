---
type: Guide
title: 反模式 — 在 forward 中查询设备属性
description: import triton.runtime.driver as driver。
tags: [task.design, topic.programming_model.execution_unit.vector, topic.tiling, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 反模式 — 在 forward 中查询设备属性

#### 重要注意事项

**禁止在 forward 中重复查询设备属性**

```python
import torch_npu
import triton.runtime.driver as driver

# 错误：错误：每次 forward 都重复查询设备属性，引入无谓 host 开销
class ModelNew(torch.nn.Module):
    def forward(self, input_tensor):
        core_num = driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]
        grid = (core_num,)
        ...

# 正确：正确：在 __init__ 中查询一次并缓存，forward 直接复用
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.VEC_CORE_NUM = driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]

    def forward(self, input_tensor):
        grid = (self.VEC_CORE_NUM,)
        ...
```

**原因**：每次 forward 都重复查询设备属性是无谓的 host 开销；应在 `__init__` 查一次缓存为 `self.VEC_CORE_NUM` / `self.CUBE_CORE_NUM`，forward 直接复用。新 API 不再触发旧 API 那种 device init 同步，但 per-forward 重复调用仍是浪费，且会引入对 `driver` 的运行期依赖。

---
