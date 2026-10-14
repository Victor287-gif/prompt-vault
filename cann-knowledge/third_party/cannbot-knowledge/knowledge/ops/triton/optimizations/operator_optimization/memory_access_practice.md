---
type: Optimization
title: 内存访问优化实践
description: 对于 2D 张量，使用 tl.make_block_ptr 可简化索引并提升访存效率：。
tags: [task.optimization, topic.tiling, topic.implementation.masking, topic.data_movement.memory_access, topic.compute.elementwise.arithmetic, topic.compute.reduction]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md
  role: primary
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 内存访问优化实践

## 4. 内存访问优化

### 4.1 Block Size 选择

| 算子类型 | 推荐 BLOCK_SIZE | 说明 |
|----------|-----------------|------|
| Element-wise | 1024–2048 | 平衡并行度与资源占用 |
| Reduce | 256 / 128 | 归约需要更多寄存器 |
| MatMul | M=128, K=256, N=256（或 128/128/32） | 满足 512B 行宽对齐 |

### 4.2 访问优化
- **2D数据**: 优先使用 `tl.make_block_ptr` 配合 `boundary_check`，自动优化内存合并
- **步幅设计**: 仔细设计stride参数，错误设置会严重影响性能
- **数据布局**: 保持内存访问的连续性和局部性

### 4.3 2D 数据访问：tl.make_block_ptr

对于 2D 张量，使用 `tl.make_block_ptr` 可简化索引并提升访存效率：

```python
# 2D 块访问示例
ptr = tl.make_block_ptr(
    base=a_ptr,
    shape=(M, K),
    strides=(K, 1),
    offsets=(block_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0)
)
a = tl.load(ptr, boundary_check=(0, 1), padding="zero")
```

### 4.4 1D 连续内存优化

**优先使用 `.contiguous()` + 1D 访问**，避免 stride 访问带来的额外开销：

```python
class ModelNew(torch.nn.Module):
    def forward(self, input_tensor):
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        output_tensor = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
        return output_tensor

@triton.jit
def elementwise_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 4.4 对齐规则

- **256B**：基础对齐单位
- **512B**：MatMul 等带宽敏感算子推荐（256 个 fp16/bf16 元素）

---

## 失效边界与不触发场景

- **不触发**：算子不含本卡所列代码特征时不适用。
- **失效**：切分策略或 kernel 结构变更后，本卡基于旧结构的结论需重新判定。
