---
type: Optimization
title: 连续拷贝聚合优化
description: 对于纯内存拷贝型算子（Split / Chunk / Slice / Pad / Unbind 等），当多个输出块在输入侧构成连续内存区域时，可以通过聚合拷贝操作，减少
  kernel 启动次数和调度开销，显著提升性能。
tags: [task.optimization, topic.tiling, topic.implementation.masking, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md
  role: primary
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 连续拷贝聚合优化

## 概述

对于纯内存拷贝型算子（Split / Chunk / Slice / Pad / Unbind 等），当多个输出块在输入侧构成连续内存区域时，可以通过聚合拷贝操作，减少 kernel 启动次数和调度开销，显著提升性能。

## 触发条件

**当代码同时满足以下条件时，应启用连续拷贝聚合优化：**

1. **算子类型**：Split / Chunk / Slice / Unbind / Pad 等纯拷贝型算子
2. **内存连续性**：
   - 输入张量在被切分维度上连续（stride = 1 或切分 dim 为最后一维）
   - 所有输出块在输入侧的偏移满足 `offset[i+1] == offset[i] + size[i]`（无间隙）
3. **当前实现模式**：
   - 存在 `grid = (num_chunks, ...)` 或每个 program 只处理一个分块
   - 每个 program 处理的数据量小于 4096 元素（粒度太细）

## 优化方法

### 原始代码（逐 chunk 拷贝）

```python
@triton.jit
def split_by_chunk(
    in_ptr, out_ptrs,
    chunk_offsets_ptr,
    num_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    row = tl.program_id(1)

    # 加载 chunk 偏移量
    src_offset = tl.load(chunk_offsets_ptr + chunk_idx)
    chunk_size = tl.load(chunk_offsets_ptr + chunk_idx + 1) - src_offset

    # 只拷贝当前 chunk
    for i in range(0, chunk_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < chunk_size
        data = tl.load(in_ptr + src_offset + offsets, mask=mask)
        tl.store(out_ptrs + chunk_idx * max_chunk_size + offsets, data, mask=mask)
```

**问题分析：**
- grid = (num_chunks, num_rows)，如 (64, 16) = 1024，远超 48 核
- 每个 program 只处理一个 chunk，粒度太细
- 需要动态加载偏移量，增加访存开销

### 优化后代码（连续拷贝聚合）

```python
import torch_npu
import triton.runtime.driver as driver

@triton.jit
def copy_aggregated(
    in_ptr, out_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # 直接连续拷贝，无需 chunk 偏移量计算
    data = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, data, mask=mask)

# Host 侧：聚合多个连续 chunk
def forward(self, x, chunks):
    # 检查所有 chunk 在输入侧是否连续
    total_size = sum(chunks)
    if x.stride(-1) == 1 and all(
        x[..., sum(chunks[:i]):sum(chunks[:i+1])].is_contiguous()
        for i in range(len(chunks))
    ):
        # 连续聚合路径：一次拷贝所有数据
        output = x.clone()  # 或 torch.empty_like(x)
        num_cores = driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]
        grid = ((total_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        copy_aggregated[grid](x, output, total_size, BLOCK_SIZE=1024)

        # 在输出侧做 view/split 即可
        outputs = output.split(chunks, dim=-1)
    else:
        # 非连续路径：逐 chunk 拷贝（兜底）
        outputs = []
        offset = 0
        for chunk_size in chunks:
            out = torch.empty(...)
            grid = ((chunk_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            copy_aggregated[grid](
                x[..., offset:offset+chunk_size],
                out,
                chunk_size,
                BLOCK_SIZE=1024
            )
            outputs.append(out)
            offset += chunk_size

    return outputs
```

## 关键点

1. **连续性判断**：
   - 必须严格检查输入侧是否连续
   - 可通过 `tensor.is_contiguous()` 或手动检查 stride

2. **聚合粒度**：
   - 最小聚合单元：4096 元素（一个 BLOCK_SIZE）
   - 理想聚合粒度：>= num_cores * BLOCK_SIZE

3. **输出处理**：
   - 聚合拷贝后，输出侧通过 `view` / `split` / `slice` 获取各 chunk
   - 避免在 kernel 内处理 chunk 边界

4. **与多路径特化配合**：
   - 连续路径：聚合拷贝，grid = (total_blocks,)
   - 非连续路径：逐 chunk 拷贝，grid = (num_chunks,)

## 性能收益

- **连续场景**：
  - 减少 grid 发射数量：从 num_chunks 到 total_blocks / BLOCK_SIZE
  - 消除 chunk 偏移量加载开销
  - 性能提升：1.5x~5.0x（取决于 chunk 数量和大小）

- **非连续场景**：
  - 保持原有实现，无性能退化

## 常见错误

### 错误 1：未检查连续性直接聚合

```python
# ❌ 错误：未检查连续性，导致非连续数据被错误拷贝
output = x.clone()
copy_aggregated[grid](x, output, total_size)
outputs = output.split(chunks, dim=-1)

# ✅ 正确：严格检查连续性
if x.is_contiguous() and all_contiguous:
    # 聚合拷贝
else:
    # 逐 chunk 拷贝
```

### 错误 2：聚合后未正确处理输出 view

```python
# ❌ 错误：聚合拷贝后直接使用，未做 view/split
output = torch.empty(total_size)
copy_aggregated[grid](x, output, total_size)
# output 是一维的，需要 reshape/split

# ✅ 正确：聚合后做 view/split
output = torch.empty_like(x)
copy_aggregated[grid](x, output, total_size)
outputs = output.split(chunks, dim=-1)
```

### 错误 3：忽略 stride 检查

```python
# ❌ 错误：只检查 is_contiguous，未检查切分维度
if x.is_contiguous():  # 整体连续，但切分维度可能不连续
    ...

# ✅ 正确：检查切分维度的 stride
if x.stride(split_dim) == 1 or split_dim == x.dim() - 1:
    ...
```

## 适用算子

| 算子 | 适用性 | 说明 |
|------|--------|------|
| Split | 高 | 切分维度连续时可聚合 |
| Chunk | 高 | 同上 |
| Slice | 中 | 需检查 slice 范围是否连续 |
| Unbind | 中 | 需检查 unbind 维度是否连续 |
| Pad | 低 | 只有 constant 模式的 fill 部分可聚合 |

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| 连续拷贝聚合 | 合并多个连续 chunk 的拷贝 | 减少 kernel 启动次数，消除偏移量计算 |
| 多路径配合 | 连续路径聚合，非连续路径逐 chunk | 覆盖所有场景，无性能退化 |

**核心原则：**
1. 严格检查输入侧连续性
2. 聚合后通过 view/split 获取输出
3. 非连续场景保持原有实现作为兜底

---

## 失效边界与不触发场景

- **不触发**：不满足 满足连续性且当前按 chunk 细粒度分核 时本优化点不适用，强行套用通常无收益甚至劣化。
- **不触发**：该改写与算子已有的其他优化手段冲突时（如已由 autotune 覆盖同一参数空间），不重复施加。
- **失效**：结构性改动（切分策略变更、kernel 分裂、模板切换）之后，本卡基于旧结构得出的收益结论失效，需重新判定。
