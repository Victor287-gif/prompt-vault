---
type: Optimization
title: Grid 形状与多路径特化
description: 单一 kernel 实现无法在不同 workload 规模下同时达到最优性能。
tags: [task.optimization, topic.tiling, topic.implementation.masking, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Grid 形状与多路径特化

## 概述

单一 kernel 实现无法在不同 workload 规模下同时达到最优性能。通过 Host 侧动态 dispatch，在运行时根据 workload 特征选择不同的 kernel 路径，可以消除小 workload 下的调度开销，同时保持大 workload 下的并行效率。

## 触发条件

**当代码中存在以下特征时，应考虑 Grid 形状与多路径特化：**

1. **Grid 被钳制到核数**：`grid = (min(total_blocks, num_cores),)` 导致小 workload 时调度开销占比高
2. **Kernel 内存在兼容大小 grid 的通用循环结构**：为了兼容"program 可能处理多 block"而引入的标量分区循环、分支判断
3. **同一算子同时存在多种 workload 场景**：
   - `total_blocks <= num_cores`（小 grid，每个 program 本可直接映射 1 个 block）
   - `total_blocks > num_cores`（大 grid，必须进行多核分区）

## 优化方法

### 原始代码（通用 kernel，兼顾大小 grid）

```python
@triton.jit
def kernel_generic(
    in_ptr, out_ptr,
    total_elements, num_cores,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    elements_per_core = tl.cdiv(total_elements, num_cores)
    core_start = pid * elements_per_core
    core_end = tl.minimum(core_start + elements_per_core, total_elements)

    for block_idx in range(tl.cdiv(core_end - core_start, BLOCK_SIZE)):
        block_start = core_start + block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < core_end
        data = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, data, mask=mask)
```

**问题分析：**
- 小 workload（如 total_elements=1024, num_cores=48）：每个 program 只处理 21 个元素，但 kernel 内仍有循环和分区计算
- 大 workload（如 total_elements=1M）：循环和分区计算是必要的

### 优化后代码（多路径特化）

```python
import torch_npu
import triton.runtime.driver as driver

# 路径 A：小 grid，每个 program 处理 1 个 block，无循环
@triton.jit
def kernel_small_grid(
    in_ptr, out_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    data = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, data, mask=mask)

# 路径 B：大 grid，多核分区，有循环
@triton.jit
def kernel_large_grid(
    in_ptr, out_ptr,
    total_elements, num_cores,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    elements_per_core = tl.cdiv(total_elements, num_cores)
    core_start = pid * elements_per_core
    core_end = tl.minimum(core_start + elements_per_core, total_elements)

    for block_idx in range(tl.cdiv(core_end - core_start, BLOCK_SIZE)):
        block_start = core_start + block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < core_end
        data = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, data, mask=mask)

# Host 侧动态 dispatch
def forward(self, x):
    total_elements = x.numel()
    num_cores = driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]

    # 阈值判断：当 total_blocks <= num_cores 时走小 grid 路径
    BLOCK_SIZE = 1024
    total_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    if total_blocks <= num_cores:
        # 小 grid 路径：每个 program 1 个 block
        grid = (total_blocks,)
        kernel_small_grid[grid](x, output, total_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # 大 grid 路径：多核分区
        grid = (num_cores,)
        kernel_large_grid[grid](x, output, total_elements, num_cores, BLOCK_SIZE=BLOCK_SIZE)

    return output
```

## 关键点

1. **阈值选择**：
   - 通常以 `total_blocks <= num_cores` 作为分界
   - 也可根据实测性能调整阈值（如 `total_elements < num_cores * BLOCK_SIZE * 2`）

2. **路径设计原则**：
   - **小 grid 路径**：消除所有循环和分区计算，直接 `pid * BLOCK_SIZE` 映射
   - **大 grid 路径**：保留完整的分区循环逻辑，确保负载均衡

3. **避免过度特化**：
   - 一般 2 条路径即可覆盖大部分场景
   - 超过 3 条路径会增加维护成本，收益递减

4. **与 autotune 配合**：
   - 每条路径可独立配置 autotune 参数
   - 小 grid 路径的 BLOCK_SIZE 可更大（因为无循环开销）

## 性能收益

- **小 workload 场景**：消除循环开销和分区计算，性能提升 1.2x~2.0x
- **大 workload 场景**：无额外开销，保持最优并行效率
- **综合收益**：在 workload 分布广泛的场景中，几何平均加速比提升 5%~15%

## 常见错误

### 错误 1：阈值设置不合理

```python
# ❌ 错误：阈值过高，导致大 workload 走小 grid 路径，并行度不足
if total_elements < 1000000:  # 阈值太高
    grid = (total_blocks,)

# ✅ 正确：基于 block 数量与核数的关系
if total_blocks <= num_cores:
    grid = (total_blocks,)
```

### 错误 2：小 grid 路径仍保留循环

```python
# ❌ 错误：小 grid 路径仍有循环，未消除核心开销
@triton.jit
def kernel_small_grid(...):
    pid = tl.program_id(0)
    for block_idx in range(...):  # 不应存在
        ...

# ✅ 正确：小 grid 路径直接映射，无循环
@triton.jit
def kernel_small_grid(...):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ...
```

### 错误 3：两条路径 BLOCK_SIZE 不一致导致结果差异

```python
# ❌ 错误：两条路径使用不同 BLOCK_SIZE，可能导致边界处理不一致
kernel_small_grid[grid](..., BLOCK_SIZE=1024)
kernel_large_grid[grid](..., BLOCK_SIZE=512)

# ✅ 正确：保持 BLOCK_SIZE 一致，或确保边界处理逻辑等价
```

## 适用算子

| 算子类型 | 适用性 | 说明 |
|----------|--------|------|
| Element-wise | 高 | 无依赖，直接映射 |
| Reduction | 中 | 需注意规约方向的一致性 |
| Transformation-memory | 高 | 数据搬运为主，收益明显 |
| Transformation-compute | 低 | 计算逻辑复杂，路径拆分成本高 |

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| Grid 形状特化 | Host 侧动态 dispatch | 消除小 workload 下的循环/分区开销 |
| 多路径特化 | 2 条 kernel 路径 | 各 workload 规模下均达到最优 |

**核心原则：**
1. 小 grid 路径：极简，无循环，直接映射
2. 大 grid 路径：完整分区逻辑，负载均衡
3. Host 侧阈值判断：基于 `total_blocks` 与 `num_cores` 的关系

---

## 失效边界与不触发场景

- **不触发**：不满足 Host 侧可在运行时根据 workload 选择不同 kernel 路径 时本优化点不适用，强行套用通常无收益甚至劣化。
- **不触发**：该改写与算子已有的其他优化手段冲突时（如已由 autotune 覆盖同一参数空间），不重复施加。
- **失效**：结构性改动（切分策略变更、kernel 分裂、模板切换）之后，本卡基于旧结构得出的收益结论失效，需重新判定。
