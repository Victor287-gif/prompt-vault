---
type: Guide
title: 大规模 Amax 归约优化（reduce轴映射多核）
description: 非reduce轴很小、reduce轴很大的归约优化：将reduce轴映射到多核（而非常规的非reduce轴），使用原子操作跨线程块归约，通过二次切分避免超UB，适用于极端shape比例（M<<N如16×262144）的归约场景
tags: [task.design, topic.compute.reduction, topic.tiling, topic.implementation.memory_management, topic.implementation.atomic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/sum.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/reduce.md
  role: supporting
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 大规模 Amax 归约优化（reduce轴映射多核）

> 本卡源自 designer 设计案例，仅含**定性**配置结论（如「性能最优」
> 「并行度低」），无定量实测数值，故按 SPEC §9 不作为 `perf_card`。
> 全局约束见 [global_constraints.md](global_constraints.md)。

## 任务特征
- **数据尺寸**：(16, 262144)，非reduce轴很小，reduce轴很大
- **策略**：将reduce轴映射到多核，使用原子操作

## 优化 1：切分策略调整

```python
# 错误：简单方式：非reduce轴映射多核
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)

# 正确：优化方式：reduce轴映射多核
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)

# Kernel内对列进行二次切分
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
```

## 优化 2：原子操作

### 方案一：循环内原子操作
```python
for m_start in range(0, M, BLOCK_SIZE_M):
    row_min = tl.min(curr_min, 1)
    tl.atomic_min(output_ptrs, row_min, mask=mmask)
```

### 方案二：循环外原子操作
```python
all_row_min = tl.full((M,), float('inf'), dtype=tl.float32)
for m_start in range(0, M, BLOCK_SIZE_M):
    row_min = tl.min(curr_min, 1)
    all_row_min = tl.insert_slice(all_row_min, row_min, ...)
tl.atomic_min(output_ptrs, all_row_min)
```

## 优化 3：配置

```python
# grid=32<40, UB用满
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 1024})
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 512})
```

### 总结
非reduce轴很小、reduce轴很大时，将reduce轴映射到多核并结合原子操作，通过二次切分避免超出UB。
