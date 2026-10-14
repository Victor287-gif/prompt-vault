---
type: Runbook
title: Reduction + Elementwise 融合算子优化
description: Reduction+Elementwise融合算子优化：先逐元素操作再归约，行二次切分+计算重组，grid=40且SUB切分不含尾块时性能最优（47.58us），融合优化逻辑以reduce为主，适用于需要先逐元素计算再reduce的融合场景
tags: [task.optimization, topic.compute.reduction, topic.tiling, topic.implementation.memory_management, topic.compute.composite]
status: stable
sources: []
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Reduction + Elementwise 融合算子优化

## 任务特征
- **数据尺寸**：(1000, 8192), (8192,)，融合算子
- **特点**：先进行向量化逐元素操作，再沿列方向求和归约

## 优化 1：行二次切分

```python
pid = tl.program_id(0)
for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
    m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
```

## 优化 2：计算重组

```python
# 错误：简单
total_sum = 0.0
for n_offset in range(0, N, BLOCK_SIZE):
    total_sum += tl.sum(tl.where(mask, t3, 0.0))

# 正确：优化
acc = tl.zeros([SUB_BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    acc += tl.where(mask, t3, 0.0)
total_sum = tl.sum(acc, axis=1)
```

## Autotune 配置

```python
# （AI core=40）
# 1. grid=20<40 -> 91.69 us
triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256})

# 2. grid=40，SUB切分含尾块 -> 53.30 us
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 2048})

# 3. grid=40，SUB切分不含尾块 -> 47.58 us 最优
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256})

# 4. grid>40，且非核数整数倍 -> 79.00 us
triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 256})
```

### 总结
融合算子优化逻辑以reduce为主。grid等于核数、SUB切分不含尾块时性能最优。
## 实测观测（迁移自源卡 frontmatter）

- **现象**：shape_axis_slow
- **负载**：数据尺寸：(1000, 8192), (8192,)，融合算子 特点：先进行向量化逐元素操作，再沿列方向求和归约
- **解决层**：tiling
- **严重度**：medium
- **置信度**：single_run
- **复现次数**：1
- **信号**：
  - 1. grid=20<40 -> 91.69 us
  - 2. grid=40，SUB切分含尾块 -> 53.30 us
  - 3. grid=40，SUB切分不含尾块 -> 47.58 us 最优
- **实测最优延迟**：47.58 us（source_case: reduction-sum-fused）
  - 1. grid=20<40 -> 91.69 us
  - 2. grid=40，SUB切分含尾块 -> 53.30 us
  - 3. grid=40，SUB切分不含尾块 -> 47.58 us 最优
  - 4. grid>40，且非核数整数倍 -> 79.00 us
  - Reduction+Elementwise融合算子优化：先逐元素操作再归约，行二次切分+计算重组，grid=40且SUB切分不含尾块时性能最优（47.58us），融合优化逻辑以
- **证据**：benchmark_run `reduction_sum_fused`，最优 47.58us（源插件仓库 `ops/triton-op-designer/references/cases/reduction-sum-fused.md`，未搬入）
