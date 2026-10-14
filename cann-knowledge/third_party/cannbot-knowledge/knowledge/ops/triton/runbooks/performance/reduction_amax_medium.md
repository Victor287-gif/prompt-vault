---
type: Runbook
title: 中等规模 Amax 归约优化
description: 中等规模归约（amax）优化：计算重组（循环内累加、循环外归约）减少归约次数，grid=40等于核数时性能最优（25.73us），适用于非reduce轴中等、reduce轴较大（千万级元素）的2D归约场景
tags: [task.optimization, topic.compute.reduction, topic.tiling, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 中等规模 Amax 归约优化

## 任务特征
- **数据尺寸**：(2048, 8192)，非reduce轴中等，reduce轴较大

## 优化 1：计算重组

```python
# 错误：简单方式：循环内多次归约
row_max = -float('inf')
for n_offset in range(0, N, BLOCK_SIZE):
    curr_max = tl.max(data_block, 1)
    row_max = tl.maximum(curr_max, row_max)

# 正确：优化方式：维护矩阵结构，循环外归约
curr_max = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), -float('inf'), dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    curr_max = tl.maximum(data_block, curr_max)
row_max = tl.max(curr_max, 1)
```

## 优化 2：Grid 配置

```python
# （AI core=40）
# 1. grid=32<40, UB用满 -> 29.05 us
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256})

# 2. grid>40, UB用满 -> 29.09 us
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512})

# 3. grid=40, UB不超出 -> 25.73 us 最优
triton.Config({'BLOCK_SIZE_M': 52, 'BLOCK_SIZE_N': 256})
```

### 总结
1. 计算重组：将多次归约合并为一次，减少归约次数
2. Grid配置：grid等于核数时性能最优，需确保UB不超出
## 实测观测（迁移自源卡 frontmatter）

- **现象**：shape_axis_slow
- **负载**：数据尺寸：(2048, 8192)，非reduce轴中等，reduce轴较大
- **解决层**：tiling
- **严重度**：medium
- **置信度**：single_run
- **复现次数**：1
- **信号**：
  - 1. grid=32<40, UB用满 -> 29.05 us
  - 2. grid>40, UB用满 -> 29.09 us
  - 3. grid=40, UB不超出 -> 25.73 us 最优
- **实测最优延迟**：25.73 us（source_case: reduction-amax-medium）
  - 1. grid=32<40, UB用满 -> 29.05 us
  - 2. grid>40, UB用满 -> 29.09 us
  - 3. grid=40, UB不超出 -> 25.73 us 最优
  - 中等规模归约（amax）优化：计算重组（循环内累加、循环外归约）减少归约次数，grid=40等于核数时性能最优（25.73us），适用于非reduce轴中等、reduce轴较大
- **证据**：benchmark_run `reduction_amax_medium`，最优 25.73us（源插件仓库 `ops/triton-op-designer/references/cases/reduction-amax-medium.md`，未搬入）
