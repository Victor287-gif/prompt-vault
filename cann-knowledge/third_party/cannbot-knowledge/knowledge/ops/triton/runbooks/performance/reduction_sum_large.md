---
type: Runbook
title: 大规模 Sum 归约优化
description: 大规模归约（sum）非reduce轴很大优化：计算重组减少归约次数，在优先占满UB前提下为reduce轴分配较大切分尺寸（BLOCK_SIZE_N=1024最优685.65us），适用于非reduce轴非常大（6万+）、reduce轴中等（千级）的2D归约场景
tags: [task.optimization, topic.compute.reduction, topic.tiling, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 大规模 Sum 归约优化

## 任务特征
- **数据尺寸**：(65536, 2048)，非reduce轴非常大，reduce轴中等

## 优化：reduce轴大切分 + 计算重组

```python
# 简单
total_sum = 0.0
for n_offset in range(0, N, BLOCK_SIZE):
    row_sum += tl.sum(block_vals)

# 正确：优化
acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    acc += block_vals
row_sum = tl.sum(acc, axis=1)
```

## Autotune 配置

```python
# 1. reduce轴切分较小，UB占满 -> 700.42 us
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256})

# 2. reduce轴切分增至512 -> 695.08 us
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512})

# 3. reduce轴切分增至1024 -> 685.65 us 最优
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1024})

# 4. reduce轴切分增至2048 -> 686.89 us
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048})

# 5. reduce轴切分较大，UB未占满 -> 743.83 us
triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 2048})
```

### 总结
在优先占满UB前提下，为reduce轴分配较大切分尺寸，减少循环次数。配置3和4性能最优，共同特征：reduce轴切分值较大且占满UB。
## 实测观测（迁移自源卡 frontmatter）

- **现象**：shape_axis_slow
- **负载**：数据尺寸：(65536, 2048)，非reduce轴非常大，reduce轴中等
- **解决层**：tiling
- **严重度**：medium
- **置信度**：single_run
- **复现次数**：1
- **信号**：
  - 1. reduce轴切分较小，UB占满 -> 700.42 us
  - 2. reduce轴切分增至512 -> 695.08 us
  - 3. reduce轴切分增至1024 -> 685.65 us 最优
- **实测最优延迟**：685.65 us（source_case: reduction-sum-large）
  - 1. reduce轴切分较小，UB占满 -> 700.42 us
  - 2. reduce轴切分增至512 -> 695.08 us
  - 3. reduce轴切分增至1024 -> 685.65 us 最优
  - 4. reduce轴切分增至2048 -> 686.89 us
  - 5. reduce轴切分较大，UB未占满 -> 743.83 us
  - 大规模归约（sum）非reduce轴很大优化：计算重组减少归约次数，在优先占满UB前提下为reduce轴分配较大切分尺寸（BLOCK_SIZE_N=1024最优685.65us
- **证据**：benchmark_run `reduction_sum_large`，最优 685.65us（源插件仓库 `ops/triton-op-designer/references/cases/reduction-sum-large.md`，未搬入）
