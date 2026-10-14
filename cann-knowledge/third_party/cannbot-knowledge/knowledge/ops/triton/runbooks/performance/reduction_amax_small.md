---
type: Runbook
title: 极小规模 Amax 归约优化
description: 极小规模归约（amax）优化：单核处理（grid=1）优于多核并行（2.16us vs 3.51us），避免并行化带来的调度开销，适用于数据规模很小（<1000元素）的归约场景
tags: [task.optimization, topic.compute.reduction, topic.tiling]
status: stable
sources: []
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 极小规模 Amax 归约优化

## 任务特征
- **数据尺寸**：(16, 16)，非常小
- **策略**：单核处理优于多核并行

## 优化：单核/少核处理

```python
# 1. 单核处理 -> 性能：2.16 us 最优
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16})

# 2. 多核并行 -> 性能：3.51 us
triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16})
```

### 总结
对于小规模计算任务，单核（少核）处理性能更优，并行化带来的开销超过了收益。
## 实测观测（迁移自源卡 frontmatter）

- **现象**：below_compute_capacity
- **负载**：数据尺寸：(16, 16)，非常小 策略：单核处理优于多核并行
- **解决层**：tiling
- **严重度**：medium
- **置信度**：single_run
- **复现次数**：1
- **信号**：
  - 1. 单核处理 -> 性能：2.16 us 最优
  - 2. 多核并行 -> 性能：3.51 us
  - 极小规模归约（amax）优化：单核处理（grid=1）优于多核并行（2.16us vs 3.51us），避免并行化带来的调度开销，适用于数据规模很小（<1000元素）的归约场景
- **实测最优延迟**：2.16 us（source_case: reduction-amax-small）
  - 1. 单核处理 -> 性能：2.16 us 最优
  - 2. 多核并行 -> 性能：3.51 us
  - 极小规模归约（amax）优化：单核处理（grid=1）优于多核并行（2.16us vs 3.51us），避免并行化带来的调度开销，适用于数据规模很小（<1000元素）的归约场景
- **证据**：benchmark_run `reduction_amax_small`，最优 2.16us（源插件仓库 `ops/triton-op-designer/references/cases/reduction-amax-small.md`，未搬入）
