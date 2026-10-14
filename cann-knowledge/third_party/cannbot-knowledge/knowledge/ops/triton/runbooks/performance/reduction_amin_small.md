---
type: Runbook
title: 中等规模 1D Amin 归约优化
description: 中等规模1D归约（amin）优化：适中并行度（grid=8时最优2.21us），存在最优平衡点（过小导致单块负载过重、过大引入调度开销），适用于中等规模1D数据（6万级元素）的全量归约场景
tags: [task.optimization, topic.compute.reduction, topic.tiling]
status: stable
sources: []
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 中等规模 1D Amin 归约优化

## 任务特征
- **数据尺寸**：(65536,)，中等规模1D数据

## 优化：适中并行度

```python
# （AI core=40）
# 1. grid=4<40 -> 2.47 us
triton.Config({'BLOCK_SIZE': 16384})

# 2. grid=8<40 -> 2.21 us 最优
triton.Config({'BLOCK_SIZE': 8192})

# 3. grid=32<40 -> 2.92 us
triton.Config({'BLOCK_SIZE': 2048})

# 4. grid=40 -> 3.44 us
triton.Config({'BLOCK_SIZE': 1639})

# 5. grid=128>40 -> 6.70 us
triton.Config({'BLOCK_SIZE': 512})
```

### 总结
中等规模数据，网格数适当小时性能最佳。存在最优并行度平衡点：网格数过小导致单块负载过重，过大则引入过多调度开销。
## 实测观测（迁移自源卡 frontmatter）

- **现象**：below_compute_capacity
- **负载**：数据尺寸：(65536,)，中等规模1D数据
- **解决层**：tiling
- **严重度**：medium
- **置信度**：single_run
- **复现次数**：1
- **信号**：
  - 1. grid=4<40 -> 2.47 us
  - 2. grid=8<40 -> 2.21 us 最优
  - 3. grid=32<40 -> 2.92 us
- **实测最优延迟**：2.21 us（source_case: reduction-amin-small）
  - 1. grid=4<40 -> 2.47 us
  - 2. grid=8<40 -> 2.21 us 最优
  - 3. grid=32<40 -> 2.92 us
  - 4. grid=40 -> 3.44 us
  - 5. grid=128>40 -> 6.70 us
  - 中等规模1D归约（amin）优化：适中并行度（grid=8时最优2.21us），存在最优平衡点（过小导致单块负载过重、过大引入调度开销），适用于中等规模1D数据（6万级元素）的
- **证据**：benchmark_run `reduction_amin_small`，最优 2.21us（源插件仓库 `ops/triton-op-designer/references/cases/reduction-amin-small.md`，未搬入）
