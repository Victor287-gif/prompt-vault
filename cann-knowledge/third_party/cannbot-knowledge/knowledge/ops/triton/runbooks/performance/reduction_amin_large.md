---
type: Runbook
title: 极大规模 1D Amin 归约优化
description: 极大规模1D归约（amin）优化：二次切分避免超UB+计算重组减少归约次数，网格数接近AI Core数量（grid=32）、UB用满、无尾块时性能最优（9.61us），适用于极大规模1D数据（400万级元素）的全量归约场景
tags: [task.optimization, topic.compute.reduction, topic.tiling, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 极大规模 1D Amin 归约优化

## 任务特征
- **数据尺寸**：(4194304,)，极大规模1D数据

## 优化 1：二次切分

```python
pid = tl.program_id(0)
for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = pid * BLOCK_SIZE + start + tl.arange(0, SUB_BLOCK_SIZE)
```

## 优化 2：计算重组

```python
# 错误：简单
row_min = float('inf')
for n_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    curr_min = tl.min(block_data)
    row_min = tl.minimum(curr_min, row_min)

# 正确：优化
curr_min = tl.full((SUB_BLOCK_SIZE,), float('inf'), dtype=tl.float32)
for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    curr_min = tl.minimum(curr_min, block_data)
min_val = tl.min(curr_min)
```

## Autotune 配置

```python
# （AI core=40）
# 1. grid=16<40, UB用满 -> 15.12 us
triton.Config({'BLOCK_SIZE': 262144, 'SUB_BLOCK_SIZE': 16384})

# 2. grid=32<40, UB用满 -> 9.61 us 最优
triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 16384})

# 3. grid=32, UB未用满 -> 10.29 us
triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 8192})

# 4. grid=40, UB用满, 有尾块 -> 10.17 us
triton.Config({'BLOCK_SIZE': 104858, 'SUB_BLOCK_SIZE': 16384})

# 5. grid=64>40, UB用满 -> 11.64 us
triton.Config({'BLOCK_SIZE': 65536, 'SUB_BLOCK_SIZE': 32768})
```

### 总结
网格数接近AI Core数量、UB用满、无尾块时性能最优。二次切分避免超出硬件缓存。
## 实测观测（迁移自源卡 frontmatter）

- **现象**：shape_axis_slow
- **负载**：数据尺寸：(4194304,)，极大规模1D数据
- **解决层**：tiling
- **严重度**：medium
- **置信度**：single_run
- **复现次数**：1
- **信号**：
  - 1. grid=16<40, UB用满 -> 15.12 us
  - 2. grid=32<40, UB用满 -> 9.61 us 最优
  - 3. grid=32, UB未用满 -> 10.29 us
- **实测最优延迟**：9.61 us（source_case: reduction-amin-large）
  - 1. grid=16<40, UB用满 -> 15.12 us
  - 2. grid=32<40, UB用满 -> 9.61 us 最优
  - 3. grid=32, UB未用满 -> 10.29 us
  - 4. grid=40, UB用满, 有尾块 -> 10.17 us
  - 5. grid=64>40, UB用满 -> 11.64 us
  - 极大规模1D归约（amin）优化：二次切分避免超UB+计算重组减少归约次数，网格数接近AI Core数量（grid=32）、UB用满、无尾块时性能最优（9.61us），适用于极
- **证据**：benchmark_run `reduction_amin_large`，最优 9.61us（源插件仓库 `ops/triton-op-designer/references/cases/reduction-amin-large.md`，未搬入）
