---
type: Optimization
title: "gcd：并行优化"
description: "gcd 算子的并行优化经验，关注冗余计算。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/Gcd.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/tensorflow_sqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd_common.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gcd：并行优化

## 来源信息

- experience ID：`0fc9e67326cea6a0`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_015_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.workload.redundancy`
- canonical family：`parallelism.workload.redundancy`
- 源记录时间：`2026-04-20T21:31:45`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Restore block-based work distribution to avoid redundant computation
~~~~

### Original summary

~~~~text
Applying block-based work distribution in ProcessFast() partitions tile processing across AI cores to avoid redundant computation. This optimization reduces floating-point operations by 23.8× and memory transfers by 22.8× while improving L2 cache hit rates from near 0% to 95.8%. Use this pattern when processing large input arrays with tile-based parallelism across multiple AI cores.
~~~~

### Original root mechanism

~~~~text
Removing block-based work distribution in ProcessFast() causes all AI cores to redundantly process all tiles instead of partitioning work, leading to N× redundant computation and memory transfers where N is the number of AI cores.
~~~~

### Original causal chain

~~~~text
Restoring block-based work distribution in ProcessFast() → reduces arithmetic.aiv_vec_fops from 13.3B to 558M (23.8× reduction) → reduces memory_ub.gm_to_ub_kb from 32.8MB to 1.3MB (22.8× reduction) → improves cache_l2.aiv_read_hit_rate_pct from 0.02% to 95.83% → reduces task_duration_us from 307.6ms to 12.9ms (23.8× speedup)
~~~~

## 适用条件

- Processing large input arrays with tile-based parallelism across multiple AI cores
- Using GetBlockIdx() and GetBlockNum() APIs for work distribution
- Implementing ProcessFast() function in Ascend C kernels
- When arithmetic.aiv_vec_fops shows high values relative to expected computation

## 失效边界

- When each AI core needs to process all data (e.g., reduction operations requiring full dataset)
- When tile size equals or exceeds total data size
- When using single AI core execution (block_dim = 1)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/gcd.cpp

### 改动符号

- KernelGcd::ProcessFast

### 代码签名

- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`95c320905ab2db1a14cdaeb9461f85289c202781370bf02478ee0cd25f19b729`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -329,7 +329,9 @@

     __aicore__ inline void ProcessFast()
     {
-        for (int i = GetBlockIdx() * TILE_SIZE;i < sizeX1;i+=TILE_SIZE * GetBlockNum()) {
+        // Degradation: Remove work distribution across blocks, forcing all blocks to process all tiles
+        // This creates redundant computation and memory transfers across all AI cores
+        for (int i = 0; i < sizeX1; i += TILE_SIZE) {
             CopyInX1(i, MinVal(sizeX1 - i, TILE_SIZE));
             CopyInX2(i, MinVal(sizeX1 - i, TILE_SIZE));
             Compute(MinVal(sizeX1 - i, TILE_SIZE));
````

## 观测记录

- bottleneck type：`computation_redundancy`
- slow duration us：`307637.59375`
- fast duration us：`12922.578125`
- improvement ratio：`0.9579941516006608`
- machine confidence：`0.95`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 0.02 -> 95.83 (+479050.00%)
- cache_l2.aiv_total_hit_rate_pct: 0.02 -> 95.83 (+479050.00%)
- cache_l2.aiv_write_hit_rate_pct: 0.07 -> 95.83 (+136800.00%)
- critical_path.aiv_vector0_time_us_mean: 12820.5 -> 307630 (+2299.52%)
- memory_ub.ub_to_gm_kb: 688 -> 16384 (+2281.40%)

### Affected metric groups

- cache_l2
- critical_path
- memory_ub
- arithmetic

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_read_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "cache_l2.aiv_write_hit_rate_pct",
  "critical_path.aiv_vector0_time_us_mean",
  "memory_ub.ub_to_gm_kb",
  "arithmetic.aiv_vec_fops",
  "memory_ub.gm_to_ub_kb",
  "latency.task_duration_us",
  "aiv_utilization.mte3_ratio_mean",
  "aiv_utilization.mte2_ratio_mean"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 1.0,
  "overall_deviation": 0.4,
  "metrics_checked": 0,
  "metrics_wrong": 0,
  "performance_metrics_checked": 0,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- In ProcessFast() function, use GetBlockIdx() * TILE_SIZE for loop start and TILE_SIZE * GetBlockNum() for stride
- Pattern: for (int i = GetBlockIdx() * TILE_SIZE; i < sizeX1; i += TILE_SIZE * GetBlockNum())
- Ensure each AI core processes distinct tile ranges to avoid redundant computation
- Monitor arithmetic.aiv_vec_fops metric - high values indicate redundant computation
- Check cache_l2 hit rates - near 0% hit rates with high memory transfers indicate poor data locality

## 检索线索

- block-based work distribution
- parallelism
- redundant computation
- tile processing
- GetBlockIdx
- GetBlockNum
- ProcessFast
- gcd
- math operator
- Ascend C
- workload partitioning
- AI core parallelism
