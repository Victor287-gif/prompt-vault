---
type: Optimization
title: "add_rms_norm_dynamic_quant：并行与 Tiling优化"
description: "add_rms_norm_dynamic_quant 算子的并行与 Tiling优化经验，关注并行度受限。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.tiling", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/docs/aclnnAddRmsNormDynamicQuant.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant_proto.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant_tiling.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_kernel/add_rms_norm_dynamic_quant.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_kernel/add_rms_norm_dynamic_quant_base.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_kernel/add_rms_norm_dynamic_quant_cut_d_kernel.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_kernel/add_rms_norm_dynamic_quant_helper.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_kernel/add_rms_norm_dynamic_quant_normal_kernel.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_kernel/add_rms_norm_dynamic_quant_single_row_kernel.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/op_kernel/reduce_common.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_dynamic_quant/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# add_rms_norm_dynamic_quant：并行与 Tiling优化

## 来源信息

- experience ID：`19ea61a805b59d01`
- 源 schema version：`v2`
- 算子：`add_rms_norm_dynamic_quant`
- source iteration：`round_018_refactor`
- 源 category：`norm`
- 计算分区：`fusedcomposite`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`tiling.core_allocation`
- canonical family：`performance.degradation.removal`
- 源记录时间：`2026-04-21T03:23:51`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Restore full core utilization in DoBlockTiling to maximize parallelism
~~~~

### Original summary

~~~~text
In Ascend C operators using DoBlockTiling for workload distribution, ensure core allocation logic doesn't artificially limit parallelism. The original code had logic that halved the useCore_ value after initial calculation, reducing active cores from 48 to 24 and doubling per-core workload. This optimization restores the natural core allocation formula: useCore_ = CeilDiv(numFirstDim_, firstDimPerCore_) without artificial reductions.
~~~~

### Original root mechanism

~~~~text
Halving core usage in DoBlockTiling artificially reduces parallelism, causing each core to process twice as much data and increasing latency by 93%
~~~~

### Original causal chain

~~~~text
structural change to performance.degradation.removal in add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant.cpp -> unexpected shift in memory_ub.ub_to_gm_kb -> 48.3% latency impact
~~~~

## 适用条件

- Ascend C operators using DoBlockTiling for workload distribution
- When block_dim is significantly lower than socCoreNums_
- When operator shows high memory bandwidth usage (GM↔UB transfers > 10GB/s)
- When active_block_count is less than half of available cores
- When aiv_vec_fops per core exceeds expected range for the workload

## 失效边界

- When core allocation already uses all available cores (useCore_ == socCoreNums_)
- When memory bandwidth is already saturated
- When operator has external synchronization constraints
- When tiling logic includes legitimate resource constraints (e.g., shared buffer limits)

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant.cpp

### 改动符号

- AddRmsNormDynamicQuantTilingHelper::DoBlockTiling

### 代码签名

- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`da0604bc329ee5e2c8c2ff8ad1a96c27d7afad5e72bd2f1495c724dfdbcb4835`

````diff
--- Before/op_host/add_rms_norm_dynamic_quant.cpp
+++ After/op_host/add_rms_norm_dynamic_quant.cpp
@@ -348,13 +348,6 @@
 {
     this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->socCoreNums_);
     this->useCore_ = CeilDiv(this->numFirstDim_, this->firstDimPerCore_);
-
-    // DEGRADATION: Artificially reduce core usage to decrease parallelism
-    // Halve the number of cores used (minimum 1) to increase workload per core
-    if (this->useCore_ > 1) {
-        this->useCore_ = (this->useCore_ + 1) / 2;  // Integer division rounding up
-    }
-
     this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->useCore_);
     this->firstDimPerCoreTail_ = this->numFirstDim_ - this->firstDimPerCore_ * (this->useCore_ - 1);
     OP_LOGI("DoBlockTiling",
````

## 观测记录

- bottleneck type：`parallelism.limitation`
- slow duration us：`2172.783447`
- fast duration us：`1123.982422`
- improvement ratio：`0.48269928899177583`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 11029.5 -> 22026.8 (+99.71%)
- arithmetic.aiv_vec_fops: 6.54124e+07 -> 1.30633e+08 (+99.71%)
- memory_ub.gm_to_ub_kb: 10992.1 -> 21904.1 (+99.27%)
- critical_path.aiv_vector0_time_us_mean: 1116.79 -> 2168.45 (+94.17%)
- latency.task_duration_us: 1123.98 -> 2172.78 (+93.31%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "memory_ub.ub_to_gm_kb",
  "arithmetic.aiv_vec_fops",
  "memory_ub.gm_to_ub_kb",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "cache_l2.aiv_read_hit_rate_pct",
  "parallelism.active_block_count",
  "parallelism.block_count",
  "aiv_utilization.mte2_ratio_mean",
  "cache_l2.aiv_total_hit_rate_pct"
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

- Check DoBlockTiling implementation in op_host/*.cpp files for artificial core reduction
- Look for patterns like 'useCore_ = (useCore_ + 1) / 2' or similar halving logic
- Verify that useCore_ calculation follows: useCore_ = CeilDiv(numFirstDim_, firstDimPerCore_)
- Profile with block_dim metric - it should match active_block_count
- Monitor memory_ub.gm_to_ub_kb and ub_to_gm_kb metrics - halving should reduce transfers proportionally

## 检索线索

- DoBlockTiling
- core_allocation
- useCore_
- parallelism
- block_dim
- socCoreNums_
- CeilDiv
- memory_ub
- gm_to_ub
- ub_to_gm
- aiv_vec_fops
- active_block_count
