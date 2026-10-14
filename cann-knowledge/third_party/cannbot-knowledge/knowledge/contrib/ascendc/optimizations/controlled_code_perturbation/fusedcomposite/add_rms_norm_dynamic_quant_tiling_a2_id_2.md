---
type: Optimization
title: "add_rms_norm_dynamic_quant：并行与 Tiling优化"
description: "add_rms_norm_dynamic_quant 算子的并行与 Tiling优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.tiling", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
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

- experience ID：`bf3390ff049af1aa`
- 源 schema version：`v2`
- 算子：`add_rms_norm_dynamic_quant`
- source iteration：`round_001_degrade`
- 源 category：`norm`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.core.reduction`
- canonical family：`tiling.core.reduction`
- 源记录时间：`2026-04-20T23:57:01`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increasing core count for vector operations to improve parallel efficiency
~~~~

### Original summary

~~~~text
For Ascend C vector operators like AddRmsNormDynamicQuant, increasing the number of cores from 24 to 48 reduces per-core workload and improves parallel efficiency. This optimization reduces task duration from 2172.78μs to 1119.98μs (2x speedup) by distributing computation across more cores, reducing vector floating-point operations per core from 1.3e8 to 6.54e7 and improving memory bandwidth utilization.
~~~~

### Original root mechanism

~~~~text
Doubling the number of cores used for computation reduces workload per core and improves parallel efficiency by better utilizing available hardware resources.
~~~~

### Original causal chain

~~~~text
Increasing block_dim from 24 to 48 cores → reduces per-core FLOPS by 50% (1.3e8 to 6.54e7) → reduces vector computation time from 2168.45μs to 1113.14μs → reduces overall task duration from 2172.78μs to 1119.98μs
~~~~

## 适用条件

- Vector operators with high arithmetic intensity (aiv_vec_fops > 50M)
- Operators using tiling with CeilDiv(numFirstDim_, socCoreNums_) pattern
- When block_time_cv is near 0.0 indicating balanced workload distribution
- When aiv_vec_wait_ratio_mean > 0.3 indicates pipeline stalls
- When parallelism.block_count < socCoreNums_ (underutilizing cores)

## 失效边界

- Memory-bound operators with aiv_main_mem_read_bw_gbps > 50 Gbps
- When icache_miss_rate_mean > 0.1 indicates instruction cache thrashing
- When block_time_cv > 0.2 indicates imbalanced workload distribution
- When aiv_vec_total_cflt_ratio_mean > 0.5 indicates high conflict rates

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/add_rms_norm_dynamic_quant.cpp

### 改动符号

- AddRmsNormDynamicQuantTilingHelper::DoBlockTiling

### 代码签名

- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`8f4126d11f648a57766843fff7f1761d2f56779486f436635b13eceaaa5778c1`

````diff
--- Before/op_host/add_rms_norm_dynamic_quant.cpp
+++ After/op_host/add_rms_norm_dynamic_quant.cpp
@@ -348,6 +348,13 @@
 {
     this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->socCoreNums_);
     this->useCore_ = CeilDiv(this->numFirstDim_, this->firstDimPerCore_);
+
+    // DEGRADATION: Artificially reduce core usage to decrease parallelism
+    // Halve the number of cores used (minimum 1) to increase workload per core
+    if (this->useCore_ > 1) {
+        this->useCore_ = (this->useCore_ + 1) / 2;  // Integer division rounding up
+    }
+
     this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->useCore_);
     this->firstDimPerCoreTail_ = this->numFirstDim_ - this->firstDimPerCore_ * (this->useCore_ - 1);
     OP_LOGI("DoBlockTiling",
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2172.783447`
- fast duration us：`1119.982422`
- improvement ratio：`0.48454024557929165`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_to_gm_kb: 11029.5 -> 22026.8 (+99.71%)
- arithmetic.aiv_vec_fops: 6.54124e+07 -> 1.30633e+08 (+99.71%)
- memory_ub.gm_to_ub_kb: 10992.1 -> 21904.1 (+99.27%)
- critical_path.aiv_vector0_time_us_mean: 1113.14 -> 2168.45 (+94.80%)
- latency.task_duration_us: 1119.98 -> 2172.78 (+94.00%)

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
  "latency.task_duration_us"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 1.0,
  "overall_deviation": 0.4,
  "metrics_checked": 1,
  "metrics_wrong": 0,
  "performance_metrics_checked": 1,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- In AddRmsNormDynamicQuantTilingHelper::DoBlockTiling(), increase useCore_ to maximize core utilization
- Replace artificial core reduction with full core utilization: remove code that halves useCore_
- Set useCore_ = socCoreNums_ when numFirstDim_ > socCoreNums_ * firstDimPerCore_
- Monitor aiv_vec_fops and memory_ub metrics to verify workload distribution improves
- Check that block_count equals socCoreNums_ for maximum parallelism

## 检索线索

- core_count
- parallelism
- tiling
- CeilDiv
- vector_operator
- AddRmsNormDynamicQuant
- block_dim
- useCore_
- numFirstDim_
- socCoreNums_
