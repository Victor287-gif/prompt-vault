---
type: Optimization
title: "add_rms_norm_dynamic_quant：并行与 Tiling优化"
description: "add_rms_norm_dynamic_quant 算子的并行与 Tiling优化经验，关注混合或尚未确定的瓶颈。"
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

- experience ID：`178d880577022a5a`
- 源 schema version：`v2`
- 算子：`add_rms_norm_dynamic_quant`
- source iteration：`round_006_refactor`
- 源 category：`norm`
- 计算分区：`fusedcomposite`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`tiling.core.usage`
- canonical family：`tiling.core.usage`
- 源记录时间：`2026-04-21T01:28:49`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removed artificial core usage degradation in block tiling
~~~~

### Original summary

~~~~text
On `norm/add_rms_norm_dynamic_quant`, a structural refactoring (`Removed artificial core usage degradation in block tiling`) that appeared semantically equivalent unexpectedly caused 48.5% performance change (1119.3 vs 2172.8 us, 48.5% difference). This reveals that `mixed_or_unknown` is sensitive to this code pattern. Removed code that artificially halves the number of cores used, restoring original tiling behavior
~~~~

### Original root mechanism

~~~~text
Removed code that artificially halves the number of cores used, restoring original tiling behavior
~~~~

### Original causal chain

~~~~text
structural change to tiling.core.usage in add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant.cpp -> unexpected shift in memory_ub.ub_to_gm_kb -> 48.5% latency impact
~~~~

## 适用条件

- The target operator's profiling shows bottleneck in: memory_ub, arithmetic, critical_path
- The kernel code uses related APIs/patterns: coreNum, core_launch
- The operator has a `mixed_or_unknown` bottleneck profile
- You are considering a similar structural refactoring and want to verify it won't affect performance

## 失效边界

- The operator's bottleneck is unrelated to `mixed_or_unknown`
- The `tiling.core.usage` pattern is not present in the target code

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

- 源 diff SHA-256：`27d00a206b88f8152e5a46900facaacfbbaaf9e83a25887fb5cb9f817ae8004d`

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

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2172.783447`
- fast duration us：`1119.302368`
- improvement ratio：`0.48485323305208333`
- machine confidence：`0.5`

### Profile evidence

- memory_ub.ub_to_gm_kb: 11029.5 -> 22026.8 (+99.71%)
- arithmetic.aiv_vec_fops: 6.54124e+07 -> 1.30633e+08 (+99.71%)
- memory_ub.gm_to_ub_kb: 10992.1 -> 21904.1 (+99.27%)
- critical_path.aiv_vector0_time_us_mean: 1112.81 -> 2168.45 (+94.86%)
- latency.task_duration_us: 1119.3 -> 2172.78 (+94.12%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 11029.5,
      "after": 22026.8,
      "delta": 10997.3,
      "rel": 0.9970805566888797
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 65412416.0,
      "after": 130633440.0,
      "delta": 65221024.0,
      "rel": 0.9970740722984456
    }
  },
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 10992.1,
      "after": 21904.1,
      "delta": 10911.999999999998,
      "rel": 0.9927129483902073
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1112.808,
      "after": 2168.447,
      "delta": 1055.6390000000001,
      "rel": 0.9486263578263278
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1119.302,
      "after": 2172.783,
      "delta": 1053.481,
      "rel": 0.9411946016356623
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 0.43,
      "after": 0.21,
      "delta": -0.22,
      "rel": -0.5116279069767442
    }
  },
  {
    "parallelism.active_block_count": {
      "before": 48.0,
      "after": 24.0,
      "delta": -24.0,
      "rel": -0.5
    }
  },
  {
    "parallelism.block_count": {
      "before": 48.0,
      "after": 24.0,
      "delta": -24.0,
      "rel": -0.5
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.173903,
      "after": 0.138066,
      "delta": -0.03583700000000001,
      "rel": -0.2060746508110844
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 0.96,
      "after": 0.78,
      "delta": -0.17999999999999994,
      "rel": -0.18749999999999994
    }
  }
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

- Before applying a similar structural change, profile the `tiling.core.usage` code path in `add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant.cpp` (key symbols: AddRmsNormDynamicQuantTilingHelper::DoBlockTiling)
- After the change, verify that `memory_ub.ub_to_gm_kb` moves in the expected direction
- If metrics regress, the change is NOT performance-neutral — revert it

## 检索线索

- After
- Artificially
- Before
- CeilDiv
- DEGRADATION
- DoBlockTiling
- Halve
- Integer
- OP_LOGI
- arithmetic
- coreNum
- core_launch
- critical_path
- firstDimPerCoreTail_
- firstDimPerCore_
- latency
- memory_ub
- mixed_or_unknown
- numFirstDim_
- socCoreNums_
- tiling.core.usage
- useCore_
