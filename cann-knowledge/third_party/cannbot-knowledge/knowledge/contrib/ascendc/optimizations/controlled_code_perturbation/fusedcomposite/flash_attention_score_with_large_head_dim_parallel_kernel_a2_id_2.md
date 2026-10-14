---
type: Optimization
title: "flash_attention_score_with_large_head_dim：并行切核优化"
description: "flash_attention_score_with_large_head_dim 算子的并行切核优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/docs/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/.keep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/framework/tf_plugin/tensorflow_flash_attention_score_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_host/flash_attention_score_with_large_head_dim.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_host/flash_attention_score_with_large_head_dim_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_host/tiling_base.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_host/tiling_type.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_kernel/flash_attention_score_with_large_head_dim.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/FlashAttentionScore_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/test_flash_attention_score.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# flash_attention_score_with_large_head_dim：并行切核优化

## 来源信息

- experience ID：`9a27ee7974bdeb9d`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_017_degrade`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.split.factor.increase`
- canonical family：`parallelism.split.factor.increase`
- 源记录时间：`2026-04-20T22:55:05`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase split factor per core by adding core count to numerator
~~~~

### Original summary

~~~~text
In the FlashAttentionScoreWithLargeHeadDim operator, modifying the CeilDivision calculation in SetMultiCoreParams to add actualUsedAivNum to totalSize before division increases the splitFactorSize by approximately 1. This optimization increases work per core but reduces parallelism efficiency, leading to higher latency. Apply this pattern when you need to increase work granularity per core at the cost of reduced parallelism.
~~~~

### Original root mechanism

~~~~text
Adding core count to numerator in CeilDivision increases splitFactorSize by approximately 1, which increases work per core and reduces parallelism efficiency.
~~~~

### Original causal chain

~~~~text
Modified CeilDivision(totalSize, actualUsedAivNum) to CeilDivision(totalSize + actualUsedAivNum, actualUsedAivNum) -> Increased splitFactorSize by ~1 -> Increased work per core by 33.3% (aic_cube_fops: 17.89M to 23.86M) -> Reduced parallelism efficiency (block_time_cv increased 70.9%) -> Increased latency by 20.4% (2212μs to 2781μs)
~~~~

## 适用条件

- Working with Ascend C operators that use SetMultiCoreParams and CeilDivision for parallelism
- When totalSize is divisible by actualUsedAivNum, adding core count to numerator increases splitFactorSize by 1
- When you need to increase work granularity per core at the expense of parallelism
- In FlashAttentionScoreWithLargeHeadDimTiling::SetMultiCoreParams or similar tiling functions

## 失效边界

- When parallelism efficiency is critical for performance
- When block_time_cv (coefficient of variation) is already high (>0.5)
- When aic_cube_fops and aiv_vec_fops are already near memory bandwidth limits
- When icache miss rates are already elevated (>0.01)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/flash_attention_score_with_large_head_dim.cpp

### 改动符号

- FlashAttentionScoreWithLargeHeadDimTiling::SetMultiCoreParams

### 原始 diff

- 源 diff SHA-256：`41c1fdfcc5f8ad2439b70f0cb7517056412f0fc776664b9443189a8d813c6f50`

````diff
--- Before/op_host/flash_attention_score_with_large_head_dim.cpp
+++ After/op_host/flash_attention_score_with_large_head_dim.cpp
@@ -338,7 +338,7 @@
     int64_t totalSize = bSize * n2Size * gSize * tilingData.coreParams.get_s1OuterSize();
     actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum / 2));
     multiCoreParams.set_totalSize(totalSize);
-    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
+    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize + actualUsedAivNum, actualUsedAivNum));
 }

 ge::graphStatus FlashAttentionScoreWithLargeHeadDimTiling::DoOpTiling()
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`2780.695557`
- fast duration us：`2212.104248`
- improvement ratio：`0.20447808735071818`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aic_aiv_gap_pct: 4.36 -> 0.42 (-90.37%)
- aiv_utilization.icache_miss_rate_mean: 0.01061 -> 0.019447 (+83.29%)
- parallelism.block_time_cv: 0.4918 -> 0.8406 (+70.92%)
- parallelism.block_time_std_us: 807.762 -> 1346.81 (+66.73%)
- arithmetic.aic_cube_fops: 1.78913e+07 -> 2.38551e+07 (+33.33%)

### Affected metric groups

- critical_path
- aiv_utilization
- parallelism
- arithmetic

### Changed metrics top

~~~~json
[
  {
    "critical_path.aic_aiv_gap_pct": {
      "before": 4.36,
      "after": 0.42,
      "delta": -3.9400000000000004,
      "rel": -0.9036697247706422
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 0.01061,
      "after": 0.019447,
      "delta": 0.008837,
      "rel": 0.8328934967012253
    }
  },
  {
    "parallelism.block_time_cv": {
      "before": 0.4918,
      "after": 0.8406,
      "delta": 0.3488,
      "rel": 0.7092313948759659
    }
  },
  {
    "parallelism.block_time_std_us": {
      "before": 807.762,
      "after": 1346.811,
      "delta": 539.049,
      "rel": 0.6673364184004694
    }
  },
  {
    "arithmetic.aic_cube_fops": {
      "before": 17891328.0,
      "after": 23855104.0,
      "delta": 5963776.0,
      "rel": 0.3333333333333333
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

- Locate CeilDivision calls in SetMultiCoreParams functions within operator tiling code
- Identify totalSize and actualUsedAivNum variables used in the division
- Change from CeilDivision(totalSize, actualUsedAivNum) to CeilDivision(totalSize + actualUsedAivNum, actualUsedAivNum)
- Monitor impact on parallelism.block_time_cv and arithmetic.aic_cube_fops metrics
- Validate that splitFactorSize increases by approximately 1 and work per core increases proportionally

## 检索线索

- CeilDivision
- SetMultiCoreParams
- splitFactorSize
- parallelism
- work_per_core
- FlashAttentionScoreWithLargeHeadDimTiling
- actualUsedAivNum
- totalSize
- tiling
- Ascend C
