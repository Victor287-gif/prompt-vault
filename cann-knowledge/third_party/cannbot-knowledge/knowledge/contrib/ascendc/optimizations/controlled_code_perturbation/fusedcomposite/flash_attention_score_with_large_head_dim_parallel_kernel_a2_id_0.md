---
type: Optimization
title: "flash_attention_score_with_large_head_dim：并行切核优化"
description: "flash_attention_score_with_large_head_dim 算子的并行切核优化经验，关注标量开销。"
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

- experience ID：`55cbfeb4c80fc809`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_013_degrade`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.core.reduction`
- canonical family：`parallelism.core.reduction`
- 源记录时间：`2026-04-20T21:45:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Halve AIV core utilization to increase parallelism and reduce per-core workload
~~~~

### Original summary

~~~~text
When AIV core utilization is too high for flash_attention_score_with_large_head_dim operations, reducing the maximum core count by half can improve performance by increasing parallelism and reducing per-core workload. This optimization applies to operators where scalar overhead dominates and increasing block count reduces block time variance (CV decreased from 0.8341 to 0.4889).
~~~~

### Original root mechanism

~~~~text
Halving AIV core utilization reduces per-core workload and increases parallelism, improving overall throughput by better balancing compute resources.
~~~~

### Original causal chain

~~~~text
Halved actualUsedAivNum calculation from aivNum to aivNum/2 → Increased block_count from 12 to 24 → Reduced block_time_mean_us from 1564.653μs to 888.509μs → Improved task_duration_us from 2042.401μs to 1571.711μs (23% faster)
~~~~

## 适用条件

- FlashAttentionScoreWithLargeHeadDim operators show high aic_cube_wait_ratio_mean (>0.75) and aic_mte1_wait_ratio_mean (>0.75)
- Block time coefficient of variation (CV) is high (>0.8) indicating uneven workload distribution
- AIV utilization metrics (vec_ratio_mean, scalar_ratio_mean) are below 0.25 while AIC utilization is above 0.3
- Operator has scalar_overhead_bound bottleneck type in profiling data

## 失效边界

- Memory bandwidth is already saturated (L0/L1 bandwidth usage > 80%)
- AIV utilization metrics are already high (>0.5)
- Block time CV is already low (<0.3) indicating good load balancing
- Operator shows memory-bound characteristics with high cache miss rates

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/flash_attention_score_with_large_head_dim.cpp

### 改动符号

- FlashAttentionScoreWithLargeHeadDimTiling::SetMultiCoreParams

### 原始 diff

- 源 diff SHA-256：`12f85ed3b6bb5da66ec793363e28483071542f636fc15d312ca2337ee1e4eb32`

````diff
--- Before/op_host/flash_attention_score_with_large_head_dim.cpp
+++ After/op_host/flash_attention_score_with_large_head_dim.cpp
@@ -336,7 +336,7 @@
 {
     auto &multiCoreParams = tilingData.multiCoreParams;
     int64_t totalSize = bSize * n2Size * gSize * tilingData.coreParams.get_s1OuterSize();
-    actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
+    actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum / 2));
     multiCoreParams.set_totalSize(totalSize);
     multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
 }
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2042.400879`
- fast duration us：`1571.711426`
- improvement ratio：`0.23045889660528293`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aic_aiv_gap_pct: 1.46 -> 4.44 (+204.11%)
- critical_path.aiv_vector0_time_us_mean: 875.522 -> 1558.95 (+78.06%)
- critical_path.aic_time_us_mean: 888.509 -> 1564.65 (+76.10%)
- parallelism.block_time_mean_us: 888.509 -> 1564.65 (+76.10%)
- critical_path.aiv_vector1_time_us_mean: 886.519 -> 1495.14 (+68.65%)

### Affected metric groups

- critical_path
- parallelism
- arithmetic
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "critical_path.aic_aiv_gap_pct": {
      "before": 1.46,
      "after": 4.44,
      "delta": 2.9800000000000004,
      "rel": 2.041095890410959
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 875.522,
      "after": 1558.949,
      "delta": 683.427,
      "rel": 0.7805937486436663
    }
  },
  {
    "critical_path.aic_time_us_mean": {
      "before": 888.509,
      "after": 1564.653,
      "delta": 676.144,
      "rel": 0.7609872269161032
    }
  },
  {
    "parallelism.block_time_mean_us": {
      "before": 888.509,
      "after": 1564.653,
      "delta": 676.144,
      "rel": 0.7609872269161032
    }
  },
  {
    "critical_path.aiv_vector1_time_us_mean": {
      "before": 886.519,
      "after": 1495.142,
      "delta": 608.623,
      "rel": 0.6865312531372707
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

- Locate SetMultiCoreParams function in flash_attention_score_with_large_head_dim.cpp
- Find calculation of actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum))
- Modify to actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum / 2))
- Verify block_count doubles and block_time_mean_us reduces by ~40-50%
- Check that aic_aiv_gap_pct increases (indicating better AIC-AIV overlap)
- Monitor aic_cube_wait_ratio_mean and aic_mte1_wait_ratio_mean decrease

## 检索线索

- flash_attention_score_with_large_head_dim
- AIV core utilization
- parallelism reduction
- scalar_overhead_bound
- SetMultiCoreParams
- actualUsedAivNum
- block_count
- block_time_cv
- aic_cube_wait_ratio
- aic_mte1_wait_ratio
