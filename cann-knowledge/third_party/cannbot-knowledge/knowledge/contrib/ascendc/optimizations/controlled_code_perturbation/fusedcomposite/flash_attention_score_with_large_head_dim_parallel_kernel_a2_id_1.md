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

- experience ID：`866d210bf133a318`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_019_degrade`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`task.block.dimension.optimization`
- canonical family：`task.block.dimension.increase`
- 源记录时间：`2026-04-20T23:27:48`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Optimize block dimension calculation for better parallelism efficiency
~~~~

### Original summary

~~~~text
When calculating block dimensions for Ascend C operators with mixed AIC/AIV cores, use (sliceNum + ration) / ration for ceiling division instead of (sliceNum + ration - 1) / ration. This reduces block dimension when sliceNum is not perfectly divisible by ration, decreasing scalar overhead and improving parallelism efficiency. Apply this optimization in CalcTschBlockDim functions where ration = aivCoreNum / aicCoreNum.
~~~~

### Original root mechanism

~~~~text
Using (sliceNum + ration) / ration instead of (sliceNum + ration - 1) / ration for ceiling division reduces block dimension when sliceNum is not perfectly divisible by ration, improving parallelism efficiency.
~~~~

### Original causal chain

~~~~text
Changed ceiling division formula from (sliceNum + ration - 1) / ration to (sliceNum + ration) / ration -> Reduced block dimension from 13 to 12 and mix_block_dim from 26 to 24 -> Decreased scalar overhead (scalar_ratio_mean reduced from 0.471162 to 0.526464) -> Improved parallelism efficiency (block_time_cv reduced from 0.9213 to 0.8406) -> Reduced task duration from 2938.159us to 2780.696us (5.36% improvement).
~~~~

## 适用条件

- When implementing CalcTschBlockDim or similar block dimension calculation functions in Ascend C operators
- When using mixed AIC/AIV core configurations with ration = aivCoreNum / aicCoreNum
- When sliceNum is not perfectly divisible by ration
- When optimizing for parallelism efficiency in scalar_overhead_bound scenarios

## 失效边界

- When sliceNum is perfectly divisible by ration (both formulas produce same result)
- When block dimension calculation uses floor division instead of ceiling division
- When the operator is not bound by scalar overhead

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/flash_attention_score_with_large_head_dim.cpp

### 改动符号

- CalcTschBlockDim

### 代码签名

- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`d9603acf76f598ca28d6ea5966f5cb33c6e406900e78a72765e10da0d8b85c6f`

````diff
--- Before/op_host/flash_attention_score_with_large_head_dim.cpp
+++ After/op_host/flash_attention_score_with_large_head_dim.cpp
@@ -77,7 +77,7 @@
         return sliceNum;
     }
     ration = aivCoreNum / aicCoreNum;
-    return (sliceNum + (ration - 1)) / ration;
+    return (sliceNum + ration) / ration;
 }

 } // namespace optiling
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2938.158691`
- fast duration us：`2780.695557`
- improvement ratio：`0.05359245383250815`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.icache_miss_rate_mean: 0.019447 -> 0.039956 (+105.46%)
- critical_path.aic_aiv_gap_pct: 0.42 -> 0.35 (-16.67%)
- memory_gm.aiv_main_mem_write_bw_gbps: 4.62 -> 4.03 (-12.77%)
- memory_gm.aic_main_mem_write_bw_gbps: 12.23 -> 10.68 (-12.67%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.01931 -> 0.016866 (-12.66%)

### Affected metric groups

- aiv_utilization
- critical_path
- memory_gm
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 0.019447,
      "after": 0.039956,
      "delta": 0.020509,
      "rel": 1.0546099655473853
    }
  },
  {
    "critical_path.aic_aiv_gap_pct": {
      "before": 0.42,
      "after": 0.35,
      "delta": -0.07,
      "rel": -0.16666666666666669
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 4.62,
      "after": 4.03,
      "delta": -0.5899999999999999,
      "rel": -0.12770562770562768
    }
  },
  {
    "memory_gm.aic_main_mem_write_bw_gbps": {
      "before": 12.23,
      "after": 10.68,
      "delta": -1.5500000000000007,
      "rel": -0.12673753066230586
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.01931,
      "after": 0.016866,
      "delta": -0.0024440000000000017,
      "rel": -0.12656654583117563
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.132545,
  "overall_deviation": 0.053018,
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

- In op_host/flash_attention_score_with_large_head_dim.cpp, locate the CalcTschBlockDim function
- Change the ceiling division formula from (sliceNum + ration - 1) / ration to (sliceNum + ration) / ration
- Verify that ration = aivCoreNum / aicCoreNum is properly calculated
- Test with various sliceNum values to ensure correct block dimension calculation
- Profile the operator to confirm reduced scalar overhead and improved parallelism

## 检索线索

- CalcTschBlockDim
- block dimension
- ceiling division
- ration
- sliceNum
- aicCoreNum
- aivCoreNum
- scalar_overhead_bound
- parallelism efficiency
- mixed AIC AIV cores
