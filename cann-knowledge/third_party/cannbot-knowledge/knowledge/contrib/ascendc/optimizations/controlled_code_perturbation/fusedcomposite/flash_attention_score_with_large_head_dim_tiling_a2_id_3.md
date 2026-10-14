---
type: Optimization
title: "flash_attention_score_with_large_head_dim：Tiling优化"
description: "flash_attention_score_with_large_head_dim 算子的Tiling优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.tiling", "topic.programming_model", "topic.programming_model.execution_unit.scalar"]
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
# flash_attention_score_with_large_head_dim：Tiling优化

## 来源信息

- experience ID：`f3dd58c74a9b4644`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_003_degrade`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.loop.iteration.increase`
- canonical family：`tiling.loop.iteration.increase`
- 源记录时间：`2026-04-20T19:33:50`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce nRatio to increase loop iterations
~~~~

### Original summary

~~~~text
For flash_attention_score_with_large_head_dim operators using the FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1 template, reducing nRatio from 8 to 4 improves performance by 11.1%. This optimization reduces scalar overhead by increasing loop iterations, which better utilizes vector units and reduces pipeline stalls. Apply this pattern when the operator shows high scalar_ratio_mean and aiv_vec_wait_ratio_mean metrics indicating scalar-bound execution.
~~~~

### Original root mechanism

~~~~text
Reducing nRatio from 8 to 4 decreases the effective tile size for s2 dimension processing and increases the number of kernel loop iterations, which improves performance by reducing scalar overhead and increasing parallelism
~~~~

### Original causal chain

~~~~text
apply tiling.loop.iteration.increase in op_host/flash_attention_score_with_large_head_dim.cpp -> improve arithmetic.aiv_vec_misc_ratio_mean -> 11.1% latency improvement
~~~~

## 适用条件

- FlashAttentionScoreWithLargeHeadDim operators using the FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1 template
- When aiv_utilization.scalar_ratio_mean > 0.1 indicating scalar-bound execution
- When pipeline_stalls.aiv_vec_wait_ratio_mean > 0.05 indicating vector unit underutilization
- When s2 dimension size is large enough to benefit from increased loop iterations

## 失效边界

- When nRatio is already at optimal value (4 or lower)
- When s2 dimension is small and increasing loop iterations would add overhead
- When memory bandwidth is already saturated (memory_gm metrics show >80% utilization)
- When aic_cube_wait_ratio_mean is already low (<0.3)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/flash_attention_score_with_large_head_dim.cpp

### 改动符号

- FlashAttentionScoreWithLargeHeadDimTiling::nRatio

### 代码签名

- template
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`d9a5bf5dba0ee544568e342c7056d8458ebe878040eeffbabf9ef2ef000652ea`

````diff
--- Before/op_host/flash_attention_score_with_large_head_dim.cpp
+++ After/op_host/flash_attention_score_with_large_head_dim.cpp
@@ -201,7 +201,7 @@
     int64_t s1BasicBlock = std::numeric_limits<int64_t>::max();
     int64_t s2BasicBlock = std::numeric_limits<int64_t>::max();
     int64_t dBasicBlock = std::numeric_limits<int64_t>::max();
-    int64_t nRatio = 8L;
+    int64_t nRatio = 4L;

     const char *templateName = "FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1";
     int64_t s2sizeLimitMin = 1024;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1571.711426`
- fast duration us：`1396.827881`
- improvement ratio：`0.11126950030838557`
- machine confidence：`0.9`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.004759 -> 0.011112 (+133.49%)
- memory_ub.ub_write_bw_scalar_gbps: 0.0072 -> 0.0128 (+77.78%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.009757 -> 0.017247 (+76.77%)
- aiv_utilization.scalar_ratio_mean: 0.071278 -> 0.123172 (+72.81%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.052881 -> 0.088291 (+66.96%)

### Affected metric groups

- arithmetic
- memory_ub
- pipeline_stalls
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.004759,
      "after": 0.011112,
      "delta": 0.006353,
      "rel": 1.33494431603278
    }
  },
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.0072,
      "after": 0.0128,
      "delta": 0.005600000000000001,
      "rel": 0.7777777777777779
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.009757,
      "after": 0.017247,
      "delta": 0.007489999999999998,
      "rel": 0.7676539920057392
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.071278,
      "after": 0.123172,
      "delta": 0.05189400000000001,
      "rel": 0.7280507309408234
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.052881,
      "after": 0.088291,
      "delta": 0.03541,
      "rel": 0.6696166865225696
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

- 1. Check if operator uses FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1 template in op_host/flash_attention_score_with_large_head_dim.cpp
- 2. Locate the nRatio parameter in FlashAttentionScoreWithLargeHeadDimTiling class (line 204 in the diff)
- 3. Reduce nRatio from 8 to 4 to increase loop iterations for s2 dimension processing
- 4. Verify performance improvement by monitoring aiv_utilization.scalar_ratio_mean (should decrease)
- 5. Monitor pipeline_stalls.aiv_vec_wait_ratio_mean (should decrease)
- 6. Check that aiv_vec_fops remains stable or improves

## 检索线索

- flash_attention_score_with_large_head_dim
- nRatio
- tiling
- loop_iterations
- scalar_overhead
- vector_wait
- FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1
- s2_dimension
- tile_size
