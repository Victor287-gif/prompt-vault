---
type: Optimization
title: "flash_attention_score_with_large_head_dim：Tiling优化"
description: "flash_attention_score_with_large_head_dim 算子的Tiling优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.tiling", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.scalar"]
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

- experience ID：`06a80821789d47b7`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.block.size.reduction`
- canonical family：`tiling.block.size.reduction`
- 源记录时间：`2026-04-20T19:12:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce s2 dimension basic block size
~~~~

### Original summary

~~~~text
Applying smaller tile sizes for the s2 dimension (reducing from 128 to 64) improves performance by reducing scalar instruction overhead and optimizing memory bandwidth usage. This optimization is particularly effective for operators with large s2 dimensions where scalar operations become a bottleneck. The smaller tile size reduces the amount of data processed per iteration, decreasing scalar instruction counts and improving vector unit utilization.
~~~~

### Original root mechanism

~~~~text
Reducing s2BasicBlock maximum from 128 to 64 in MatchTemplate function forces smaller tile sizes for s2 dimension processing, which reduces scalar overhead and improves memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Reducing s2BasicBlock from 128 to 64 -> decreased scalar ratio from 0.319942 to 0.269379 (-15.8%) -> reduced aiv_vec_wait_ratio from 0.052881 to 0.030132 (-43.1%) -> improved task duration from 1396.828μs to 1239.885μs (-11.2%)
~~~~

## 适用条件

- FlashAttentionScore operators with large head dimensions where s2 dimension is large
- Operators showing high scalar_ratio_mean (>0.25) in aiv_utilization metrics
- When aiv_vec_wait_ratio_mean is elevated (>0.04) indicating vector unit stalls
- When memory bandwidth to UB shows scalar operations dominating (ub_write_bw_scalar_gbps > 0.005)
- In MatchTemplate functions where s2BasicBlock is set to maximum value (128)

## 失效边界

- Operators with small s2 dimensions where tile size reduction would increase iteration count excessively
- When memory bandwidth is already saturated and further reductions would increase overhead
- When vector unit utilization is already optimal (<0.05 aiv_vec_wait_ratio_mean)
- For operators where s2 dimension is not a major computational axis

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/flash_attention_score_with_large_head_dim.cpp

### 改动符号

- FlashAttentionScoreWithLargeHeadDimTiling::MatchTemplate

### 代码签名

- template
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`e4cdf1c8370a367b4fdea6437cb48acc589b751ddeaf5c2ac389dd7f34a48b93`

````diff
--- Before/op_host/flash_attention_score_with_large_head_dim.cpp
+++ After/op_host/flash_attention_score_with_large_head_dim.cpp
@@ -315,7 +315,7 @@
     if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) > aivNum) {
         s1BasicBlock = std::min(128L, alignedS1);
     }
-    s2BasicBlock = std::min(128L, alignedS2);
+    s2BasicBlock = std::min(64L, alignedS2);
     dBasicBlock = std::min(128L, alignedD);
     apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * s1BasicBlock * s2BasicBlock * sizeof(float);
     LOG_PRINT("[%s]final basic block: [%ld, %ld, %ld].\n", templateName, s1BasicBlock,
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1396.827881`
- fast duration us：`1239.884766`
- improvement ratio：`0.11235680296390077`
- machine confidence：`0.9`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.002424 -> 0.004759 (+96.33%)
- memory_ub.ub_write_bw_scalar_gbps: 0.0041 -> 0.0072 (+75.61%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.030132 -> 0.052881 (+75.50%)
- aiv_utilization.mte3_ratio_mean: 0.033583 -> 0.05288 (+57.46%)
- aiv_utilization.scalar_ratio_mean: 0.047731 -> 0.071278 (+49.33%)

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
      "before": 0.002424,
      "after": 0.004759,
      "delta": 0.0023350000000000003,
      "rel": 0.9632838283828384
    }
  },
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.0041,
      "after": 0.0072,
      "delta": 0.0030999999999999995,
      "rel": 0.7560975609756095
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.030132,
      "after": 0.052881,
      "delta": 0.022749,
      "rel": 0.7549780963759458
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.033583,
      "after": 0.05288,
      "delta": 0.019297,
      "rel": 0.5746061995652563
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.047731,
      "after": 0.071278,
      "delta": 0.02354699999999999,
      "rel": 0.49332718778152546
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

- In FlashAttentionScoreWithLargeHeadDimTiling::MatchTemplate function, reduce s2BasicBlock from std::min(128L, alignedS2) to std::min(64L, alignedS2)
- Verify the operator has large s2 dimension (check alignedS2 value in profiling)
- Monitor aiv_utilization.scalar_ratio_mean reduction after optimization
- Check aiv_vec_wait_ratio_mean improvement as indicator of reduced vector unit stalls
- Validate that ub_write_bw_scalar_gbps decreases while overall throughput improves

## 检索线索

- flash_attention_score
- large_head_dim
- tiling
- block_size
- s2_dimension
- scalar_overhead
- vector_stalls
- MatchTemplate
- basic_block
- memory_bandwidth
