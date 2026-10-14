---
type: Optimization
title: "triu：内存与 Tiling优化"
description: "triu 算子的内存与 Tiling优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.tiling"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/docs/Triu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/tf_plugin/tensorflow_triu_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_host/triu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_host/triu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_kernel/triu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/Triu_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/test_triu.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# triu：内存与 Tiling优化

## 来源信息

- experience ID：`3b7ca62efb0074a6`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_008_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.buffer.overallocation`
- canonical family：`tiling.buffer.overallocation`
- 源记录时间：`2026-04-21T00:26:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase buffer sharing count in fast path tiling
~~~~

### Original summary

~~~~text
Increasing ub_Sharing_Num in fast path tiling calculations reduces the effective Unified Buffer (UB) size available per buffer, forcing smaller tile sizes and increasing loop iterations. This optimization should be applied when tiling calculations for vector operations show high scalar unit utilization and memory bandwidth underutilization. The technique improves performance by better balancing compute and memory operations in the triu operator.
~~~~

### Original root mechanism

~~~~text
Increased ub_Sharing_Num from 4 to 8 in fast path tiling calculation, reducing effective UB size per buffer and forcing smaller tile sizes with more loop iterations.
~~~~

### Original causal chain

~~~~text
Increasing ub_Sharing_Num from 4 to 8 → reduces effective UB size per buffer → forces smaller tile sizes → increases loop iterations → reduces scalar unit utilization from 76.4% to 47.9% → decreases MTE2/3 wait ratios → improves overall latency by 8%
~~~~

## 适用条件

- When implementing fast path tiling for vector operations with aligned processing conditions
- When column_Length % (computeBatchSize / type_Size) == 0 and diag_Val <= 0
- When scalar unit utilization exceeds 75% indicating compute-bound operations
- When MTE2/3 wait ratios exceed 97% showing memory transfer bottlenecks
- When UB bandwidth usage rates are below 30% indicating underutilized memory bandwidth

## 失效边界

- When UB bandwidth usage rates already exceed 80%
- When scalar unit utilization is already below 50%
- When memory bandwidth is the primary bottleneck
- When working with unaligned data where column_Length % (computeBatchSize / type_Size) != 0
- When diag_Val > 0 indicating different computation patterns

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- triu/op_host/triu.cpp

### 改动符号

- optiling::setTilingInfo

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`e22e85437b427a9af033abc5a63b759b3fcbe1e1070bacde9ef985c87ef83e0e`

````diff
--- Before/op_host/triu.cpp
+++ After/op_host/triu.cpp
@@ -74,8 +74,8 @@
         if (diag_Val <= 0 && column_Length % (computeBatchSize / type_Size) == 0){
             // A faster method for aligned processing only
             key_value = keyFour;
-            // Double buffer setting
-            ub_Sharing_Num = bufferFour;
+            // Double buffer setting - degraded to use more buffers than actually allocated
+            ub_Sharing_Num = bufferFour * 2;
             // The result would not be the expected
             if (column_Length == 0){
                 column_Length = minNum;
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2361.387207`
- fast duration us：`2173.043457`
- improvement ratio：`0.07975979095748527`
- machine confidence：`0.85`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.478541 -> 0.763916 (+59.63%)
- aiv_utilization.icache_miss_rate_mean: 0.000137 -> 8.2e-05 (-40.15%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.014795 -> 0.017609 (+19.02%)
- critical_path.aiv_vector0_time_us_mean: 2172.43 -> 2360.81 (+8.67%)
- latency.task_duration_us: 2173.04 -> 2361.39 (+8.67%)

### Affected metric groups

- aiv_utilization
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.478541,
      "after": 0.763916,
      "delta": 0.28537500000000005,
      "rel": 0.5963438869396772
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 0.000137,
      "after": 8.2e-05,
      "delta": -5.4999999999999995e-05,
      "rel": -0.4014598540145985
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.014795,
      "after": 0.017609,
      "delta": 0.002813999999999999,
      "rel": 0.19019939168638045
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2172.428,
      "after": 2360.809,
      "delta": 188.3810000000003,
      "rel": 0.08671449640678555
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2173.043,
      "after": 2361.387,
      "delta": 188.34400000000005,
      "rel": 0.08667292823933996
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.733456,
  "overall_deviation": 0.293382,
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

- In triu.cpp, locate the fast path tiling section where key_value = keyFour is set
- Identify the ub_Sharing_Num assignment in the aligned processing block
- Increase ub_Sharing_Num value to reduce effective UB size per buffer
- Monitor scalar_ratio_mean and MTE wait ratios to verify reduced compute bottlenecks
- Check that UB bandwidth usage rates increase as memory operations become more efficient

## 检索线索

- triu
- tiling
- buffer
- overallocation
- ub_Sharing_Num
- fast_path
- scalar_utilization
- MTE_wait
- aligned_processing
- vector_operations
