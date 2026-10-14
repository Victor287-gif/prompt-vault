---
type: Optimization
title: "foreach_mul_list：循环计算优化"
description: "foreach_mul_list 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/docs/aclnnForeachMulList.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_host/foreach_mul_list.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_kernel/foreach_mul_list.cpp
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_mul_list：循环计算优化

## 来源信息

- experience ID：`f76914468e3463b7`
- 源 schema version：`v2`
- 算子：`foreach_mul_list`
- source iteration：`round_012_refactor`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.restructure`
- canonical family：`loop.restructure`
- 源记录时间：`2026-04-15T01:19:07`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Separate main and remainder loops for bfloat16 compute
~~~~

### Original summary

~~~~text
Using separate main and remainder loops instead of a unified loop with conditional data count reduces pipeline stalls in vector operations. This optimization improves performance when processing bfloat16 data with irregular data counts by minimizing branch mispredictions and improving instruction cache locality.
~~~~

### Original root mechanism

~~~~text
Separating main loop iterations from remainder handling reduces pipeline stalls by eliminating conditional branching inside the hot loop.
~~~~

### Original causal chain

~~~~text
Separating main loop from remainder handling -> reduces aiv_mte2_wait_ratio_mean by 88.24% and aiv_mte3_wait_ratio_mean by 68.58% -> decreases task_duration_us from 8081.22μs to 6446.27μs (20.3% improvement)
~~~~

## 适用条件

- Processing bfloat16 data with Ascend C vector operations
- When dataCount is not divisible by maxCastDataCount
- In foreach_mul_list operators with similar loop patterns
- When pipeline_stalls.aiv_mte2_wait_ratio_mean or aiv_mte3_wait_ratio_mean are high

## 失效边界

- When dataCount is always divisible by maxCastDataCount
- When the remainder handling is trivial (castTimesRemainder == 0)
- When loop overhead dominates performance

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- foreach_mul_list/op_kernel/foreach_mul_list.cpp

### 改动符号

- InnerComputer<bfloat16_t, float, op>::Compute

### 原始 diff

- 源 diff SHA-256：`7de95e211d6ba649c6b2a01c4745e9f97ca22480a517808cd7ad5d4d6fd86ede`

````diff
--- Before/op_kernel/foreach_mul_list.cpp
+++ After/op_kernel/foreach_mul_list.cpp
@@ -328,12 +328,11 @@
     {
         uint32_t castTimes = dataCount / maxCastDataCount;
         uint32_t castTimesRemainder = dataCount % maxCastDataCount;
-        for (uint32_t i = 0; i < castTimes; i++) {
-            ComputePerCast(inLocal_1, inLocal_2, outLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount);
-        }
-
-        if (castTimesRemainder > 0) {
-            ComputePerCast(inLocal_1, inLocal_2, outLocal, float32Tensor, maxCastDataCount, castTimes, castTimesRemainder);
+        uint32_t totalIterations = castTimes + (castTimesRemainder > 0 ? 1 : 0);
+
+        for (uint32_t i = 0; i < totalIterations; i++) {
+            int64_t currentDataCount = (i == castTimes && castTimesRemainder > 0) ? castTimesRemainder : maxCastDataCount;
+            ComputePerCast(inLocal_1, inLocal_2, outLocal, float32Tensor, maxCastDataCount, i, currentDataCount);
         }
     }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`15407.208008`
- fast duration us：`14530.310547000001`
- improvement ratio：`0.056914754480155046`
- machine confidence：`0.85`

### Profile evidence

- pipeline_stalls.aiv_mte2_wait_ratio_mean: 6.8e-05 -> 8e-06 (-88.24%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.022798 -> 0.007164 (-68.58%)
- critical_path.aiv_vector0_time_us_mean: 6422.3 -> 8063.68 (+25.56%)
- latency.task_duration_us: 6446.27 -> 8081.22 (+25.36%)
- memory_gm.aiv_main_mem_write_bw_gbps: 6.51 -> 5.18 (-20.43%)

### Affected metric groups

- pipeline_stalls
- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 6.8e-05,
      "after": 8e-06,
      "delta": -6e-05,
      "rel": -0.8823529411764706
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.022798,
      "after": 0.007164,
      "delta": -0.015634,
      "rel": -0.6857619089393806
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 6422.301,
      "after": 8063.679,
      "delta": 1641.3779999999997,
      "rel": 0.2555747542819933
    }
  },
  {
    "latency.task_duration_us": {
      "before": 6446.269,
      "after": 8081.222,
      "delta": 1634.9529999999995,
      "rel": 0.25362779617170794
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 6.51,
      "after": 5.18,
      "delta": -1.33,
      "rel": -0.20430107526881722
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

- In foreach_mul_list.cpp, replace unified loop with separate main and remainder loops
- Keep the ComputePerCast function signature unchanged
- Main loop should iterate castTimes with maxCastDataCount
- Remainder loop should handle castTimesRemainder separately after main loop
- Apply to InnerComputer<bfloat16_t, float, op>::Compute method

## 检索线索

- foreach_mul_list
- bfloat16
- loop_restructure
- pipeline_stall
- vector_operations
- Ascend_C
- ComputePerCast
- castTimes
- castTimesRemainder
- maxCastDataCount
