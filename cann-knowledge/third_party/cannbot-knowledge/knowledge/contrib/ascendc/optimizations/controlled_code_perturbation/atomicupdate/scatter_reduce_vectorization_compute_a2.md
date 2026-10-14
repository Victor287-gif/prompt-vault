---
type: Optimization
title: "scatter_reduce：向量化计算优化"
description: "scatter_reduce 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.atomic", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/docs/ScatterReduce.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/framework/tf_plugin/tensorflow_scatter_reduce_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/op_host/scatter_reduce.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/op_host/scatter_reduce_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/op_kernel/scatter_reduce.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/op_kernel/scatter_reduce_sca.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/op_kernel/scatter_reduce_spec1.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/scatter_reduce/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# scatter_reduce：向量化计算优化

## 来源信息

- experience ID：`002cbb285b31a3ec`
- 源 schema version：`v2`
- 算子：`scatter_reduce`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`atomicupdate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vectorization.break`
- canonical family：`vectorization.break`
- 源记录时间：`2026-04-14T20:53:42`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Replace vectorized unrolled loop with scalar loop for scatter reduction
~~~~

### Original summary

~~~~text
In scatter reduction operations, replacing a vectorized 4-element unrolled loop with a scalar loop improves performance by 26.5%. This optimization reduces pipeline stalls from 13.67% to 6.86% and improves vector unit utilization from 27.18% to 38.42%. Apply this pattern when scatter operations involve non-contiguous memory access patterns that cause pipeline conflicts in vector units.
~~~~

### Original root mechanism

~~~~text
Vectorized 4-element unrolled loop causes pipeline stalls due to memory access patterns in scatter reduction operations
~~~~

### Original causal chain

~~~~text
apply vectorization.break in scatter_reduce/op_kernel/scatter_reduce.cpp -> improve pipeline_stalls.aiv_vec_wait_ratio_mean -> 26.5% latency improvement
~~~~

## 适用条件

- Scatter reduction operations with non-contiguous memory access patterns
- Operations using GetValue/SetValue APIs for random memory access
- When pipeline_stalls.aiv_vec_wait_ratio_mean > 10%
- When vector unit utilization (vec_ratio_mean) is below 30%
- In scatter_reduce.cpp ScatterDeduceSpec1::Compute function

## 失效边界

- Contiguous memory access patterns with stride=1
- Operations where vectorization provides clear memory bandwidth benefits
- When aiv_vec_wait_ratio_mean < 5%
- When vector unit utilization (vec_ratio_mean) > 50%

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- scatter_reduce/op_kernel/scatter_reduce.cpp

### 改动符号

- ScatterDeduceSpec1::Compute

### 代码签名

- GetValue
- SetValue
- contiguous_access
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`10777a42c7373e84c73bfe95cab873ab7a03310f89dca78c277879fd45697947`

````diff
--- Before/op_kernel/scatter_reduce.cpp
+++ After/op_kernel/scatter_reduce.cpp
@@ -150,25 +150,10 @@

             ShiftRight(calIdxBuf, calIdxBuf, 2, calNum);
             int j=0;
-            for (; j < calNum-3; j+=4) {
-                int tmp1 = calIdxBuf.GetValue(j);
-                int tmp2 = calIdxBuf.GetValue(j+1);
-                int tmp3 = calIdxBuf.GetValue(j+2);
-                int tmp4 = calIdxBuf.GetValue(j+3);
-                DTYPE_Y tmpy1 = tmpYBuf.GetValue(j);
-                DTYPE_Y tmpy2 = tmpYBuf.GetValue(j+1);
-                DTYPE_Y tmpy3 = tmpYBuf.GetValue(j+2);
-                DTYPE_Y tmpy4 = tmpYBuf.GetValue(j+3);
-                outYLocal.SetValue(tmp1, tmpy1);
-                outYLocal.SetValue(tmp2, tmpy2);
-                outYLocal.SetValue(tmp3, tmpy3);
-                outYLocal.SetValue(tmp4, tmpy4);
-            }
-            while (j<calNum)
-            {
+            for (; j < calNum; j++) {
                 int tmp = calIdxBuf.GetValue(j);
-                outYLocal.SetValue(tmp, tmpYBuf.GetValue(j));
-                j++;
+                DTYPE_Y tmpy = tmpYBuf.GetValue(j);
+                outYLocal.SetValue(tmp, tmpy);
             }
         }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`1796.135864`
- fast duration us：`1319.626343`
- improvement ratio：`0.26529703601531124`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.136747 -> 0.068573 (-49.85%)
- critical_path.aiv_vector0_time_us_mean: 1319.12 -> 1795.6 (+36.12%)
- latency.task_duration_us: 1319.63 -> 1796.14 (+36.11%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.000362 -> 0.000254 (-29.83%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.001164 -> 0.000821 (-29.47%)

### Affected metric groups

- pipeline_stalls
- critical_path
- latency
- arithmetic

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.136747,
      "after": 0.068573,
      "delta": -0.06817400000000001,
      "rel": -0.49854110145012326
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1319.116,
      "after": 1795.596,
      "delta": 476.48,
      "rel": 0.3612115992831563
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1319.626,
      "after": 1796.136,
      "delta": 476.51,
      "rel": 0.36109473441717577
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 0.000362,
      "after": 0.000254,
      "delta": -0.00010800000000000002,
      "rel": -0.29834254143646416
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.001164,
      "after": 0.000821,
      "delta": -0.0003430000000000001,
      "rel": -0.2946735395189004
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

- Identify scatter reduction loops using GetValue/SetValue for non-contiguous access
- Check if loop is vectorized with 4-element unrolling (j+=4 pattern)
- Replace vectorized unrolled loop with scalar loop: for (; j < calNum; j++)
- Maintain same computation logic but access elements individually
- Test performance impact on pipeline_stalls.aiv_vec_wait_ratio_mean metric

## 检索线索

- scatter_reduce
- vectorization.break
- pipeline_stall
- GetValue
- SetValue
- non-contiguous
- memory_access
- ScatterDeduceSpec1
- scatter_reduce.cpp
- calIdxBuf
- tmpYBuf
- outYLocal
