---
type: Optimization
title: "arange：内存优化"
description: "arange 算子的内存优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/docs/Arange.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/tf_plugin/tensorflow_arange_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_host/arange.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_host/arange_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_kernel/arange.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/Arange_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/test_arange.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# arange：内存优化

## 来源信息

- experience ID：`1bf0a4ac5b7ac99c`
- 源 schema version：`v2`
- 算子：`arange`
- source iteration：`round_010_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.redundant_access`
- canonical family：`untitled`
- 源记录时间：`2026-04-15T00:54:41`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant memory accesses in scalar loops
~~~~

### Original summary

~~~~text
In Ascend C operators using scalar loops, avoid reading the same tensor value multiple times within a single iteration. Each redundant GetValue() call increases memory bandwidth usage and creates pipeline stalls. Instead, read each tensor value once per iteration and reuse the value for all computations in that iteration.
~~~~

### Original root mechanism

~~~~text
Redundant memory reads from the same tensor location within a single loop iteration cause unnecessary memory bandwidth pressure and pipeline stalls.
~~~~

### Original causal chain

~~~~text
Removing redundant GetValue() calls reduces memory bandwidth pressure → decreases aiv_utilization.scalar_ratio_mean from 0.363959 to 0.063191 (-82.6%) → reduces memory_ub.ub_write_bw_vector_gbps from 0.1 to 0.07 Gbps (-30%) → reduces critical_path.aiv_vector0_time_us_mean from 5.09409e+06 to 3.95513e+06 us (-22.4%) → improves overall latency by 22.4%.
~~~~

## 适用条件

- Operator uses scalar loops with GetValue() calls in Ascend C
- Same tensor value is read multiple times within a single loop iteration
- Operator shows high aiv_utilization.scalar_ratio_mean (>0.3)
- Memory bandwidth metrics (ub_write_bw_vector_gbps, aiv_main_mem_write_bw_gbps) show high usage
- Performance bottleneck is pipeline_stall_bound

## 失效边界

- Vectorized implementations are available and appropriate
- Memory reads are already optimized with proper caching
- Operator is compute-bound rather than memory-bound
- Tensor values change between reads within the same iteration

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/arange.cpp

### 代码签名

- AllocTensor
- EnQue
- GetValue
- SetValue
- TQue
- contiguous_access
- queue_api
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`3e11b0a68f95c01da10e5f757d94a8aa1e3b0cf7bf226d4727f45516c4ca1a93`

````diff
--- Before/op_kernel/arange.cpp
+++ After/op_kernel/arange.cpp
@@ -121,15 +121,33 @@
     uint32_t calc_num = ALIGN_UP_32B_ELEMENTS(num, float);

     AscendC::LocalTensor<TYPE_OUT> outLocal = outQueue.AllocTensor<TYPE_OUT>();
-    // Replace vectorized Add with scalar loop
+    // Replace vectorized Add with scalar loop with redundant memory accesses
     for (int32_t idx = 0; idx < calc_num; idx++) {
-      float val = this->calc_init.GetValue(idx) + this->calc_temp.GetValue(idx);
-      this->calc_out.SetValue(idx, val);
-    }
-    // Replace vectorized Add with scalar loop
+      // Read values multiple times to increase memory bandwidth pressure
+      float init_val1 = this->calc_init.GetValue(idx);
+      float temp_val1 = this->calc_temp.GetValue(idx);
+      float sum1 = init_val1 + temp_val1;
+      // Redundant second read and computation
+      float init_val2 = this->calc_init.GetValue(idx);
+      float temp_val2 = this->calc_temp.GetValue(idx);
+      float sum2 = init_val2 + temp_val2;
+      // Use average of both (same as either) to prevent optimization
+      float final_sum = (sum1 + sum2) * 0.5f;
+      this->calc_out.SetValue(idx, final_sum);
+    }
+    // Replace vectorized Add with scalar loop with redundant memory accesses
     for (int32_t idx = 0; idx < calc_num; idx++) {
-      float val = this->calc_temp.GetValue(idx) + this->calc_step.GetValue(idx);
-      this->calc_temp.SetValue(idx, val);
+      // Read values multiple times to increase memory bandwidth pressure
+      float temp_val1 = this->calc_temp.GetValue(idx);
+      float step_val1 = this->calc_step.GetValue(idx);
+      float sum1 = temp_val1 + step_val1;
+      // Redundant second read and computation
+      float temp_val2 = this->calc_temp.GetValue(idx);
+      float step_val2 = this->calc_step.GetValue(idx);
+      float sum2 = temp_val2 + step_val2;
+      // Use average of both (same as either) to prevent opti... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`5094192.0`
- fast duration us：`3955211.75`
- improvement ratio：`0.2235840835995188`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.363959 -> 0.063191 (-82.64%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.1 -> 0.07 (-30.00%)
- memory_ub.ub_write_bw_vector_gbps: 0.1 -> 0.07 (-30.00%)
- critical_path.aiv_vector0_time_us_mean: 3.95513e+06 -> 5.09409e+06 (+28.80%)
- latency.task_duration_us: 3.95521e+06 -> 5.09419e+06 (+28.80%)

### Affected metric groups

- aiv_utilization
- memory_gm
- memory_ub
- critical_path

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.363959,
      "after": 0.063191,
      "delta": -0.300768,
      "rel": -0.8263787954137691
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 0.1,
      "after": 0.07,
      "delta": -0.03,
      "rel": -0.3
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 0.1,
      "after": 0.07,
      "delta": -0.03,
      "rel": -0.3
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 3955132.0,
      "after": 5094089.5,
      "delta": 1138957.5,
      "rel": 0.28796952920913893
    }
  },
  {
    "latency.task_duration_us": {
      "before": 3955211.75,
      "after": 5094192.0,
      "delta": 1138980.25,
      "rel": 0.2879694747063795
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

- 1. Identify scalar loops in op_kernel/*.cpp files that call GetValue() multiple times on the same tensor
- 2. For each loop iteration, read each tensor value once and store in local variable
- 3. Replace multiple GetValue() calls with the stored local variable
- 4. In arange.cpp: change from reading calc_init.GetValue(idx) and calc_temp.GetValue(idx) twice per iteration to reading once each
- 5. Remove redundant computations like (sum1 + sum2) * 0.5f when sum1 == sum2
- 6. Verify performance improvement by checking aiv_utilization.scalar_ratio_mean reduction

## 检索线索

- redundant_memory_access
- scalar_loop
- GetValue
- memory_bandwidth
- pipeline_stall
- arange
- AscendC
- vector_to_scalar
- memory_optimization
- bandwidth_pressure
