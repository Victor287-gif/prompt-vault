---
type: Optimization
title: "clip_by_value_v2：内存优化"
description: "clip_by_value_v2 算子的内存优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/docs/ClipByValueV2.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/framework/tf_plugin/tensorflow_clip_by_value_v2_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/op_host/clip_by_value_v2.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/op_host/clip_by_value_v2_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/op_kernel/clip_by_value_v2.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/atk/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/atk/generate_reduce.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/atk/op_clip.yaml
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/st/ClipByValueV2_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/st/test_clip_by_value_v2.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# clip_by_value_v2：内存优化

## 来源信息

- experience ID：`2b628beea64e6a3f`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_019_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.write.redundant.initialization`
- canonical family：`memory.write.redundant.initialization`
- 源记录时间：`2026-04-20T16:23:58`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant tensor initialization before setting actual values
~~~~

### Original summary

~~~~text
Avoid redundant tensor initialization with zeros before setting actual values in Ascend C kernels. This optimization eliminates unnecessary memory write operations and reduces loop overhead, improving performance by ~41.6%. Apply this technique when tensors are initialized with placeholder values that are immediately overwritten with actual computation results.
~~~~

### Original root mechanism

~~~~text
Removing redundant zero initialization loops before setting actual clip values reduces memory write operations and loop overhead, improving pipeline efficiency.
~~~~

### Original causal chain

~~~~text
Removing redundant zero initialization loops -> reduces memory write bandwidth usage (UB write BW decreases from 0.48 to 0.28 Gbps) -> decreases pipeline stall ratios (aiv_mte3_wait_ratio_mean decreases from 0.002083 to 0.001215) -> reduces critical path latency (aiv_vector0_time_us_mean decreases from 149516 to 87376 µs) -> improves overall task duration by 41.6%.
~~~~

## 适用条件

- When tensors are initialized with placeholder values before being set to actual computation results
- When using SetValue() operations in loops for tensor initialization
- When memory write bandwidth to UB (Unified Buffer) shows high usage (e.g., ub_write_bw_vector_gbps > 0.2 Gbps)
- When pipeline stall metrics (aiv_mte3_wait_ratio_mean) indicate memory access bottlenecks

## 失效边界

- When tensor initialization is required for data dependency or safety reasons
- When the initialization values are actually used in subsequent computations
- When the initialization loop serves a purpose beyond just setting placeholder values

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- clip_by_value_v2/op_kernel/clip_by_value_v2.cpp

### 改动符号

- KernelClipByValueV2::Compute
- KernelClipByValueV2<bfloat16_t,bfloat16_t,bfloat16_t,bfloat16_t,IsExistBigCore>::Compute

### 代码签名

- SetValue
- contiguous_access
- scalar_value_access
- ub

### 原始 diff

- 源 diff SHA-256：`39fee8b65247f855b599542896ca3b0d4b907182fb21a90ca2cd5451f30bed28`

````diff
--- Before/op_kernel/clip_by_value_v2.cpp
+++ After/op_kernel/clip_by_value_v2.cpp
@@ -84,6 +84,13 @@
         // Degradation: Replace vectorized Duplicate with scalar loop to break vectorization
         // Additional degradation: Separate min and max setting into two loops
         // to break potential loop fusion optimization and increase loop overhead
+        // New degradation: Add redundant tensor initialization with zeros before setting actual values
+        for (uint64_t i = 0; i < this->processDataNum; ++i) {
+            min.SetValue(i, 0);
+        }
+        for (uint64_t i = 0; i < this->processDataNum; ++i) {
+            max.SetValue(i, 0);
+        }
         for (uint64_t i = 0; i < this->processDataNum; ++i) {
             min.SetValue(i, this->clip_value_min);
         }
@@ -208,6 +215,13 @@
         // Degradation: Replace vectorized Duplicate with scalar loop to break vectorization
         // Additional degradation: Separate min and max setting into two loops
         // to break potential loop fusion optimization and increase loop overhead
+        // New degradation: Add redundant tensor initialization with zeros before setting actual values
+        for (uint64_t i = 0; i < this->processDataNum; ++i) {
+            min.SetValue(i, 0.0f);
+        }
+        for (uint64_t i = 0; i < this->processDataNum; ++i) {
+            max.SetValue(i, 0.0f);
+        }
         for (uint64_t i = 0; i < this->processDataNum; ++i) {
             min.SetValue(i, this->clip_value_min);
         }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`149528.6875`
- fast duration us：`87387.40625`
- improvement ratio：`0.4155809984622516`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 87376.4 -> 149516 (+71.12%)
- latency.task_duration_us: 87387.4 -> 149529 (+71.11%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 0.14 -> 0.08 (-42.86%)
- aiv_utilization.mte2_ratio_mean: 0.003444 -> 0.002007 (-41.72%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.002083 -> 0.001215 (-41.67%)

### Affected metric groups

- critical_path
- latency
- memory_ub
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 87376.449,
      "after": 149516.371,
      "delta": 62139.92200000002,
      "rel": 0.7111747239808295
    }
  },
  {
    "latency.task_duration_us": {
      "before": 87387.406,
      "after": 149528.688,
      "delta": 62141.28199999999,
      "rel": 0.7111011167902156
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 0.14,
      "after": 0.08,
      "delta": -0.06000000000000001,
      "rel": -0.4285714285714286
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.003444,
      "after": 0.002007,
      "delta": -0.001437,
      "rel": -0.41724738675958184
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.002083,
      "after": 0.001215,
      "delta": -0.0008680000000000003,
      "rel": -0.41670667306769094
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

- 1. Examine the KernelClipByValueV2::Compute function in clip_by_value_v2/op_kernel/clip_by_value_v2.cpp for redundant initialization patterns
- 2. Look for loops that call SetValue() with placeholder values (like 0 or 0.0f) followed by another SetValue() with actual computation results
- 3. Remove the redundant initialization loops that write placeholder values before the actual computation
- 4. Verify that removing the initialization doesn't break data dependencies or correctness
- 5. Check performance metrics: expect ~40% reduction in ub_write_bw_vector_gbps and similar improvements in pipeline stall ratios

## 检索线索

- redundant_initialization
- memory_write
- SetValue
- clip_by_value_v2
- tensor_initialization
- loop_overhead
- memory_bandwidth
- pipeline_stall
- ub_write
- ascend_c
