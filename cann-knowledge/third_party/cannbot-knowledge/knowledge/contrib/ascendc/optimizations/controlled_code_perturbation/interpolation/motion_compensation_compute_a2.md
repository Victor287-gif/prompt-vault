---
type: Optimization
title: "motion_compensation：计算优化"
description: "motion_compensation 算子的计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/docs/MotionCompensation.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/tf_plugin/tensorflow_motion_compensation_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_host/motion_compensation.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_host/motion_compensation_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_kernel/motion_compensation.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/opp_kernel_aicpu/.gitkeep
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# motion_compensation：计算优化

## 来源信息

- experience ID：`e638549e49395499`
- 源 schema version：`v2`
- 算子：`motion_compensation`
- source iteration：`round_001_degrade`
- 源 category：`image`
- 计算分区：`interpolation`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`compute.redundant.scalar`
- canonical family：`compute.redundant.scalar`
- 源记录时间：`2026-04-14T21:45:49`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant scalar validation loops in vector compute paths
~~~~

### Original summary

~~~~text
Avoid inserting scalar validation loops with branching logic inside vector compute kernels, especially when operating on data already in vector registers. These loops force serial execution of operations that could remain vectorized, dramatically increasing scalar unit utilization (from 77% to 96%) while reducing vector unit utilization (from 33% to 6%). Apply this optimization when you see high scalar_ratio_mean with low vec_ratio_mean in AIV utilization metrics.
~~~~

### Original root mechanism

~~~~text
Removing per-element scalar computation loops with branching and redundant arithmetic operations from vector compute paths eliminates serial execution bottlenecks and reduces scalar unit pressure, allowing vector operations to proceed without interference.
~~~~

### Original causal chain

~~~~text
apply compute.redundant.scalar in motion_compensation/op_kernel/motion_compensation.cpp -> improve memory_ub.ub_read_bw_scalar_gbps -> 75.6% latency improvement
~~~~

## 适用条件

- AIV utilization shows scalar_ratio_mean > 0.9 and vec_ratio_mean < 0.1
- Vector compute kernels contain per-element scalar loops with GetValue()/SetValue() calls
- Code patterns show branching logic inside loops operating on LocalTensor<float> data
- UB scalar bandwidth metrics (ub_read_bw_scalar_gbps, ub_write_bw_scalar_gbps) are unexpectedly high (>0.2 Gbps)
- Critical path time (aiv_vector0_time_us_mean) dominates task duration

## 失效边界

- Actual scalar validation is required for correctness (e.g., NaN/inf checks)
- Data-dependent branching cannot be vectorized
- Operations require per-element precision adjustments
- Scalar operations are truly independent and cannot be batched

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- motion_compensation/op_kernel/motion_compensation.cpp

### 改动符号

- KernelMotionCompensation::Compute

### 代码签名

- GetValue
- SetValue
- contiguous_access
- scalar_value_access
- ub

### 原始 diff

- 源 diff SHA-256：`2b994d9f31c166dc715397a2dd5944013790d81979052aad5e777128bbf981b4`

````diff
--- Before/op_kernel/motion_compensation.cpp
+++ After/op_kernel/motion_compensation.cpp
@@ -210,6 +210,28 @@
         Sub(low32Local, high32Local, low32Local, this->processDataNum);

         Cast(low32LocalFp, low32Local, RoundMode::CAST_NONE, this->processDataNum);
+
+        // Degradation: Add redundant scalar computation loop
+        for (int32_t i = 0; i < this->processDataNum; i++) {
+            float val = low32LocalFp.GetValue(i);
+            // Redundant computation with multiple branches
+            float temp = val;
+            if (temp < 0.0f) {
+                temp = -temp;
+            }
+            if (temp > 1000.0f) {
+                temp = temp * 0.001f;
+            } else if (temp < 0.001f && temp > 0.0f) {
+                temp = temp * 1000.0f;
+            }
+            // Force dependency chain
+            temp = temp + 1.0f;
+            temp = temp - 1.0f;
+            temp = temp * 2.0f;
+            temp = temp / 2.0f;
+            // Final redundant assignment
+            low32LocalFp.SetValue(i, val + (temp - temp));
+        }

         LocalTensor<float> tLocal = low32LocalFp[0];
         Muls(tLocal, tLocal, f, this->processDataNum);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`7587.271973`
- fast duration us：`1852.99707`
- improvement ratio：`0.7557755835570334`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_read_bw_scalar_gbps: 0.0036 -> 0.2678 (+7338.89%)
- memory_ub.ub_write_bw_scalar_gbps: 0.0042 -> 0.2677 (+6273.81%)
- critical_path.aiv_vector0_time_us_mean: 1280.99 -> 6985.03 (+445.28%)
- latency.task_duration_us: 1853 -> 7587.27 (+309.46%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 0.88 -> 0.15 (-82.95%)

### Affected metric groups

- memory_ub
- critical_path
- latency
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "metric": "memory_ub.ub_read_bw_scalar_gbps",
    "before": 0.0036,
    "after": 0.2678,
    "delta": 0.2642,
    "rel": 73.38888888888889
  },
  {
    "metric": "memory_ub.ub_write_bw_scalar_gbps",
    "before": 0.0042,
    "after": 0.2677,
    "delta": 0.2635,
    "rel": 62.73809523809525
  },
  {
    "metric": "critical_path.aiv_vector0_time_us_mean",
    "before": 1280.994,
    "after": 6985.028,
    "delta": 5704.034000000001,
    "rel": 4.452818670501189
  },
  {
    "metric": "latency.task_duration_us",
    "before": 1852.997,
    "after": 7587.272,
    "delta": 5734.275,
    "rel": 3.094594864427735
  },
  {
    "metric": "memory_ub.ub_to_gm_bw_usage_rate_pct",
    "before": 0.88,
    "after": 0.15,
    "delta": -0.73,
    "rel": -0.8295454545454545
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

- 1. Search for GetValue()/SetValue() patterns in vector compute kernels, particularly in motion_compensation/op_kernel/motion_compensation.cpp::KernelMotionCompensation::Compute
- 2. Identify scalar loops that operate on LocalTensor<float> data already in vector registers
- 3. Remove redundant validation logic (absolute value checks, range normalization, dependency chains like temp = temp + 1.0f; temp = temp - 1.0f)
- 4. Convert scalar operations to vector operations using Ascend C vector intrinsics where possible
- 5. Profile with AIV metrics to verify vec_ratio_mean increases and scalar_ratio_mean decreases

## 检索线索

- scalar_loop
- GetValue
- SetValue
- vectorization
- redundant_computation
- branching
- LocalTensor
- AIV_utilization
- scalar_ratio
- vec_ratio
- motion_compensation
- compute_path
- pipeline_stall
