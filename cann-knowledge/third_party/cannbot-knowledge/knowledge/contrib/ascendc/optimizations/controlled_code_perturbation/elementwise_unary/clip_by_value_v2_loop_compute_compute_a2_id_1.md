---
type: Optimization
title: "clip_by_value_v2：循环计算优化"
description: "clip_by_value_v2 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access"]
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
# clip_by_value_v2：循环计算优化

## 来源信息

- experience ID：`113a743c052b7bbc`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_013_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`loop.optimization.fusion`
- canonical family：`loop.optimization.fusion.break`
- 源记录时间：`2026-04-20T15:40:12`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Fuse min/max initialization loops to reduce loop control overhead
~~~~

### Original summary

~~~~text
Combining separate min and max value initialization loops into a single loop reduces loop control overhead and improves performance by approximately 14.2%. This optimization is applicable when initializing multiple local tensors with constant values in Ascend C kernels, particularly in ClipByValueV2 and similar operators with scalar initialization patterns.
~~~~

### Original root mechanism

~~~~text
Fusing separate initialization loops for min and max values reduces loop control overhead and improves instruction cache efficiency by combining similar operations into a single loop.
~~~~

### Original causal chain

~~~~text
Fusing min and max initialization loops → reduces loop control instructions → decreases pipeline stalls (aiv_vec_bank_cflt_ratio_mean improved by -20.00%) → reduces vector wait ratio → improves overall latency (task_duration_us improved by 16.58%)
~~~~

## 适用条件

- When initializing multiple local tensors with constant values in Ascend C kernels
- When using scalar loops with SetValue operations on contiguous memory
- When similar initialization patterns exist for multiple variables in clip_by_value_v2.cpp
- When loop fusion opportunities exist between consecutive initialization loops

## 失效边界

- When initialization values depend on loop index or require different computation per iteration
- When loops have different iteration counts or control flow
- When memory access patterns would become non-contiguous after fusion

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

### 原始 diff

- 源 diff SHA-256：`f31a9a41598088edea3f95a646bd9539a1827f09f86bdc38e991833474eec084`

````diff
--- Before/op_kernel/clip_by_value_v2.cpp
+++ After/op_kernel/clip_by_value_v2.cpp
@@ -82,8 +82,12 @@
         AscendC::LocalTensor<TYPE_CLIP_VALUE_MIN> min = Q_min.Get<TYPE_CLIP_VALUE_MIN>();
         AscendC::LocalTensor<TYPE_CLIP_VALUE_MAX> max = Q_max.Get<TYPE_CLIP_VALUE_MAX>();
         // Degradation: Replace vectorized Duplicate with scalar loop to break vectorization
+        // Additional degradation: Separate min and max setting into two loops
+        // to break potential loop fusion optimization and increase loop overhead
         for (uint64_t i = 0; i < this->processDataNum; ++i) {
             min.SetValue(i, this->clip_value_min);
+        }
+        for (uint64_t i = 0; i < this->processDataNum; ++i) {
             max.SetValue(i, this->clip_value_max);
         }
         AscendC::Max(y, x, min, this->processDataNum);
@@ -202,8 +206,12 @@
         AscendC::LocalTensor<float> min = Q_min_fp32.Get<float>();
         AscendC::LocalTensor<float> max = Q_max_fp32.Get<float>();
         // Degradation: Replace vectorized Duplicate with scalar loop to break vectorization
+        // Additional degradation: Separate min and max setting into two loops
+        // to break potential loop fusion optimization and increase loop overhead
         for (uint64_t i = 0; i < this->processDataNum; ++i) {
             min.SetValue(i, this->clip_value_min);
+        }
+        for (uint64_t i = 0; i < this->processDataNum; ++i) {
             max.SetValue(i, this->clip_value_max);
         }
         AscendC::Max(y_fp32, x_fp32, min, this->processDataNum);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`87387.40625`
- fast duration us：`74956.734375`
- improvement ratio：`0.1422478639477871`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 5e-06 -> 4e-06 (-20.00%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.003501 -> 0.004175 (+19.25%)
- critical_path.aiv_vector0_time_us_mean: 74946.3 -> 87376.4 (+16.59%)
- latency.task_duration_us: 74956.7 -> 87387.4 (+16.58%)
- aiv_utilization.mte2_ratio_mean: 0.004037 -> 0.003444 (-14.69%)

### Affected metric groups

- pipeline_stalls
- critical_path
- latency
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 5e-06,
      "after": 4e-06,
      "delta": -1.0000000000000006e-06,
      "rel": -0.2000000000000001
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.003501,
      "after": 0.004175,
      "delta": 0.0006739999999999997,
      "rel": 0.19251642387889165
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 74946.272,
      "after": 87376.449,
      "delta": 12430.176999999996,
      "rel": 0.16585450707941812
    }
  },
  {
    "latency.task_duration_us": {
      "before": 74956.734,
      "after": 87387.406,
      "delta": 12430.672000000006,
      "rel": 0.16583796193681552
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.004037,
      "after": 0.003444,
      "delta": -0.0005929999999999998,
      "rel": -0.14689125588308144
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

- Identify consecutive loops initializing local tensors with constant values in clip_by_value_v2.cpp
- Combine loops that use SetValue on AscendC::LocalTensor objects with the same iteration count
- Apply to KernelClipByValueV2::Compute function where min and max tensors are initialized
- Ensure the combined loop maintains contiguous memory access patterns for both tensors

## 检索线索

- loop fusion
- SetValue
- AscendC
- LocalTensor
- clip_by_value_v2
- pipeline stall
- loop control overhead
- scalar initialization
- constant value initialization
- min max initialization
