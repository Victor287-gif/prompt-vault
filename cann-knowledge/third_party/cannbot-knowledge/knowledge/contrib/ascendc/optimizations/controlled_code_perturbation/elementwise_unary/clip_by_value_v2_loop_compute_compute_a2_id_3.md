---
type: Optimization
title: "clip_by_value_v2：循环计算优化"
description: "clip_by_value_v2 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
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

- experience ID：`4a6a39ff0b7910c8`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_019_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.combine`
- canonical family：`loop.combine`
- 源记录时间：`2026-04-20T16:25:23`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Combine min and max setting loops into single loop
~~~~

### Original summary

~~~~text
When initializing multiple buffers with constant values in Ascend C operators, combine separate loops into a single loop to reduce pipeline stalls and improve vector unit utilization. This optimization is particularly effective when the same loop bounds and similar operations are applied to different buffers, as it reduces loop overhead and improves instruction cache locality.
~~~~

### Original root mechanism

~~~~text
Combining separate loops that perform similar operations on same-sized buffers into a single loop reduces pipeline stalls and improves vector unit utilization by minimizing loop overhead and improving instruction cache locality.
~~~~

### Original causal chain

~~~~text
Combining min and max initialization loops -> reduces pipeline stalls (aiv_vec_bank_cflt_ratio_mean decreased 20%, aiv_mte3_wait_ratio_mean decreased 14.6%) -> improves vector unit utilization (vec_ratio_mean increased 16.6%) -> reduces task duration by 14.2% (87387μs to 74957μs)
~~~~

## 适用条件

- When initializing multiple LocalTensor buffers with constant values using SetValue operations
- When separate loops perform identical operations on buffers with the same size and data dependencies
- When loop bounds are identical across multiple initialization loops
- When using AscendC::LocalTensor<T> and SetValue() for buffer initialization

## 失效边界

- When operations in different loops have data dependencies that require separate execution
- When loop bounds differ between operations
- When operations have different computational patterns that would create pipeline conflicts
- When the combined loop would exceed vector unit capacity or cause register pressure

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- clip_by_value_v2/op_kernel/clip_by_value_v2.cpp

### 改动符号

- KernelClipByValueV2::Compute

### 代码签名

- AllocTensor
- SetValue
- contiguous_access
- queue_api
- scalar_value_access
- template

### 原始 diff

- 源 diff SHA-256：`74d8813685b37d0e1215cc333127c188e7ab9c2c0502c80ac1dbc4602dd342ef`

````diff
--- Before/op_kernel/clip_by_value_v2.cpp
+++ After/op_kernel/clip_by_value_v2.cpp
@@ -81,13 +81,9 @@
         AscendC::LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
         AscendC::LocalTensor<TYPE_CLIP_VALUE_MIN> min = Q_min.Get<TYPE_CLIP_VALUE_MIN>();
         AscendC::LocalTensor<TYPE_CLIP_VALUE_MAX> max = Q_max.Get<TYPE_CLIP_VALUE_MAX>();
-        // Degradation: Replace vectorized Duplicate with scalar loop to break vectorization
-        // Additional degradation: Separate min and max setting into two loops
-        // to break potential loop fusion optimization and increase loop overhead
+        // Combine min and max setting into a single loop to reduce loop overhead
         for (uint64_t i = 0; i < this->processDataNum; ++i) {
             min.SetValue(i, this->clip_value_min);
-        }
-        for (uint64_t i = 0; i < this->processDataNum; ++i) {
             max.SetValue(i, this->clip_value_max);
         }
         AscendC::Max(y, x, min, this->processDataNum);
@@ -205,13 +201,9 @@
         AscendC::Cast(x_fp32,x_bf16,AscendC::RoundMode::CAST_NONE, this->processDataNum);
         AscendC::LocalTensor<float> min = Q_min_fp32.Get<float>();
         AscendC::LocalTensor<float> max = Q_max_fp32.Get<float>();
-        // Degradation: Replace vectorized Duplicate with scalar loop to break vectorization
-        // Additional degradation: Separate min and max setting into two loops
-        // to break potential loop fusion optimization and increase loop overhead
+        // Combine min and max setting into a single loop to reduce loop overhead
         for (uint64_t i = 0; i < this->processDataNum; ++i) {
             min.SetValue(i, this->clip_value_min);
-        }
-        for (uint64_t i = 0; i < this->processDataNum; ++i) {
             max.SetValue(i, this->clip_value_max);
         }
         AscendC::Max(y_fp32, x_fp32, min, this->processDataNum);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`87387.40625`
- fast duration us：`74956.601562`
- improvement ratio：`0.14224938376632507`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 5e-06 -> 4e-06 (-20.00%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.003498 -> 0.004175 (+19.35%)
- critical_path.aiv_vector0_time_us_mean: 74947 -> 87376.4 (+16.58%)
- latency.task_duration_us: 74956.6 -> 87387.4 (+16.58%)
- aiv_utilization.mte2_ratio_mean: 0.004038 -> 0.003444 (-14.71%)

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
      "before": 0.003498,
      "after": 0.004175,
      "delta": 0.0006770000000000001,
      "rel": 0.1935391652372785
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 74946.961,
      "after": 87376.449,
      "delta": 12429.487999999998,
      "rel": 0.16584378918312642
    }
  },
  {
    "latency.task_duration_us": {
      "before": 74956.602,
      "after": 87387.406,
      "delta": 12430.804000000004,
      "rel": 0.16584001499961276
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.004038,
      "after": 0.003444,
      "delta": -0.0005939999999999999,
      "rel": -0.14710252600297175
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

- In clip_by_value_v2/op_kernel/clip_by_value_v2.cpp, combine separate min and max initialization loops into a single loop in the KernelClipByValueV2::Compute function
- Replace two sequential loops with identical bounds with a single loop containing both SetValue operations
- Ensure data dependencies are preserved when combining loops - operations should maintain their original execution order
- Verify that the combined loop doesn't introduce register pressure or exceed vector unit capacity

## 检索线索

- loop_combine
- pipeline_stall
- vector_utilization
- clip_by_value
- SetValue
- LocalTensor
- AscendC
- buffer_initialization
- loop_overhead
- instruction_cache
