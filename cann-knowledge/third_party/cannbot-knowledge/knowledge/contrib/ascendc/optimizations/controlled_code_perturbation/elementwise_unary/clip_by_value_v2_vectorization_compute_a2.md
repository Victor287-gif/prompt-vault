---
type: Optimization
title: "clip_by_value_v2：向量化计算优化"
description: "clip_by_value_v2 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
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
# clip_by_value_v2：向量化计算优化

## 来源信息

- experience ID：`df86f6a5ba35a29c`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_002_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vectorization.break.scalar.loop`
- canonical family：`vectorization.break.scalar.loop`
- 源记录时间：`2026-04-20T14:09:58`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Vectorized Duplicate for Broadcast Operations
~~~~

### Original summary

~~~~text
Replace scalar loops with AscendC::Duplicate for broadcast operations to enable SIMD vectorization. This optimization improves performance by reducing pipeline stalls and increasing vector unit utilization. Apply this technique when broadcasting scalar values to local tensors in Ascend C kernels.
~~~~

### Original root mechanism

~~~~text
Using AscendC::Duplicate for broadcast operations enables SIMD vectorization and reduces per-element overhead compared to scalar loops.
~~~~

### Original causal chain

~~~~text
Replacing scalar loops with AscendC::Duplicate -> increases vector unit utilization from 0.002784 to 0.200725 -> reduces MTE wait ratios from ~0.995 to near zero -> decreases latency from 74,956us to 1,607us (46.6x speedup)
~~~~

## 适用条件

- Broadcasting scalar values to local tensors in Ascend C kernels
- Using AscendC::LocalTensor with scalar initialization patterns
- Operations where the same value needs to be duplicated across tensor elements
- When performance profiling shows high scalar_ratio_mean (>0.9) and low vector_ratio_mean (<0.01)
- When pipeline_stalls.aiv_mte2_wait_ratio_mean or aiv_mte3_wait_ratio_mean are high (>0.9)

## 失效边界

- When broadcasting requires per-element conditional logic
- When values differ across tensor positions
- When AscendC::Duplicate API is not available for the data type
- When tensor size is very small (<16 elements) where loop overhead may be acceptable

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

- AllocTensor
- FreeTensor
- SetValue
- contiguous_access
- queue_api
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`d81032d271fef12cd486ce442fa1ea794ce96ecae3949515d6cf031cd0fa1da9`

````diff
--- Before/op_kernel/clip_by_value_v2.cpp
+++ After/op_kernel/clip_by_value_v2.cpp
@@ -81,8 +81,11 @@
         AscendC::LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
         AscendC::LocalTensor<TYPE_CLIP_VALUE_MIN> min = Q_min.Get<TYPE_CLIP_VALUE_MIN>();
         AscendC::LocalTensor<TYPE_CLIP_VALUE_MAX> max = Q_max.Get<TYPE_CLIP_VALUE_MAX>();
-        AscendC::Duplicate(min, this->clip_value_min, this->processDataNum);
-        AscendC::Duplicate(max, this->clip_value_max, this->processDataNum);
+        // Degradation: Replace vectorized Duplicate with scalar loop to break vectorization
+        for (uint64_t i = 0; i < this->processDataNum; ++i) {
+            min.SetValue(i, this->clip_value_min);
+            max.SetValue(i, this->clip_value_max);
+        }
         AscendC::Max(y, x, min, this->processDataNum);
         AscendC::Min(y, y, max, this->processDataNum);
         Q_x.FreeTensor(x);
@@ -198,8 +201,11 @@
         AscendC::Cast(x_fp32,x_bf16,AscendC::RoundMode::CAST_NONE, this->processDataNum);
         AscendC::LocalTensor<float> min = Q_min_fp32.Get<float>();
         AscendC::LocalTensor<float> max = Q_max_fp32.Get<float>();
-        AscendC::Duplicate(min, this->clip_value_min, this->processDataNum);
-        AscendC::Duplicate(max, this->clip_value_max, this->processDataNum);
+        // Degradation: Replace vectorized Duplicate with scalar loop to break vectorization
+        for (uint64_t i = 0; i < this->processDataNum; ++i) {
+            min.SetValue(i, this->clip_value_min);
+            max.SetValue(i, this->clip_value_max);
+        }
         AscendC::Max(y_fp32, x_fp32, min, this->processDataNum);
         AscendC::Min(y_fp32, y_fp32, max, this->processDataNum);
         AscendC::Cast(y_bf16, y_fp32, AscendC::RoundMode::CAST_RINT, this->processDataNum);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`74956.734375`
- fast duration us：`1606.832153`
- improvement ratio：`0.9785632049421844`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1596.51 -> 74946.3 (+4594.38%)
- latency.task_duration_us: 1606.83 -> 74956.7 (+4564.88%)
- aiv_utilization.scalar_ratio_mean: 0.105417 -> 0.996614 (+845.40%)
- memory_ub.ub_write_bw_scalar_gbps: 0 -> 4.4476 (+100.00%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.995324 -> 0 (-100.00%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1596.509,
      "after": 74946.272,
      "delta": 73349.76299999999,
      "rel": 45.94384560312531
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1606.832,
      "after": 74956.734,
      "delta": 73349.902,
      "rel": 45.648768508468834
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.105417,
      "after": 0.996614,
      "delta": 0.891197,
      "rel": 8.454015955680774
    }
  },
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.0,
      "after": 4.4476,
      "delta": 4.4476,
      "rel": 1.0
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 0.995324,
      "after": 0.0,
      "delta": -0.995324,
      "rel": -1.0
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

- In clip_by_value_v2.cpp:81-86 and 198-203, replace scalar loops with AscendC::Duplicate(min, scalar_value, count)
- Use AscendC::Duplicate for both clip_value_min and clip_value_max initialization
- Ensure the scalar value is properly typed to match the tensor element type
- Verify the count parameter matches this->processDataNum or the tensor size
- Check that vector unit utilization (vec_ratio_mean) increases after optimization
- Monitor reduction in MTE wait ratios (aiv_mte2_wait_ratio_mean, aiv_mte3_wait_ratio_mean)

## 检索线索

- AscendC::Duplicate
- vectorization
- broadcast
- scalar loop
- clip_by_value_v2
- pipeline stall
- MTE wait
- vector unit utilization
- scalar ratio
- SIMD
