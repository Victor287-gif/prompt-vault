---
type: Optimization
title: "gelu_grad：流水与内存优化"
description: "gelu_grad 算子的流水与内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/docs/GeluGrad.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/tf_plugin/tensorflow_gelu_grad_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_host/gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_host/gelu_grad_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_kernel/gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/GeluGrad_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/test_gelu_grad.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gelu_grad：流水与内存优化

## 来源信息

- experience ID：`5bddf4a127597a35`
- 源 schema version：`v2`
- 算子：`gelu_grad`
- source iteration：`round_007_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.buffer.reduction`
- canonical family：`pipeline.buffer.reduction`
- 源记录时间：`2026-04-20T20:09:56`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce double buffering to single buffering
~~~~

### Original summary

~~~~text
Changing BUFFER_NUM from 2 to 1 in Ascend C operators eliminates double-buffering overhead, which reduces pipeline stalls and improves vector unit utilization. Apply this optimization when vector wait ratio is high (>0.2) and MTE utilization is low, indicating excessive pipeline synchronization overhead.
~~~~

### Original root mechanism

~~~~text
Reducing BUFFER_NUM from 2 to 1 eliminates pipelining overhead between data transfer and computation stages, reducing vector wait stalls and improving vector unit utilization
~~~~

### Original causal chain

~~~~text
BUFFER_NUM reduced from 2 to 1 → vector wait ratio decreased from 0.227 to 0.001 (199x reduction) → vector utilization increased from 0.768 to 0.975 → latency improved from 2156us to 1753us (18.7% speedup)
~~~~

## 适用条件

- Vector wait ratio (aiv_vec_wait_ratio_mean) > 0.2
- MTE utilization (mte2_ratio_mean, mte3_ratio_mean) < 0.4
- Vector unit utilization (vec_ratio_mean) < 0.8
- Operator uses TQue/DeQue/AllocTensor pipeline patterns with BUFFER_NUM > 1
- Memory bandwidth utilization (gm_to_ub_bw_usage_rate_pct) < 10%

## 失效边界

- Memory bandwidth utilization (gm_to_ub_bw_usage_rate_pct) > 50%
- Vector wait ratio (aiv_vec_wait_ratio_mean) < 0.05
- Operator already has BUFFER_NUM = 1
- Memory-bound workloads where double buffering helps hide memory latency

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- gelu_grad/op_kernel/gelu_grad.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- AllocTensor
- BUFFER_NUM
- DeQue
- TQue
- compile_time_specialization
- constexpr
- queue_api
- template
- ub

### 原始 diff

- 源 diff SHA-256：`e2fa93fadf81a446f245b11543cd6fc1b2b00c26fb40a9f482f4e9f9c8d40ddf`

````diff
--- Before/op_kernel/gelu_grad.cpp
+++ After/op_kernel/gelu_grad.cpp
@@ -11,7 +11,7 @@
       op.Process();                                                                 \
   }while(0)
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 2;
+constexpr int32_t BUFFER_NUM = 1;

 template <typename TYPE_DY, typename TYPE_X, typename TYPE_Z>
 class KernelGeluGrad
@@ -91,13 +91,13 @@
         LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
         LocalTensor<TYPE_DY> dyLocal = inQueueDY.DeQue<TYPE_DY>();
         LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();
-        LocalTensor<float> fun = funBuffer.Get<float>();
-        LocalTensor<float> tmp = tmpBuffer.Get<float>();
-        const float COEFF0 = -0.0713548162726002527220f;
-        const float COEFF1 = -1.595769121605730711759f;
-        const float COEFF2 = 0.2140644488178007f;
-        const float COEFF3 = 1.595769121605730711759f;
-        const float COEFF4 = 1.0f;
+         LocalTensor<float> fun = funBuffer.Get<float>();
+         LocalTensor<float> tmp = tmpBuffer.Get<float>();
+         const float COEFF0 = -0.0713548162726002527220f;
+         const float COEFF1 = -1.595769121605730711759f;
+         const float COEFF2 = 0.2140644488178007f;
+         const float COEFF3 = 1.595769121605730711759f;
+         const float COEFF4 = 1.0f;
         if constexpr (!std::is_same_v<T, float32_t>)
         {
             LocalTensor<float> p1 = tmp1.Get<float>(); // xlocal
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2156.843018`
- fast duration us：`1753.215088`
- improvement ratio：`0.18713829733156784`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.001132 -> 0.227171 (+19968.11%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.045743 -> 0.007083 (-84.52%)
- aiv_utilization.mte2_ratio_mean: 0.351132 -> 0.207508 (-40.90%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.307469 -> 0.190785 (-37.95%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.261726 -> 0.183702 (-29.81%)

### Affected metric groups

- pipeline_stalls
- aiv_utilization
- latency
- critical_path

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.001132,
      "after": 0.227171,
      "delta": 0.22603900000000002,
      "rel": 199.68109540636044
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.045743,
      "after": 0.007083,
      "delta": -0.03866,
      "rel": -0.84515663598802
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.351132,
      "after": 0.207508,
      "delta": -0.143624,
      "rel": -0.4090313614253329
    }
  },
  {
    "pipeline_stalls.aiv_vec_total_cflt_ratio_mean": {
      "before": 0.307469,
      "after": 0.190785,
      "delta": -0.11668399999999998,
      "rel": -0.3794984209790255
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.261726,
      "after": 0.183702,
      "delta": -0.07802400000000001,
      "rel": -0.29811329405561543
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

- 1. Locate BUFFER_NUM definition in operator kernel file (e.g., gelu_grad/op_kernel/gelu_grad.cpp)
- 2. Change constexpr int32_t BUFFER_NUM = 2 to constexpr int32_t BUFFER_NUM = 1
- 3. Verify the operator uses TQue/DeQue/AllocTensor pipeline pattern
- 4. Recompile and profile to check if vector wait ratio decreases and vector utilization increases
- 5. Monitor aiv_vec_wait_ratio_mean reduction and vec_ratio_mean increase as success indicators

## 检索线索

- BUFFER_NUM
- double buffering
- pipeline stalls
- vector wait ratio
- AscendC
- TQue
- DeQue
- AllocTensor
- gelu_grad
- pipeline.buffer.reduction
