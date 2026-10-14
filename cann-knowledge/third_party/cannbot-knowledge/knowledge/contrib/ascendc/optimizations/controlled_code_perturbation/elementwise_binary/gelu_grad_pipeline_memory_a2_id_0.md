---
type: Optimization
title: "gelu_grad：流水与内存优化"
description: "gelu_grad 算子的流水与内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
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

- experience ID：`015a583d03ce7cb9`
- 源 schema version：`v2`
- 算子：`gelu_grad`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.double_buffer`
- canonical family：`pipeline.break`
- 源记录时间：`2026-04-18T21:55:40`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Maintain double buffering for pipeline efficiency
~~~~

### Original summary

~~~~text
Setting BUFFER_NUM=2 in Ascend C kernels enables double buffering, allowing concurrent data transfer and computation stages. This optimization reduces pipeline stalls by overlapping memory operations with computation, improving vector unit utilization from 76.7% to 97.5%. Apply this technique to memory-bound operators with regular data access patterns where computation can be overlapped with data movement.
~~~~

### Original root mechanism

~~~~text
Using BUFFER_NUM=2 enables concurrent data transfer and computation through double buffering, hiding memory latency and improving pipeline efficiency.
~~~~

### Original causal chain

~~~~text
Setting BUFFER_NUM=2 enables double buffering -> reduces pipeline stalls (aiv_vec_wait_ratio_mean from 22.7% to 0.1%) -> improves vector unit utilization (vec_ratio_mean from 76.7% to 97.5%) -> reduces task duration by 18.8% (2159.6μs to 1753.2μs)
~~~~

## 适用条件

- Ascend C kernel uses data transfer APIs like DataCopy
- Operator has regular memory access patterns with predictable data movement
- Computation workload is sufficient to overlap with memory transfers
- Memory bandwidth is not fully saturated (GM-to-UB bandwidth usage < 10%)
- Vector unit utilization is below 80% indicating pipeline inefficiency

## 失效边界

- Operator is compute-bound with vector utilization already > 95%
- Memory bandwidth is already saturated (GM-to-UB bandwidth usage > 90%)
- Data access patterns are irregular or unpredictable
- UB buffer size constraints prevent double buffering
- Kernel has insufficient computation to overlap with memory transfers

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/gelu_grad.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- template
- ub

### 原始 diff

- 源 diff SHA-256：`9daf605d01500e562ef01addeab996aadd51b2c00d66c83827f86be6198d7bb9`

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
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2159.603271`
- fast duration us：`1753.195068`
- improvement ratio：`0.18818651020648502`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.00114 -> 0.227057 (+19817.28%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.045744 -> 0.007077 (-84.53%)
- aiv_utilization.mte2_ratio_mean: 0.358059 -> 0.208131 (-41.87%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.307475 -> 0.190623 (-38.00%)
- aiv_utilization.mte3_ratio_mean: 0.174682 -> 0.12238 (-29.94%)

### Affected metric groups

- pipeline_stalls
- aiv_utilization
- latency
- critical_path

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "aiv_utilization.mte2_ratio_mean",
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean",
  "latency.task_duration_us",
  "critical_path.aiv_vector0_time_us_mean",
  "memory_ub.ub_read_bw_vector_gbps",
  "aiv_utilization.vec_ratio_mean"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.5454,
  "overall_deviation": 0.21816,
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

- In Ascend C kernel files (e.g., op_kernel/gelu_grad.cpp), set BUFFER_NUM=2 in the template class definition
- Ensure DataCopy operations use different buffer indices (0 and 1) for ping-pong buffering
- Structure kernel loop to alternate between buffer indices for concurrent transfer/compute
- Verify pipeline efficiency by monitoring aiv_vec_wait_ratio_mean (< 1% indicates good pipelining)
- Check vector unit utilization (vec_ratio_mean) improves to > 90% after optimization

## 检索线索

- double_buffering
- BUFFER_NUM
- pipeline_stalls
- vector_utilization
- AscendC
- DataCopy
- memory_latency_hiding
- concurrent_transfer_compute
- ping_pong_buffering
- pipeline_efficiency
