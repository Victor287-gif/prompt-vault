---
type: Optimization
title: "rsqrt：流水与内存优化"
description: "rsqrt 算子的流水与内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/docs/Rsqrt.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/tf_plugin/tensorflow_rsqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_host/rsqrt.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_host/rsqrt_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_kernel/rsqrt.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/Rsqrt_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/test_rsqrt.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# rsqrt：流水与内存优化

## 来源信息

- experience ID：`41864d3fa851c840`
- 源 schema version：`v2`
- 算子：`rsqrt`
- source iteration：`round_017_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`buffering.ub_double_buffer`
- canonical family：`constant.change`
- 源记录时间：`2026-04-21T00:44:44`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Double-buffered UB queue optimization
~~~~

### Original summary

~~~~text
Using BUFFER_NUM=2 instead of BUFFER_NUM=1 enables double-buffering for the unified buffer (UB) queue in Ascend C kernels. This optimization hides memory access latency by overlapping computation with data transfers, reducing vector wait ratios from 45.3% to 27.8% and improving MTE utilization from 39.7%/25.0% to 91.5%/61.1%. Apply this optimization when kernel performance is limited by memory access latency rather than compute intensity.
~~~~

### Original root mechanism

~~~~text
Changing compile-time constant BUFFER_NUM from 1 to 2 enables double-buffering of the UB queue, reducing pipeline stalls and improving memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
structural change to constant.change in rsqrt/op_kernel/rsqrt.cpp -> unexpected shift in pipeline_stalls.aiv_vec_wait_ratio_mean -> 25.5% latency impact
~~~~

## 适用条件

- Kernel shows high aiv_vec_wait_ratio_mean (>0.3) indicating pipeline stalls
- MTE utilization is low (<0.5) despite sufficient compute work
- Memory bandwidth usage is below theoretical maximum
- Kernel uses Ascend C with UB queue for data transfers
- BUFFER_NUM is set to 1 in the kernel code

## 失效边界

- Kernel is compute-bound with high vector utilization (>0.8)
- Memory bandwidth is already saturated (>90% of theoretical max)
- UB queue size is already optimal for the data access pattern
- Kernel has very small data transfers that fit in single buffer

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- rsqrt/op_kernel/rsqrt.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- template
- ub

### 原始 diff

- 源 diff SHA-256：`2d42727f6df4fc365df84868b60e3efa2392de29c86bf8e1972140fbddb4b5fc`

````diff
--- Before/op_kernel/rsqrt.cpp
+++ After/op_kernel/rsqrt.cpp
@@ -1,6 +1,6 @@
 #include "kernel_operator.h"
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 1;
+constexpr int32_t BUFFER_NUM = 2;

 template <typename TYPE_X>
 class KernelRsqrt
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1729.69458`
- fast duration us：`1287.945801`
- improvement ratio：`0.25539120264804205`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.278325 -> 0.452957 (+62.74%)
- aiv_utilization.mte3_ratio_mean: 0.610588 -> 0.250039 (-59.05%)
- aiv_utilization.mte2_ratio_mean: 0.914793 -> 0.397479 (-56.55%)
- latency.task_duration_us: 1287.95 -> 1729.69 (+34.30%)
- critical_path.aiv_vector0_time_us_mean: 1281.01 -> 1716.82 (+34.02%)

### Affected metric groups

- pipeline_stalls
- aiv_utilization
- latency
- critical_path

### Changed metrics top

~~~~json
[
  {
    "metric": "pipeline_stalls.aiv_vec_wait_ratio_mean",
    "before": 0.278325,
    "after": 0.452957,
    "delta": 0.174632,
    "rel": 0.6274391448845774
  },
  {
    "metric": "aiv_utilization.mte3_ratio_mean",
    "before": 0.610588,
    "after": 0.250039,
    "delta": -0.360549,
    "rel": -0.5904947362214783
  },
  {
    "metric": "aiv_utilization.mte2_ratio_mean",
    "before": 0.914793,
    "after": 0.397479,
    "delta": -0.5173139999999999,
    "rel": -0.565498424233679
  },
  {
    "metric": "latency.task_duration_us",
    "before": 1287.946,
    "after": 1729.695,
    "delta": 441.749,
    "rel": 0.34298720598534416
  },
  {
    "metric": "critical_path.aiv_vector0_time_us_mean",
    "before": 1281.011,
    "after": 1716.824,
    "delta": 435.8130000000001,
    "rel": 0.3402101933550923
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

- Locate BUFFER_NUM definition in Ascend C kernel files (typically in op_kernel/*.cpp)
- Change constexpr int32_t BUFFER_NUM = 1 to constexpr int32_t BUFFER_NUM = 2
- Verify kernel still compiles without errors
- Profile before/after to confirm reduced aiv_vec_wait_ratio_mean and increased MTE utilization
- Check that memory bandwidth utilization increases appropriately

## 检索线索

- BUFFER_NUM
- double-buffering
- UB queue
- Ascend C
- pipeline stalls
- MTE utilization
- memory bandwidth
- vector wait ratio
- unified buffer
- compile-time constant
