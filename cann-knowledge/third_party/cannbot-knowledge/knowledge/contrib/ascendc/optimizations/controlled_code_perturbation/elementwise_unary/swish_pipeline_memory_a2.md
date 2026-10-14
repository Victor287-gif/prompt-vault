---
type: Optimization
title: "swish：流水与内存优化"
description: "swish 算子的流水与内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/docs/Swish.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/framework/tf_plugin/tensorflow_swish_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/op_host/swish.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/op_host/swish_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/op_kernel/swish.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/Swish_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/test_swish.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# swish：流水与内存优化

## 来源信息

- experience ID：`c3036989ae0afef4`
- 源 schema version：`v2`
- 算子：`swish`
- source iteration：`round_002_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.buffer.reduction`
- canonical family：`pipeline.buffer.reduction`
- 源记录时间：`2026-04-20T23:30:03`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Disable double buffering to eliminate pipeline overhead
~~~~

### Original summary

~~~~text
Setting BUFFER_NUM=1 instead of BUFFER_NUM=2 in Ascend C kernel code reduces scalar overhead by eliminating double buffering. This optimization applies to memory-bound operators where scalar instruction overhead dominates, particularly when memory transfer bandwidth is not fully utilized. The change reduces MTE utilization from 94.6% to 43.2% for MTE2 and 78.6% to 25.0% for MTE3 while improving vector unit utilization from 55.1% to 63.1%.
~~~~

### Original root mechanism

~~~~text
Reducing BUFFER_NUM from 2 to 1 eliminates overlapping memory transfers and computation, reducing scalar overhead and improving pipeline efficiency.
~~~~

### Original causal chain

~~~~text
apply pipeline.buffer.reduction in op_kernel/swish.cpp -> improve aiv_utilization.mte3_ratio_mean -> 14.3% latency improvement
~~~~

## 适用条件

- Ascend C kernels using BUFFER_NUM=2 for double buffering
- Memory-bound operators with low GM bandwidth utilization (<15%)
- High MTE utilization (>70%) with low vector unit utilization (<60%)
- Operators where scalar instruction overhead limits performance
- Tile-based computations with separate copy-compute-copy stages

## 失效边界

- Compute-bound operators with high arithmetic intensity
- Memory bandwidth utilization >50% of theoretical maximum
- When overlapping memory transfers with computation improves performance
- When vector unit utilization is already high (>80%)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/swish.cpp

### 改动符号

- KernelSwish

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`bcf6fb00b53ef995133e34981e0b81672aa85edf02f8f15c2e554f8733d470ae`

````diff
--- Before/op_kernel/swish.cpp
+++ After/op_kernel/swish.cpp
@@ -3,7 +3,7 @@
  */
 #include "kernel_operator.h"
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 2;
+constexpr int32_t BUFFER_NUM = 1;

 class KernelSwish
 {
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1415.728271`
- fast duration us：`1212.964233`
- improvement ratio：`0.1432224263324046`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.mte3_ratio_mean: 0.786202 -> 0.250134 (-68.18%)
- aiv_utilization.mte2_ratio_mean: 0.946392 -> 0.431819 (-54.37%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.017583 -> 0.025433 (+44.65%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.386275 -> 0.468772 (+21.36%)
- critical_path.aiv_vector0_time_us_mean: 1201.42 -> 1409.07 (+17.28%)

### Affected metric groups

- aiv_utilization
- pipeline_stalls
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "metric": "aiv_utilization.mte3_ratio_mean",
    "before": 0.786202,
    "after": 0.250134,
    "delta": -0.536068,
    "rel": -0.6818451237722621
  },
  {
    "metric": "aiv_utilization.mte2_ratio_mean",
    "before": 0.946392,
    "after": 0.431819,
    "delta": -0.514573,
    "rel": -0.5437207837767013
  },
  {
    "metric": "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
    "before": 0.017583,
    "after": 0.025433,
    "delta": 0.00785,
    "rel": 0.4464539612125348
  },
  {
    "metric": "pipeline_stalls.aiv_vec_wait_ratio_mean",
    "before": 0.386275,
    "after": 0.468772,
    "delta": 0.08249700000000004,
    "rel": 0.21357064267684953
  },
  {
    "metric": "critical_path.aiv_vector0_time_us_mean",
    "before": 1201.419,
    "after": 1409.066,
    "delta": 207.64699999999993,
    "rel": 0.17283478952804968
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

- Check MTE utilization metrics in profiling: if mte2_ratio_mean > 70% and mte3_ratio_mean > 70%
- Examine vector unit utilization: if vec_ratio_mean < 60%
- Review BUFFER_NUM setting in kernel code (e.g., op_kernel/swish.cpp)
- Change constexpr int32_t BUFFER_NUM = 2; to constexpr int32_t BUFFER_NUM = 1;
- Verify GM bandwidth usage rate is low (<20%) before applying
- Test with single buffer to confirm performance improvement

## 检索线索

- BUFFER_NUM
- double_buffering
- pipeline_overhead
- scalar_bound
- MTE_utilization
- AscendC
- memory_transfer
- tile_pipeline
- copy_compute_copy
- vector_utilization
