---
type: Optimization
title: "rsqrt：流水优化"
description: "rsqrt 算子的流水优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
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
# rsqrt：流水优化

## 来源信息

- experience ID：`3076a97bd06639db`
- 源 schema version：`v2`
- 算子：`rsqrt`
- source iteration：`round_005_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.depth.reduction`
- canonical family：`pipeline.depth.reduction`
- 源记录时间：`2026-04-20T23:17:08`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce pipeline depth by decreasing buffer count
~~~~

### Original summary

~~~~text
For Ascend C vector operators with simple memory access patterns, reducing BUFFER_NUM from 2 to 1 can improve performance by eliminating pipeline stalls between memory transfer and compute stages. This optimization is particularly effective for memory-bound operators where the overhead of managing multiple buffers outweighs the benefits of pipelining. Apply this when profiling shows high aiv_mte2_wait_ratio_mean and aiv_mte3_wait_ratio_mean indicating memory transfer unit stalls.
~~~~

### Original root mechanism

~~~~text
Reducing BUFFER_NUM from 2 to 1 eliminates pipelining overhead between copy-in, compute, and copy-out operations in Ascend C vector operators
~~~~

### Original causal chain

~~~~text
BUFFER_NUM reduction from 2 to 1 → reduced mte2_ratio_mean from 0.489656 to 0.961552 and mte3_ratio_mean from 0.282189 to 0.800975 → decreased aiv_vector0_time_us_mean from 1363.387μs to 1190.095μs → improved task_duration_us from 1374.527μs to 1200.324μs
~~~~

## 适用条件

- Ascend C vector operators with BUFFER_NUM=2 configuration
- Profiling shows aiv_mte2_wait_ratio_mean > 0.99 and aiv_mte3_wait_ratio_mean > 0.99
- Memory bandwidth utilization is low (gm_to_ub_bw_usage_rate_pct < 10%)
- Operator has simple memory access patterns without complex data dependencies

## 失效边界

- Operators with complex data dependencies requiring double-buffering
- When aiv_vec_wait_ratio_mean is already low (< 0.3)
- Memory-bound operators with high gm_to_ub_bw_usage_rate_pct (> 50%)
- Operators requiring overlapping computation with memory transfers

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- rsqrt/op_kernel/rsqrt.cpp
- rsqrt/op_host/rsqrt.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- template

### 原始 diff

- 源 diff SHA-256：`0cfe19308ec1c2c25953206c65d8318405e860853879cc6a9bfb730b1e39fa3f`

````diff
--- Before/op_host/rsqrt.cpp
+++ After/op_host/rsqrt.cpp
@@ -5,7 +5,7 @@

 namespace optiling {
 const uint32_t BLOCK_SIZE = 32;
-const uint32_t BUFFER_NUM = 2;
+const uint32_t BUFFER_NUM = 1;

 static ge::graphStatus TilingFunc(gert::TilingContext* context)
 {

--- Before/op_kernel/rsqrt.cpp
+++ After/op_kernel/rsqrt.cpp
@@ -1,6 +1,6 @@
 #include "kernel_operator.h"
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 2;
+constexpr int32_t BUFFER_NUM = 1;

 template <typename TYPE_X>
 class KernelRsqrt
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1374.527466`
- fast duration us：`1200.323975`
- improvement ratio：`0.126737002576564`
- machine confidence：`0.85`

### Profile evidence

- aiv_utilization.mte3_ratio_mean: 0.800975 -> 0.282189 (-64.77%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.008229 -> 0.013557 (+64.75%)
- aiv_utilization.mte2_ratio_mean: 0.961552 -> 0.489656 (-49.08%)
- aiv_utilization.scalar_ratio_mean: 0.111553 -> 0.094069 (-15.67%)
- critical_path.aiv_vector0_time_us_mean: 1190.1 -> 1363.39 (+14.56%)

### Affected metric groups

- aiv_utilization
- pipeline_stalls
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.800975,
      "after": 0.282189,
      "delta": -0.518786,
      "rel": -0.6476931240051187
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.008229,
      "after": 0.013557,
      "delta": 0.005327999999999999,
      "rel": 0.6474662777980312
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.961552,
      "after": 0.489656,
      "delta": -0.471896,
      "rel": -0.49076493002978516
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.111553,
      "after": 0.094069,
      "delta": -0.017484,
      "rel": -0.156732674154886
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1190.095,
      "after": 1363.387,
      "delta": 173.29199999999992,
      "rel": 0.14561190493195914
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.81413,
  "overall_deviation": 0.325652,
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

- 1. Check op_kernel/*.cpp files for BUFFER_NUM constant definitions
- 2. Reduce BUFFER_NUM from 2 to 1 in both op_host/*.cpp and op_kernel/*.cpp files
- 3. Verify the change affects both host-side TilingFunc and kernel-side template class definitions
- 4. Profile the operator and validate reduction in aiv_mte2_wait_ratio_mean and aiv_mte3_wait_ratio_mean
- 5. Monitor aiv_vector0_time_us_mean for expected latency reduction of 10-15%

## 检索线索

- BUFFER_NUM
- pipeline depth
- Ascend C
- vector operator
- mte2_wait_ratio
- mte3_wait_ratio
- scalar_overhead_bound
- memory transfer stalls
- rsqrt
- math operator
