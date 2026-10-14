---
type: Optimization
title: "gelu：流水与并行优化"
description: "gelu 算子的流水与并行优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/docs/Gelu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/tf_plugin/tensorflow_gelu_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_host/gelu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_host/gelu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_kernel/gelu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/Gelu_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/test_gelu.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gelu：流水与并行优化

## 来源信息

- experience ID：`6890fdc0d999d831`
- 源 schema version：`v2`
- 算子：`gelu`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.parallelism.double.buffer`
- canonical family：`pipeline.parallelism.double.buffer`
- 源记录时间：`2026-04-18T21:53:57`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Enable double-buffering for pipeline overlap
~~~~

### Original summary

~~~~text
For Ascend C vector operators with significant GM-to-UB data transfer, setting BUFFER_NUM=2 enables double-buffering pipeline parallelism. This optimization overlaps data loading for the next tile with computation on the current tile, reducing vector wait stalls and improving vector unit utilization. Apply when the operator processes data in tiles and spends significant time waiting for memory transfers.
~~~~

### Original root mechanism

~~~~text
Setting BUFFER_NUM=2 enables double-buffering which overlaps data transfer with computation, hiding memory latency.
~~~~

### Original causal chain

~~~~text
Setting BUFFER_NUM=2 enables double-buffering -> reduces aiv_vec_wait_ratio_mean from 0.249847 to 0.003587 (-98.6%) -> increases vector utilization from 0.762023 to 0.980437 (+28.7%) -> reduces task_duration_us from 1326.286μs to 1007.38μs (-24.0%)
~~~~

## 适用条件

- Operator uses Ascend C vector processing with GM-to-UB data transfers
- Profile shows high aiv_vec_wait_ratio_mean (>0.1) indicating memory transfer stalls
- Vector unit utilization (aiv_utilization.vec_ratio_mean) is below 0.9
- Operator processes data in tiles with pipelined data transfer patterns
- Code uses BUFFER_NUM parameter in Ascend C kernel templates

## 失效边界

- Operator has insufficient computation to hide memory transfer latency
- Memory bandwidth is already saturated (gm_to_ub_bw_usage_rate_pct > 90%)
- Kernel uses single-buffer pattern with immediate data consumption
- Vector unit utilization is already near 1.0 (aiv_utilization.vec_ratio_mean > 0.95)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/gelu.cpp
- op_host/gelu.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- template
- ub

### 原始 diff

- 源 diff SHA-256：`a711d699d1253d4372e2de4c37f89b46023f3e1cc1f02a1521fd7710130f9309`

````diff
--- Before/op_host/gelu.cpp
+++ After/op_host/gelu.cpp
@@ -2,7 +2,7 @@
 #include "register/op_def_registry.h"
 #include "tiling/platform/platform_ascendc.h"

-const uint32_t BUFFER_NUM = 2;
+const uint32_t BUFFER_NUM = 1;
 const uint32_t BLOCK_SIZE = 32;
 const int32_t DATATYPE1 = 2;
 const int32_t DATATYPE2 = 4;

--- Before/op_kernel/gelu.cpp
+++ After/op_kernel/gelu.cpp
@@ -1,6 +1,6 @@
 #include "kernel_operator.h"
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 2;
+constexpr int32_t BUFFER_NUM = 1;

 template <typename T>
 class KernelGelu
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1326.286499`
- fast duration us：`1007.380127`
- improvement ratio：`0.2404505906080252`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.003587 -> 0.249847 (+6865.35%)
- aiv_utilization.mte2_ratio_mean: 0.582892 -> 0.223993 (-61.57%)
- aiv_utilization.mte3_ratio_mean: 0.22372 -> 0.102323 (-54.26%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.018667 -> 0.026248 (+40.61%)
- latency.task_duration_us: 1007.38 -> 1326.29 (+31.66%)

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
      "before": 0.003587,
      "after": 0.249847,
      "delta": 0.24626,
      "rel": 68.65347086701979
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.582892,
      "after": 0.223993,
      "delta": -0.35889899999999997,
      "rel": -0.6157212656890126
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.22372,
      "after": 0.102323,
      "delta": -0.121397,
      "rel": -0.542629179331307
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.018667,
      "after": 0.026248,
      "delta": 0.007581000000000001,
      "rel": 0.40611774789735905
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1007.38,
      "after": 1326.286,
      "delta": 318.90600000000006,
      "rel": 0.31656971549961294
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

- In Ascend C kernel templates (e.g., op_kernel/gelu.cpp), set BUFFER_NUM=2 instead of BUFFER_NUM=1
- Ensure both op_host and op_kernel files have consistent BUFFER_NUM values
- Verify the kernel uses tiled processing with separate load/compute/store phases
- Check profile metrics: aiv_vec_wait_ratio_mean should decrease significantly (<0.01)
- Monitor aiv_utilization.vec_ratio_mean increase (target >0.95)
- Validate that gm_to_ub_bw_usage_rate_pct increases due to better overlap

## 检索线索

- double-buffering
- pipeline overlap
- BUFFER_NUM
- AscendC
- vector wait
- memory latency hiding
- tiled processing
- GM-to-UB transfer
