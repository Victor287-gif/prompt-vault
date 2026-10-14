---
type: Optimization
title: "heaviside：计算优化"
description: "heaviside 算子的计算优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/docs/Heaviside.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/tf_plugin/tensorflow_add_custom_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_host/heaviside.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_host/heaviside_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_kernel/heaviside.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# heaviside：计算优化

## 来源信息

- experience ID：`012e476d8ad1cac7`
- 源 schema version：`v2`
- 算子：`heaviside`
- source iteration：`round_008_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`constant.extraction`
- canonical family：`constant.extraction`
- 源记录时间：`2026-04-20T20:21:43`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Extracted magic number 100*512 into named constant BUFFER_SIZE_ELEMENTS
~~~~

### Original summary

~~~~text
Extracting magic number 100*512 into a named compile-time constant BUFFER_SIZE_ELEMENTS improved performance by 5.2% despite being a structural refactoring. This optimization reduces scalar operation overhead and improves vector unit utilization in Ascend C kernels. Apply this technique when buffer size calculations involve compile-time arithmetic that affects loop unrolling and instruction scheduling.
~~~~

### Original root mechanism

~~~~text
Replacing hardcoded arithmetic expressions with compile-time constants can affect compiler optimization decisions and instruction scheduling, leading to unexpected performance changes in vectorized code.
~~~~

### Original causal chain

~~~~text
Replaced hardcoded 100*512 with BUFFER_SIZE_ELEMENTS constant → Reduced scalar ratio from 0.160621 to 0.085706 (-46.6%) → Improved vector ratio from 0.232722 to 0.215628 → Reduced total task duration from 1464.369μs to 1388.588μs
~~~~

## 适用条件

- Ascend C kernels use hardcoded arithmetic expressions for buffer size calculations
- Buffer size calculations involve compile-time constants like 100*512
- Kernel uses InitBuffer API with arithmetic expressions in size parameters
- Code contains repeated magic numbers in buffer initialization calls

## 失效边界

- Buffer sizes are already defined as named compile-time constants
- Arithmetic expressions are simple and unlikely to affect compiler optimizations
- Performance is already optimal and changes risk regression

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- heaviside/op_kernel/heaviside.cpp

### 改动符号

- BUFFER_SIZE_ELEMENTS

### 代码签名

- InitBuffer
- TBuf
- TQue
- bool
- bool_control
- compile_time_specialization
- constexpr
- queue_api
- tbuf
- template
- ub

### 原始 diff

- 源 diff SHA-256：`1023fa30105c27db608a26c0009832b10565bc327d6df38a8cd3507b45bab425`

````diff
--- Before/op_kernel/heaviside.cpp
+++ After/op_kernel/heaviside.cpp
@@ -2,6 +2,7 @@
 #include "kernel_operator.h"
 using namespace AscendC;
 constexpr uint16_t BufferNum = 1;
+constexpr uint32_t BUFFER_SIZE_ELEMENTS = 100 * 512;
 template <bool flag, bool valuesCountIsOne>
 class KernelHeaviside {
 public:
@@ -31,10 +32,10 @@
             pipe->InitBuffer(selMaskQueue, BufferNum, size >> 3);
             pipe->InitBuffer(outQueueY, BufferNum, spaceSize);
         } else {
-            pipe->InitBuffer(inQueueInput, BufferNum, 100 * 512 / BufferNum);
-            pipe->InitBuffer(inQueueValues, BufferNum, 100 * 512 / BufferNum);
-            pipe->InitBuffer(selMaskQueue, BufferNum, 100 * 512 / BufferNum / 8 / sizeof(DTYPE_VALUES));
-            pipe->InitBuffer(outQueueY, BufferNum, 100 * 512 / BufferNum);
+            pipe->InitBuffer(inQueueInput, BufferNum, BUFFER_SIZE_ELEMENTS / BufferNum);
+            pipe->InitBuffer(inQueueValues, BufferNum, BUFFER_SIZE_ELEMENTS / BufferNum);
+            pipe->InitBuffer(selMaskQueue, BufferNum, BUFFER_SIZE_ELEMENTS / BufferNum / 8 / sizeof(DTYPE_VALUES));
+            pipe->InitBuffer(outQueueY, BufferNum, BUFFER_SIZE_ELEMENTS / BufferNum);
         }
     }
     __aicore__ inline void Process_iter(uint32_t beginIndex, uint32_t iterSize) {
@@ -81,7 +82,7 @@
                 Process_iter(0, size);
             }
         } else {
-            uint32_t n_elements_per_iter = 50 * 512 / sizeof(DTYPE_VALUES) / BufferNum;
+            uint32_t n_elements_per_iter = BUFFER_SIZE_ELEMENTS / sizeof(DTYPE_VALUES) / BufferNum;
             uint16_t repeatTimes = size / n_elements_per_iter;
             uint16_t remainSize = size % n_elements_per_iter;
             for (uint16_t i = 0; i < repeatTimes; i++) {
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1464.369263`
- fast duration us：`1388.587769`
- improvement ratio：`0.05175026266581782`
- machine confidence：`0.8`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.002932 -> 0.006235 (+112.65%)
- aiv_utilization.scalar_ratio_mean: 0.085706 -> 0.160621 (+87.41%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.016693 -> 0.020846 (+24.88%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.020759 -> 0.025256 (+21.66%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 20.54 -> 18.22 (-11.30%)

### Affected metric groups

- arithmetic
- aiv_utilization
- pipeline_stalls
- memory_ub

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_misc_ratio_mean",
  "aiv_utilization.scalar_ratio_mean",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean",
  "memory_ub.gm_to_ub_bw_usage_rate_pct"
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

- Identify hardcoded arithmetic expressions in heaviside/op_kernel/heaviside.cpp InitBuffer calls
- Extract expressions like 100*512 into named constexpr constants (e.g., BUFFER_SIZE_ELEMENTS)
- Replace all occurrences of the magic number with the named constant
- Ensure constant is defined at appropriate scope (global or class-level constexpr)
- Verify performance impact by comparing scalar_ratio_mean and vec_ratio_mean metrics

## 检索线索

- constant extraction
- magic number
- buffer size
- InitBuffer
- AscendC
- compile-time constant
- heaviside
- scalar ratio
- vector ratio
- performance sensitivity
