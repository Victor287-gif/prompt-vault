---
type: Optimization
title: "reciprocal：循环计算优化"
description: "reciprocal 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/docs/Reciprocal.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/tf_plugin/tensorflow_reciprocal_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_host/reciprocal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_host/reciprocal_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_kernel/reciprocal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/Reciprocal_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/test_reciprocal.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# reciprocal：循环计算优化

## 来源信息

- experience ID：`e68fab56e6fa735a`
- 源 schema version：`v2`
- 算子：`reciprocal`
- source iteration：`round_019_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.simplification`
- canonical family：`loop.simplification`
- 源记录时间：`2026-04-21T02:31:45`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Replace chunked copy loops with single DataCopy calls
~~~~

### Original summary

~~~~text
In Ascend C operators, using a single DataCopy call instead of chunked copy loops (while loops with CHUNK_SIZE=16) significantly improves performance. This optimization eliminates loop control overhead, increases vector utilization from 0.018255 to 0.035476, and reduces MTE2 utilization from 0.498083 to 0.01914. Apply this pattern when copying data between global memory and UB in CopyIn/CopyOut functions.
~~~~

### Original root mechanism

~~~~text
Chunked copy loops introduce unnecessary loop control overhead compared to single DataCopy calls, reducing vector utilization and increasing memory bandwidth pressure.
~~~~

### Original causal chain

~~~~text
Replacing chunked copy loop with single DataCopy call → reduces loop control overhead → increases vector utilization (0.018255→0.035476) and reduces MTE2 utilization (0.498083→0.01914) → improves L2 cache read hit rate (0.68%→93.75%) → reduces latency from 1347313.25μs to 692377.19μs.
~~~~

## 适用条件

- Copying data between global memory and UB in Ascend C operators
- Implementing CopyIn or CopyOut functions in KernelReciprocal::CopyIn
- When using DataCopy with LocalTensor<TYPE_X> and xGm pointers
- When processDataNum is known at compile time or runtime

## 失效边界

- When data must be processed in chunks due to UB size limitations
- When non-contiguous memory access patterns require chunked processing
- When using DataCopy with stride or offset patterns that require loop control

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- reciprocal/op_kernel/reciprocal.cpp

### 改动符号

- KernelReciprocal::CopyIn

### 代码签名

- AllocTensor
- DataCopy
- EnQue
- compile_time_specialization
- constexpr
- datacopy
- queue_api
- ub

### 原始 diff

- 源 diff SHA-256：`03c78858c9d44c147ed3c7ddf279a53a05718380f01c1415f8e269d5e4ca1781`

````diff
--- Before/op_kernel/reciprocal.cpp
+++ After/op_kernel/reciprocal.cpp
@@ -82,16 +82,7 @@
     __aicore__ inline void CopyIn(int32_t progress)
     {
         AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
-        // Use chunked copies instead of single DMA to increase overhead
-        constexpr int32_t CHUNK_SIZE = 16;
-        int32_t remaining = this->processDataNum;
-        int32_t offset = 0;
-        while (remaining > 0) {
-            int32_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
-            DataCopy(xLocal[offset], xGm[progress * this->ubPartDataNum + offset], chunk);
-            offset += chunk;
-            remaining -= chunk;
-        }
+        DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
         inQueueX.EnQue(xLocal);
     }
     __aicore__ inline void Compute(int32_t progress)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`1347313.25`
- fast duration us：`692377.1875`
- improvement ratio：`0.48610526356806777`
- machine confidence：`0.9`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 0.68 -> 93.75 (+13686.76%)
- aiv_utilization.mte2_ratio_mean: 0.01914 -> 0.498083 (+2502.31%)
- memory_ub.gm_to_ub_kb: 1.04994e+06 -> 4.1943e+06 (+299.48%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.74 -> 1.52 (+105.41%)
- memory_gm.aiv_main_mem_read_bw_gbps: 1.45 -> 2.97 (+104.83%)

### Affected metric groups

- cache_l2
- aiv_utilization
- memory_ub
- memory_gm

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_read_hit_rate_pct",
  "aiv_utilization.mte2_ratio_mean",
  "memory_ub.gm_to_ub_kb",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "memory_gm.aiv_main_mem_read_bw_gbps"
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

- Replace while loops with CHUNK_SIZE chunking in CopyIn functions with single DataCopy calls
- Change from: while (remaining > 0) { DataCopy(xLocal[offset], xGm[progress * this->ubPartDataNum + offset], chunk); offset += chunk; remaining -= chunk; }
- Change to: DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
- Apply to reciprocal.cpp:82-93 in KernelReciprocal::CopyIn function

## 检索线索

- DataCopy
- chunked copy
- loop overhead
- vector utilization
- MTE2
- L2 cache hit rate
- CopyIn
- reciprocal
- Ascend C
- memory bandwidth
