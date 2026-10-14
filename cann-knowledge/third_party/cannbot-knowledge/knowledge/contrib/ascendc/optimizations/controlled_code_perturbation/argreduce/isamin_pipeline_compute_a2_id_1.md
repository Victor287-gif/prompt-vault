---
type: Optimization
title: "isamin：流水与计算优化"
description: "isamin 算子的流水与计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.reduction", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/docs/aclnnIsamin.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/op_host/isamin.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/op_host/isamin_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/op_kernel/isamin.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/op_kernel/isamin_aiv.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamin/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# isamin：流水与计算优化

## 来源信息

- experience ID：`3dfe1f31039059cc`
- 源 schema version：`v2`
- 算子：`isamin`
- source iteration：`round_013_refactor`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.unnecessary.sync`
- canonical family：`remove.unnecessary.sync`
- 源记录时间：`2026-04-20T22:06:43`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing unnecessary V_S event synchronization in vector processing loops
~~~~

### Original summary

~~~~text
Avoid adding explicit V_S (Vector to Scalar) event synchronization in vector processing loops unless required by data dependencies. This synchronization forces serialization between vector and scalar operations, reducing MTE2 utilization by 57.1% and increasing vector wait ratio by 56.0%. Apply this optimization when vector operations can proceed independently without waiting for scalar completion.
~~~~

### Original root mechanism

~~~~text
Explicit V_S event synchronization in vector processing loops introduces pipeline stalls by forcing serialization between vector operations, reducing MTE2 utilization from 77.3% to 33.2%.
~~~~

### Original causal chain

~~~~text
Removed explicit V_S event synchronization in Isamin::SingleProcess -> MTE2 utilization increased from 33.2% to 77.3% and vector wait ratio decreased from 72.4% to 31.8% -> Task duration improved by 15.9% (1686.4μs to 1418.6μs)
~~~~

## 适用条件

- Vector processing loops using AscendC::SetMaskNorm() and AscendC::ResetMask()
- Operations with independent vector and scalar computation phases
- Code patterns using GetTPipePtr()->FetchEventID(HardEvent::V_S) for synchronization
- MTE2 utilization below 40% with high vector wait ratios (>50%)

## 失效边界

- Actual data dependencies require vector-to-scalar synchronization
- Scalar operations depend on vector computation results
- Memory consistency requires explicit synchronization between vector and scalar units

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- isamin/op_kernel/isamin.cpp

### 改动符号

- Isamin::SingleProcess

### 代码签名

- SetFlag
- WaitFlag
- pipebarrier

### 原始 diff

- 源 diff SHA-256：`ed82164b790bffc9c5197e7eadec4bc4c173d192bc4686cb8f61cf799454de80`

````diff
--- Before/op_kernel/isamin.cpp
+++ After/op_kernel/isamin.cpp
@@ -364,11 +364,6 @@
     AscendC::SetMaskNorm();
     AscendC::ResetMask();

-    // Add unnecessary memory barrier synchronization to degrade performance
-    event_t unnecessaryEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
-    SetFlag<HardEvent::V_S>(unnecessaryEvent);
-    WaitFlag<HardEvent::V_S>(unnecessaryEvent);
-
     // 如果是第63次（k=62）或者是最后一次,本循环内马上就要进行压缩，此时必须把本次结果取出来参与压缩，不能等下次再处理；
     if (((k + 1) % DEAL_TIMES_EACH_CORE_REDUCE == 0 && k > 0) || k == this->dealTimesEachCore - 1) {
         GetReduceMinCount(k);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`1686.393677`
- fast duration us：`1418.568359`
- improvement ratio：`0.15881541875586616`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.mte2_ratio_mean: 0.773086 -> 0.33163 (-57.10%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.723704 -> 0.318254 (-56.02%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0004 -> 0.0003 (-25.00%)
- critical_path.aiv_vector0_time_us_mean: 1413.8 -> 1681.87 (+18.96%)
- latency.task_duration_us: 1418.57 -> 1686.39 (+18.88%)

### Affected metric groups

- aiv_utilization
- pipeline_stalls
- memory_ub
- critical_path

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.773086,
      "after": 0.33163,
      "delta": -0.44145600000000007,
      "rel": -0.5710309072987999
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.723704,
      "after": 0.318254,
      "delta": -0.40545000000000003,
      "rel": -0.5602428617224722
    }
  },
  {
    "memory_ub.ub_read_bw_scalar_gbps": {
      "before": 0.0004,
      "after": 0.0003,
      "delta": -0.00010000000000000005,
      "rel": -0.2500000000000001
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1413.801,
      "after": 1681.871,
      "delta": 268.07000000000016,
      "rel": 0.18960942876684922
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1418.568,
      "after": 1686.394,
      "delta": 267.826,
      "rel": 0.18880025490494642
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

- 1. Identify vector processing loops in isamin.cpp Isamin::SingleProcess function
- 2. Check for SetFlag<HardEvent::V_S>() and WaitFlag<HardEvent::V_S>() calls
- 3. Verify no actual data dependencies exist between vector and scalar operations
- 4. Remove explicit V_S event synchronization using GetTPipePtr()->FetchEventID(HardEvent::V_S)
- 5. Monitor MTE2 utilization and vector wait ratio metrics after removal

## 检索线索

- V_S
- HardEvent
- SetFlag
- WaitFlag
- FetchEventID
- vector synchronization
- pipeline stall
- MTE2
- vector wait ratio
- AscendC
