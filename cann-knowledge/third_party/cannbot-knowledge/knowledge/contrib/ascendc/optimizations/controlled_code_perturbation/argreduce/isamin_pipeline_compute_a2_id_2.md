---
type: Optimization
title: "isamin：流水与计算优化"
description: "isamin 算子的流水与计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.reduction", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
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

- experience ID：`6ac78f7093f49b80`
- 源 schema version：`v2`
- 算子：`isamin`
- source iteration：`round_018_refactor`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.unnecessary.barrier`
- canonical family：`remove.unnecessary.barrier`
- 源记录时间：`2026-04-20T23:15:54`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminating unnecessary memory barrier synchronization in vector reduction loops
~~~~

### Original summary

~~~~text
In Ascend C vector operators performing reduction operations, unnecessary memory barrier synchronization can introduce significant pipeline stalls. The optimization involves removing `SetFlag<HardEvent::V_S>` and `WaitFlag<HardEvent::V_S>` calls that create artificial synchronization points when they are not required for data dependencies. This improves vector unit utilization and reduces wait ratios, particularly beneficial in reduction loops where compute-intensive operations dominate.
~~~~

### Original root mechanism

~~~~text
Removing unnecessary memory barriers in vector reduction loops reduces pipeline stalls by eliminating artificial synchronization points that block vector unit execution.
~~~~

### Original causal chain

~~~~text
Removing unnecessary memory barrier -> aiv_vec_wait_ratio_mean decreases from 0.718581 to 0.318254 (-55.71%) -> vector unit utilization improves (vec_ratio_mean increases from 0.468023 to 0.560565) -> task duration reduces from 1686.39us to 1418.69us (-15.87%)
~~~~

## 适用条件

- Vector operators performing reduction operations with `Isamin::SingleProcess` pattern
- Code contains `SetFlag<HardEvent::V_S>` and `WaitFlag<HardEvent::V_S>` calls in compute loops
- Profile shows high `aiv_vec_wait_ratio_mean` (>0.3) with low vector unit utilization
- MTE2 utilization ratio is high (>0.3) indicating memory transfer overhead
- Operator uses AscendC pipeline synchronization APIs in compute-intensive sections

## 失效边界

- Actual data dependencies require memory barrier synchronization
- Cross-core data sharing requires explicit synchronization
- Memory consistency between vector and scalar operations is required
- Operator uses shared memory buffers accessed by multiple threads

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- isamin/op_kernel/isamin.cpp

### 改动符号

- Isamin<T>::SingleProcess

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
- fast duration us：`1418.688354`
- improvement ratio：`0.15874426395871746`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.mte2_ratio_mean: 0.76917 -> 0.33163 (-56.88%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.718581 -> 0.318254 (-55.71%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0004 -> 0.0003 (-25.00%)
- critical_path.aiv_vector0_time_us_mean: 1414.41 -> 1681.87 (+18.91%)
- latency.task_duration_us: 1418.69 -> 1686.39 (+18.87%)

### Affected metric groups

- aiv_utilization
- pipeline_stalls
- memory_ub
- critical_path

### Changed metrics top

~~~~json
[
  {
    "metric": "aiv_utilization.mte2_ratio_mean",
    "before": 0.76917,
    "after": 0.33163,
    "delta": -0.43754,
    "rel": -0.5688469389081738
  },
  {
    "metric": "pipeline_stalls.aiv_vec_wait_ratio_mean",
    "before": 0.718581,
    "after": 0.318254,
    "delta": -0.400327,
    "rel": -0.5571076886252211
  },
  {
    "metric": "memory_ub.ub_read_bw_scalar_gbps",
    "before": 0.0004,
    "after": 0.0003,
    "delta": -0.0001,
    "rel": -0.25
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

- 1. Examine the operator's `SingleProcess` function in `isamin/op_kernel/isamin.cpp` for unnecessary `SetFlag`/`WaitFlag` calls
- 2. Check if memory barriers are placed between independent compute operations without data dependencies
- 3. Remove `event_t unnecessaryEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S))` lines
- 4. Remove corresponding `SetFlag<HardEvent::V_S>(unnecessaryEvent)` and `WaitFlag<HardEvent::V_S>(unnecessaryEvent)` calls
- 5. Verify functional correctness is preserved by checking reduction results remain unchanged
- 6. Profile the operator to confirm reduced `aiv_vec_wait_ratio_mean` and improved vector unit utilization

## 检索线索

- memory_barrier
- pipeline_stall
- vector_wait
- SetFlag
- WaitFlag
- HardEvent::V_S
- reduction
- isamin
- AscendC
- synchronization
- vector_unit
- MTE2
