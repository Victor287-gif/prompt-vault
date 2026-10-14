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

- experience ID：`2e537171c33670d5`
- 源 schema version：`v2`
- 算子：`isamin`
- source iteration：`round_003_refactor`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.unnecessary.sync`
- canonical family：`remove.unnecessary.sync`
- 源记录时间：`2026-04-20T19:35:49`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing unnecessary V_S event barrier synchronization in vector reduction loops
~~~~

### Original summary

~~~~text
In Ascend C operators performing vector reductions, unnecessary V_S event barriers inserted in SingleProcess methods cause significant pipeline stalls. Removing these barriers improves vector unit utilization from 46.8% to 55.9% and reduces vector wait ratio from 72.1% to 31.8%, resulting in 16.1% performance improvement. Apply this optimization when you see high aiv_vec_wait_ratio_mean and low vec_ratio_mean in operators with vector reduction patterns.
~~~~

### Original root mechanism

~~~~text
Removing explicit V_S event barriers in vector reduction loops eliminates unnecessary pipeline stalls, allowing vector units to maintain higher utilization by avoiding forced synchronization points.
~~~~

### Original causal chain

~~~~text
Removing V_S event barrier in SingleProcess -> aiv_vec_wait_ratio_mean decreases from 72.1% to 31.8% -> mte2_ratio_mean decreases from 76.7% to 33.2% -> vec_ratio_mean increases from 46.8% to 55.9% -> task_duration_us decreases from 1686µs to 1415µs
~~~~

## 适用条件

- Operator uses vector reduction patterns with SingleProcess method
- Profile shows high aiv_vec_wait_ratio_mean (>0.3) and low vec_ratio_mean (<0.6)
- Code contains explicit V_S event barriers using SetFlag<HardEvent::V_S> and WaitFlag<HardEvent::V_S>
- Vector unit utilization is below expected levels despite sufficient arithmetic operations

## 失效边界

- When V_S event barriers are required for correct memory synchronization
- When profiling shows no pipeline stalls (aiv_vec_wait_ratio_mean < 0.1)
- When vector unit utilization is already high (>0.8)
- When the barrier serves a legitimate synchronization purpose between vector and scalar operations

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
- fast duration us：`1414.968262`
- improvement ratio：`0.16095020913672456`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.mte2_ratio_mean: 0.766689 -> 0.33163 (-56.75%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.7207 -> 0.318254 (-55.84%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0004 -> 0.0003 (-25.00%)
- critical_path.aiv_vector0_time_us_mean: 1410.66 -> 1681.87 (+19.23%)
- latency.task_duration_us: 1414.97 -> 1686.39 (+19.18%)

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
      "before": 0.766689,
      "after": 0.33163,
      "delta": -0.435059,
      "rel": -0.5674517307539302
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.7207,
      "after": 0.318254,
      "delta": -0.402446,
      "rel": -0.5584098792840294
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
      "before": 1410.665,
      "after": 1681.871,
      "delta": 271.20600000000013,
      "rel": 0.19225400786154057
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1414.968,
      "after": 1686.394,
      "delta": 271.42599999999993,
      "rel": 0.1918248327877379
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

- 1. Examine SingleProcess method in isamin.cpp for V_S event barriers
- 2. Identify unnecessary SetFlag<HardEvent::V_S> and WaitFlag<HardEvent::V_S> calls
- 3. Remove explicit V_S event barriers that don't synchronize actual data dependencies
- 4. Verify vector reduction logic still works correctly without the barrier
- 5. Profile after removal to confirm reduced aiv_vec_wait_ratio_mean and increased vec_ratio_mean

## 检索线索

- V_S
- event_barrier
- pipeline_stall
- vector_reduction
- SingleProcess
- SetFlag
- WaitFlag
- vec_wait_ratio
- mte2_ratio
- AscendC
