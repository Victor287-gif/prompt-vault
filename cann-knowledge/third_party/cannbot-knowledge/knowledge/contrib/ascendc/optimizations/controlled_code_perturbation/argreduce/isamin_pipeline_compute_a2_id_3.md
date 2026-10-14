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

- experience ID：`752741faf330042f`
- 源 schema version：`v2`
- 算子：`isamin`
- source iteration：`round_006_refactor`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.unnecessary.sync`
- canonical family：`remove.unnecessary.sync`
- 源记录时间：`2026-04-20T20:28:10`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate unnecessary explicit memory barrier synchronization in Ascend C kernels
~~~~

### Original summary

~~~~text
In Ascend C kernels, avoid adding explicit SetFlag/WaitFlag synchronization pairs unless they are necessary for data dependencies. Unnecessary memory barriers create artificial pipeline stalls that reduce vector unit utilization and increase latency. This optimization applies when synchronization is added for safety but analysis shows no actual data hazard requiring the barrier.
~~~~

### Original root mechanism

~~~~text
Removing SetFlag/WaitFlag synchronization pairs that aren't required for correctness eliminates artificial pipeline stalls and improves vector unit utilization.
~~~~

### Original causal chain

~~~~text
Removing unnecessary SetFlag/WaitFlag pair → reduces aiv_vec_wait_ratio_mean from 0.71888 to 0.318254 (-55.73%) → increases vector unit utilization from 0.468023 to 0.558905 (+19.4%) → reduces task duration from 1686.39μs to 1421.09μs (-15.73%)
~~~~

## 适用条件

- Ascend C kernel uses SetFlag/WaitFlag synchronization with HardEvent::V_S
- Code analysis shows no actual data dependency requiring the barrier
- Profile shows high aiv_vec_wait_ratio_mean (>0.3)
- Vector unit utilization (vec_ratio_mean) is below 0.5
- MTE2 utilization ratio is abnormally high (>0.7)

## 失效边界

- Actual data dependencies exist between vector operations
- Memory consistency requirements mandate explicit synchronization
- Cross-core communication requires barrier synchronization
- Hardware documentation specifies required synchronization points

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
- fast duration us：`1421.088379`
- improvement ratio：`0.15732109389307203`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.mte2_ratio_mean: 0.765118 -> 0.33163 (-56.66%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.71888 -> 0.318254 (-55.73%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0004 -> 0.0003 (-25.00%)
- critical_path.aiv_vector0_time_us_mean: 1416.56 -> 1681.87 (+18.73%)
- latency.task_duration_us: 1421.09 -> 1686.39 (+18.67%)

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
      "before": 0.765118,
      "after": 0.33163,
      "delta": -0.433488,
      "rel": -0.566563588884329
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.71888,
      "after": 0.318254,
      "delta": -0.400626,
      "rel": -0.5572918985087915
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
      "before": 1416.56,
      "after": 1681.871,
      "delta": 265.31100000000015,
      "rel": 0.18729245496131486
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1421.088,
      "after": 1686.394,
      "delta": 265.30600000000004,
      "rel": 0.18669216825418275
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

- 1. Identify SetFlag/WaitFlag pairs in Ascend C kernel (e.g., in isamin.cpp:SingleProcess function)
- 2. Analyze data flow to verify if synchronization is required for correctness
- 3. Remove unnecessary SetFlag<HardEvent::V_S>/WaitFlag<HardEvent::V_S> pairs
- 4. Test kernel functionality to ensure correctness is maintained
- 5. Profile before/after to verify reduction in aiv_vec_wait_ratio_mean and improvement in vector utilization

## 检索线索

- SetFlag
- WaitFlag
- memory barrier
- pipeline stall
- Ascend C
- synchronization
- vector wait
- aiv_vec_wait_ratio_mean
- mte2_ratio_mean
- isamin
