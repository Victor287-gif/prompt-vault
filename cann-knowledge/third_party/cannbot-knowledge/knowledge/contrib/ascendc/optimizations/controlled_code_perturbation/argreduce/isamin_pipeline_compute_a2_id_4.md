---
type: Optimization
title: "isamin：流水与计算优化"
description: "isamin 算子的流水与计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.reduction", "topic.implementation.synchronization", "topic.data_movement.memory_access"]
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

- experience ID：`8692c9b867abedd0`
- 源 schema version：`v2`
- 算子：`isamin`
- source iteration：`round_017_refactor`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.unnecessary.sync`
- canonical family：`remove.unnecessary.sync`
- 源记录时间：`2026-04-20T22:53:51`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove unnecessary memory barrier synchronization
~~~~

### Original summary

~~~~text
On `math/isamin`, a structural refactoring (`Remove unnecessary memory barrier synchronization`) that appeared semantically equivalent unexpectedly caused 15.9% performance change (1418.4 vs 1686.4 us, 15.9% difference). This reveals that `pipeline_stall_bound` is sensitive to this code pattern. Removed an explicitly marked unnecessary V_S memory barrier synchronization that was added to degrade performance
~~~~

### Original root mechanism

~~~~text
Removed an explicitly marked unnecessary V_S memory barrier synchronization that was added to degrade performance
~~~~

### Original causal chain

~~~~text
structural change to remove.unnecessary.sync in isamin/op_kernel/isamin.cpp -> unexpected shift in aiv_utilization.mte2_ratio_mean -> 15.9% latency impact
~~~~

## 适用条件

- The target operator's profiling shows bottleneck in: aiv_utilization, pipeline_stalls, memory_ub
- The kernel code uses related APIs/patterns: SetFlag, WaitFlag, pipebarrier
- The operator has a `pipeline_stall_bound` bottleneck profile
- You are considering a similar structural refactoring and want to verify it won't affect performance

## 失效边界

- The operator's bottleneck is unrelated to `pipeline_stall_bound`
- The `remove.unnecessary.sync` pattern is not present in the target code

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

- 源 diff SHA-256：`acab70dfffddf28a2cf6481955185c03c37ed359479b79abde867a33494764c6`

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
- fast duration us：`1418.368408`
- improvement ratio：`0.15893398596987265`
- machine confidence：`0.5`

### Profile evidence

- aiv_utilization.mte2_ratio_mean: 0.772374 -> 0.33163 (-57.06%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.724036 -> 0.318254 (-56.04%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0004 -> 0.0003 (-25.00%)
- critical_path.aiv_vector0_time_us_mean: 1413.31 -> 1681.87 (+19.00%)
- latency.task_duration_us: 1418.37 -> 1686.39 (+18.90%)

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
      "before": 0.772374,
      "after": 0.33163,
      "delta": -0.440744,
      "rel": -0.5706354693451618
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.724036,
      "after": 0.318254,
      "delta": -0.40578200000000003,
      "rel": -0.5604445082841185
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
      "before": 1413.307,
      "after": 1681.871,
      "delta": 268.5640000000001,
      "rel": 0.19002523867779617
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1418.368,
      "after": 1686.394,
      "delta": 268.02600000000007,
      "rel": 0.18896788421622604
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 8.2e-05,
      "after": 6.7e-05,
      "delta": -1.4999999999999999e-05,
      "rel": -0.18292682926829265
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 0.559737,
      "after": 0.468023,
      "delta": -0.09171400000000002,
      "rel": -0.16385195189883822
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.553502,
      "after": 0.462811,
      "delta": -0.09069100000000008,
      "rel": -0.16384945311850738
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.006234,
      "after": 0.005213,
      "delta": -0.0010210000000000002,
      "rel": -0.1637792749438563
    }
  },
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.0104,
      "after": 0.0087,
      "delta": -0.0017000000000000001,
      "rel": -0.1634615384615385
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

- Before applying a similar structural change, profile the `remove.unnecessary.sync` code path in `isamin/op_kernel/isamin.cpp` (key symbols: Isamin::SingleProcess)
- After the change, verify that `aiv_utilization.mte2_ratio_mean` moves in the expected direction
- If metrics regress, the change is NOT performance-neutral — revert it

## 检索线索

- After
- AscendC
- Before
- DEAL_TIMES_EACH_CORE_REDUCE
- FetchEventID
- GetReduceMinCount
- GetTPipePtr
- HardEvent
- ResetMask
- SetFlag
- SetMaskNorm
- WaitFlag
- aiv_utilization
- critical_path
- dealTimesEachCore
- memory_ub
- pipebarrier
- pipeline_stall_bound
- pipeline_stalls
- remove.unnecessary.sync
- unnecessaryEvent
