---
type: Optimization
title: "isamin：内存优化"
description: "isamin 算子的内存优化经验，关注流水线停顿。"
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
# isamin：内存优化

## 来源信息

- experience ID：`0550d1b4288b99ce`
- 源 schema version：`v2`
- 算子：`isamin`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.barrier.overhead`
- canonical family：`memory.barrier.overhead`
- 源记录时间：`2026-04-20T19:02:00`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant V_S memory barrier in reduction loop
~~~~

### Original summary

~~~~text
Removing unnecessary V_S memory barrier synchronization in reduction loops improves performance by reducing pipeline stalls. This optimization applies to Ascend C operators performing vector reductions where memory barriers are inserted after WholeReduceMin operations. The barrier creates synchronization overhead that stalls the vector pipeline while waiting for memory operations that have already completed.
~~~~

### Original root mechanism

~~~~text
Redundant V_S memory barrier synchronization forces pipeline stalls by inserting unnecessary synchronization points that block vector unit execution while waiting for memory operations to complete.
~~~~

### Original causal chain

~~~~text
apply memory.barrier.overhead in op_kernel/isamin.cpp -> improve aiv_utilization.mte2_ratio_mean -> 15.9% latency improvement
~~~~

## 适用条件

- Operator uses WholeReduceMin or similar reduction operations in Ascend C
- Code contains SetFlag<HardEvent::V_S> followed by WaitFlag<HardEvent::V_S> calls
- Reduction loop has unnecessary synchronization after vector operations
- Profile shows high aiv_vec_wait_ratio_mean (>0.3) indicating pipeline stalls

## 失效边界

- Actual data dependencies require memory barrier synchronization
- Cross-core synchronization is necessary for correctness
- Memory operations have producer-consumer dependencies across pipeline stages

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/isamin.cpp

### 改动符号

- Isamin::SingleProcess

### 代码签名

- SetFlag
- WaitFlag
- pipebarrier

### 原始 diff

- 源 diff SHA-256：`7c3c8be7551dc3d175b436f9a89f136e32ef54a20c9b05d4f4c77201d5488ebf`

````diff
--- Before/op_kernel/isamin.cpp
+++ After/op_kernel/isamin.cpp
@@ -364,6 +364,11 @@
     AscendC::SetMaskNorm();
     AscendC::ResetMask();

+    // Add unnecessary memory barrier synchronization to degrade performance
+    event_t unnecessaryEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
+    SetFlag<HardEvent::V_S>(unnecessaryEvent);
+    WaitFlag<HardEvent::V_S>(unnecessaryEvent);
+
     // 如果是第63次（k=62）或者是最后一次,本循环内马上就要进行压缩，此时必须把本次结果取出来参与压缩，不能等下次再处理；
     if (((k + 1) % DEAL_TIMES_EACH_CORE_REDUCE == 0 && k > 0) || k == this->dealTimesEachCore - 1) {
         GetReduceMinCount(k);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`1686.393677`
- fast duration us：`1418.948364`
- improvement ratio：`0.15859008287778342`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.mte2_ratio_mean: 0.766076 -> 0.33163 (-56.71%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.719105 -> 0.318254 (-55.74%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0004 -> 0.0003 (-25.00%)
- critical_path.aiv_vector0_time_us_mean: 1414.82 -> 1681.87 (+18.88%)
- latency.task_duration_us: 1418.95 -> 1686.39 (+18.85%)

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
    "before": 0.766076,
    "after": 0.33163,
    "delta": -0.434446,
    "rel_change": -0.5671056135422595
  },
  {
    "metric": "pipeline_stalls.aiv_vec_wait_ratio_mean",
    "before": 0.719105,
    "after": 0.318254,
    "delta": -0.400851,
    "rel_change": -0.5574304169766585
  },
  {
    "metric": "memory_ub.ub_read_bw_scalar_gbps",
    "before": 0.0004,
    "after": 0.0003,
    "delta": -0.00010000000000000005,
    "rel_change": -0.2500000000000001
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

- 1. Search for SetFlag<HardEvent::V_S> and WaitFlag<HardEvent::V_S> calls in op_kernel/isamin.cpp or similar reduction kernels
- 2. Verify if the barrier is necessary by checking data dependencies between vector operations
- 3. Remove redundant V_S memory barriers inserted after WholeReduceMin operations
- 4. Test that operator still produces correct results without the barrier
- 5. Profile to confirm reduction in aiv_vec_wait_ratio_mean and improvement in vector utilization

## 检索线索

- memory.barrier.overhead
- V_S
- SetFlag
- WaitFlag
- pipeline_stall
- WholeReduceMin
- reduction_loop
- vector_utilization
- aiv_vec_wait_ratio
- isamin
- AscendC
- memory_synchronization
