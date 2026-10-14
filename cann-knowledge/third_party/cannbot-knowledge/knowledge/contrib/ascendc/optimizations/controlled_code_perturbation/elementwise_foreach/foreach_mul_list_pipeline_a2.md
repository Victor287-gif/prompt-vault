---
type: Optimization
title: "foreach_mul_list：流水优化"
description: "foreach_mul_list 算子的流水优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/docs/aclnnForeachMulList.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_host/foreach_mul_list.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_kernel/foreach_mul_list.cpp
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_mul_list：流水优化

## 来源信息

- experience ID：`e16ac3529b0a8de6`
- 源 schema version：`v2`
- 算子：`foreach_mul_list`
- source iteration：`round_009_refactor`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`pipeline.mte_wait`
- canonical family：`untitled`
- 源记录时间：`2026-04-15T00:36:50`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
MTE Wait Reduction via Buffer Setup Reordering
~~~~

### Original summary

~~~~text
In Ascend C foreach operators, setting global buffers in the data copy function (CopyInPlus) instead of the processing loop (ProcessPlusInLoop) causes significant MTE wait stalls. Moving SetGlobalBuffer() calls to ProcessPlusInLoop reduces aiv_mte2_wait_ratio_mean from 0.96682 to 6.8e-05 and aiv_mte3_wait_ratio_mean from 0.97529 to 0.022798, improving overall performance by 26.9%. Apply this pattern when implementing foreach operators with multiple input tensors to enable concurrent memory transfers.
~~~~

### Original root mechanism

~~~~text
Moving SetGlobalBuffer() from data copy function to processing loop reduces MTE wait stalls by allowing concurrent memory transfers and computation.
~~~~

### Original causal chain

~~~~text
SetGlobalBuffer() moved from CopyInPlus to ProcessPlusInLoop -> aiv_mte2_wait_ratio_mean reduced by 99.99% and aiv_mte3_wait_ratio_mean reduced by 97.66% -> task_duration_us reduced from 6446.269μs to 5303.106μs (26.9% improvement)
~~~~

## 适用条件

- Implementing foreach operators with multiple input tensors in Ascend C
- Using SetGlobalBuffer() API for tensor address configuration
- Experiencing high MTE wait ratios (aiv_mte2_wait_ratio_mean > 0.9 or aiv_mte3_wait_ratio_mean > 0.9)
- Operator shows pipeline_stall_bound bottleneck type

## 失效边界

- Single tensor operations without multiple input buffers
- Operators with memory-bound rather than pipeline-bound bottlenecks
- When SetGlobalBuffer() must be called before AllocTensor() due to API constraints

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- op_kernel/foreach_mul_list.cpp

### 代码签名

- AllocTensor
- DataCopy
- bool
- bool_control
- datacopy
- queue_api

### 原始 diff

- 源 diff SHA-256：`98bb1659b5d0eca6b2734d13a61c10e745689697fc6f6fcab20d287a028404f6`

````diff
--- Before/op_kernel/foreach_mul_list.cpp
+++ After/op_kernel/foreach_mul_list.cpp
@@ -389,9 +389,6 @@

     __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder, uint16_t tensorIdx, int64_t cursorStart)
     {
-        // Degradation: Set global buffer for each chunk instead of once per tensor
-        inTensorsGM_2.SetGlobalBuffer(Base::GetTensorAddr(tensorIdx, inTensorsPtr_2) + cursorStart);
-
         LocalTensor<T> inLocal_2 = InQueue_2.AllocTensor<T>();
         if (isRemainder) {
             DataCopyExtParams copyParams {1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
@@ -414,7 +411,7 @@

     __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart)
     {
-        // Empty - buffer setup moved to CopyInPlus for degradation
+        inTensorsGM_2.SetGlobalBuffer(Base::GetTensorAddr(index, inTensorsPtr_2) + cursorStart);
     }

     friend Base;
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`14530.310547000001`
- fast duration us：`10626.712402000001`
- improvement ratio：`0.2686520795528321`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.355605 -> 0.99944 (+181.05%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.96682 -> 6.8e-05 (-99.99%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.97529 -> 0.022798 (-97.66%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.983898 -> 0.307758 (-68.72%)
- aiv_utilization.mte2_ratio_mean: 0.987743 -> 0.620059 (-37.22%)

### Affected metric groups

- aiv_utilization
- pipeline_stalls
- latency
- critical_path

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.355605,
      "after": 0.99944,
      "delta": 0.6438349999999999,
      "rel": 1.8105341600933618
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 0.96682,
      "after": 6.8e-05,
      "delta": -0.9667520000000001,
      "rel": -0.9999296663287893
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.97529,
      "after": 0.022798,
      "delta": -0.952492,
      "rel": -0.9766243886433779
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.983898,
      "after": 0.307758,
      "delta": -0.6761400000000001,
      "rel": -0.6872053810455963
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.987743,
      "after": 0.620059,
      "delta": -0.367684,
      "rel": -0.3722466269059867
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

- 1. In foreach_mul_list.cpp, locate CopyInPlus and ProcessPlusInLoop functions
- 2. Move SetGlobalBuffer() calls from CopyInPlus to ProcessPlusInLoop
- 3. Ensure SetGlobalBuffer() uses the correct tensor index and cursorStart offset
- 4. Verify that AllocTensor() still works correctly after buffer setup relocation
- 5. Profile to confirm reduction in aiv_mte2_wait_ratio_mean and aiv_mte3_wait_ratio_mean

## 检索线索

- foreach_mul_list
- SetGlobalBuffer
- MTE_wait
- pipeline_stall
- Ascend_C
- buffer_setup
- concurrent_memory
- CopyInPlus
- ProcessPlusInLoop
- memory_transfer_overlap
