---
type: Optimization
title: "icamin：循环计算优化"
description: "icamin 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.reduction", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/docs/aclnnIcamin.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/op_host/icamin.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/op_host/icamin_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/op_kernel/icamin.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/op_kernel/icamin_aiv.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# icamin：循环计算优化

## 来源信息

- experience ID：`75bd4b56ae1ffbc0`
- 源 schema version：`v2`
- 算子：`icamin`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`loop.restructure.separate`
- canonical family：`loop.restructure.unified`
- 源记录时间：`2026-04-20T18:58:56`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Separate batch processing loops for Abs and PairReduceSum operations
~~~~

### Original summary

~~~~text
Apply separate processing loops for Abs and PairReduceSum operations in Ascend C kernels to maintain pipeline efficiency. This optimization restructures unified loops with conditional branching into separate specialized loops, allowing vector instructions to chain effectively and reducing pipeline wait ratios from 62.7% to 6.2%. Use this pattern when processing multiple operation types on the same data buffer.
~~~~

### Original root mechanism

~~~~text
Separating batch processing loops for different operations enables vector instruction chaining and reduces pipeline stalls by eliminating frequent conditional branching within unified loops.
~~~~

### Original causal chain

~~~~text
Replacing unified conditional loop with separate batch loops → reduces pipeline wait ratio from 0.627 to 0.062 → decreases vector unit utilization wait → improves arithmetic efficiency → reduces task duration from 5698μs to 1418μs.
~~~~

## 适用条件

- Processing multiple operation types (Abs, PairReduceSum) on same data buffer in Ascend C
- Using unified loops with conditional branching for different operation types
- Experiencing high pipeline wait ratios (>0.6) in vector units
- Vector unit utilization shows low vec_ratio_mean (<0.5) with high scalar_ratio_mean (>0.7)
- Working with batch processing patterns using MAX_REPEATS and remainder handling

## 失效边界

- Single operation type processing
- Data-dependent branching is required within loops
- Operation count is small (<10 iterations)
- Memory bandwidth is the primary bottleneck (gm_to_ub_bw_usage_rate_pct < 5%)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/icamin.cpp

### 改动符号

- Icamin::SingleProcess

### 代码签名

- PipeBarrier
- contiguous_access
- pipebarrier

### 原始 diff

- 源 diff SHA-256：`3ef5e45a79ee1470c7498bb22ab2a770e8f872fbfe3b15630da4e8347c4f0ffb`

````diff
--- Before/op_kernel/icamin.cpp
+++ After/op_kernel/icamin.cpp
@@ -322,19 +322,34 @@
     }

     uint32_t offset = 0;
-
-    for (uint32_t i = 0; i < repeatBatchCnt; i++) {
-        Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, MAX_REPEATS, {1, 1, 8, 8});
-        offset += maxRepeatLen;
-    }
-
-    if (repeatBatchCntRemainder > 0) {
-        Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, repeatBatchCntRemainder, {1, 1, 8, 8});
-        offset += rmdRepeatLen;
-    }
-
-    if (totalRepeatCntRemainder > 0) {
-        Abs(srcLocal[offset], srcLocal[offset], totalRepeatCntRemainder, 1, {1, 1, 8, 8});
+    uint32_t totalIterations = repeatBatchCnt * MAX_REPEATS + repeatBatchCntRemainder + (totalRepeatCntRemainder > 0 ? 1 : 0);
+
+    for (uint32_t i = 0; i < totalIterations; i++) {
+        if (i < repeatBatchCnt * MAX_REPEATS) {
+            // Full batch iterations
+            uint32_t batchIdx = i / MAX_REPEATS;
+            uint32_t repeatInBatch = i % MAX_REPEATS;
+            if (repeatInBatch == 0) {
+                // Start of a new batch
+                offset = batchIdx * maxRepeatLen;
+            }
+            Abs(srcLocal[offset + repeatInBatch * elementsPerRepeat],
+                srcLocal[offset + repeatInBatch * elementsPerRepeat],
+                elementsPerRepeat, 1, {1, 1, 8, 8});
+        } else if (i < repeatBatchCnt * MAX_REPEATS + repeatBatchCntRemainder) {
+            // Remainder batch iterations
+            uint32_t remainderIdx = i - repeatBatchCnt * MAX_REPEATS;
+            if (remainderIdx == 0) {
+                offset = repeatBatchCnt * maxRepeatLen;
+            }
+            Abs(srcLocal[offset + remainderIdx * elementsPerRepeat],
+                srcLocal[offset + remainderIdx * elementsPerRepeat],
+                elementsPerRepeat, 1, {1, 1, 8, 8});
+        } else {
+            // Tail iteration
+            offset = repeatBa... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`5698.333984`
- fast duration us：`1417.508301`
- improvement ratio：`0.7512416251872681`
- machine confidence：`0.9`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.005514 -> 0.463984 (+8314.65%)
- aiv_utilization.scalar_ratio_mean: 0.07557 -> 0.718999 (+851.43%)
- critical_path.aiv_vector0_time_us_mean: 1412.73 -> 5693.75 (+303.03%)
- latency.task_duration_us: 1417.51 -> 5698.33 (+302.00%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.020914 -> 0 (-100.00%)

### Affected metric groups

- arithmetic
- aiv_utilization
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_misc_ratio_mean",
  "aiv_utilization.scalar_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean"
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

- In Icamin::SingleProcess function in op_kernel/icamin.cpp, replace unified conditional loops with separate batch processing loops
- Create separate loops for full batches (repeatBatchCnt * MAX_REPEATS), remainder batches (repeatBatchCntRemainder), and tail iterations (totalRepeatCntRemainder)
- Use dedicated loops for Abs operations followed by PairReduceSum operations instead of conditional branching within single loops
- Maintain contiguous memory access patterns by preserving offset calculation logic
- Apply PipeBarrier between separate operation loops to ensure data dependencies are respected

## 检索线索

- AscendC
- icamin
- loop_restructure
- pipeline_stall
- vector_chaining
- batch_processing
- MAX_REPEATS
- PairReduceSum
- PipeBarrier
- conditional_branching
- vec_wait_ratio
- aiv_utilization
