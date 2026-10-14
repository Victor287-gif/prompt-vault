---
type: Optimization
title: "foreach_minimum_scalar：计算优化"
description: "foreach_minimum_scalar 算子的计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/docs/aclnnForeachMinimumScalar.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/op_host/foreach_minimum_scalar.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_minimum_scalar/op_kernel/foreach_minimum_scalar.cpp
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_minimum_scalar：计算优化

## 来源信息

- experience ID：`3b615415813f3cfa`
- 源 schema version：`v2`
- 算子：`foreach_minimum_scalar`
- source iteration：`round_003_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`computation.loop.artificial.chunking`
- canonical family：`computation.loop.artificial.chunking`
- 源记录时间：`2026-04-19T04:44:48`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminating artificial computation chunking in non-bfloat16 type operations
~~~~

### Original summary

~~~~text
For scalar-vector operations on non-bfloat16 data types, avoid breaking computations into artificial small chunks with excessive synchronization barriers. The original implementation added unnecessary chunking with PipeBarrier<PIPE_V>() calls before and after each small computation segment, causing pipeline stalls and reducing vector unit utilization. Instead, perform the operation in a single contiguous computation block to maximize hardware efficiency.
~~~~

### Original root mechanism

~~~~text
Removing unnecessary loop chunking and synchronization barriers in the generic InnerComputer::Compute template for non-bfloat16 data types reduces pipeline stalls and improves vector unit utilization.
~~~~

### Original causal chain

~~~~text
Removing artificial chunking reduces pipeline wait ratios (aiv_vec_wait_ratio_mean: 0.993476 -> 0.343269, -65.45%) and MTE wait ratios (aiv_mte2_wait_ratio_mean: 0.9967 -> 0.000017, -99.998%), which improves vector unit utilization (vec_ratio_mean: 0.034653 -> 0.551312, +1490.95%) and reduces task duration from 3723.994μs to 3300.626μs (-423.368μs, -11.37%).
~~~~

## 适用条件

- Implementing vector-scalar operations in Ascend C kernels
- Working with non-bfloat16 data types (float32, int32, etc.)
- When computation can be performed in a single contiguous block without data dependencies
- When using InnerComputer::Compute template pattern for generic data type handling

## 失效边界

- When working with bfloat16 data types that require special handling
- When memory bandwidth is the primary bottleneck and chunking helps hide latency
- When data dependencies require synchronization between computation segments
- When working with very large data sets that exceed L1 cache capacity

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- foreach_minimum_scalar/op_kernel/foreach_minimum_scalar.cpp

### 改动符号

- InnerComputer::Compute

### 代码签名

- PipeBarrier
- compile_time_specialization
- constexpr
- pipebarrier
- template

### 原始 diff

- 源 diff SHA-256：`3390cfd87373340ceb459dadc66540ce494b7f5090a3f4083c63f05e07b2aad3`

````diff
--- Before/op_kernel/foreach_minimum_scalar.cpp
+++ After/op_kernel/foreach_minimum_scalar.cpp
@@ -294,9 +294,31 @@
         T scalarVal,
         uint32_t maxCastDataCount,
         int64_t dataCount) {
-        PipeBarrier<PIPE_V>();
-        op(outLocal, dataLocal, scalarVal, dataCount);
-        PipeBarrier<PIPE_V>();
+        // Artificial chunking for degradation - break computation into smaller chunks
+        // even though it's not needed for non-bfloat16 types
+        // Use a small chunk size to maximize overhead
+        constexpr uint32_t CHUNK_SIZE = 64;  // Small chunk size for degradation
+        if (dataCount <= CHUNK_SIZE) {
+            // For small chunks, just do the operation once
+            PipeBarrier<PIPE_V>();
+            op(outLocal, dataLocal, scalarVal, dataCount);
+            PipeBarrier<PIPE_V>();
+        } else {
+            uint32_t numChunks = dataCount / CHUNK_SIZE;
+            uint32_t remainder = dataCount % CHUNK_SIZE;
+
+            for (uint32_t i = 0; i < numChunks; i++) {
+                PipeBarrier<PIPE_V>();
+                op(outLocal[i * CHUNK_SIZE], dataLocal[i * CHUNK_SIZE], scalarVal, CHUNK_SIZE);
+                PipeBarrier<PIPE_V>();
+            }
+
+            if (remainder > 0) {
+                PipeBarrier<PIPE_V>();
+                op(outLocal[numChunks * CHUNK_SIZE], dataLocal[numChunks * CHUNK_SIZE], scalarVal, remainder);
+                PipeBarrier<PIPE_V>();
+            }
+        }
     }
 };
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`7469.969483`
- fast duration us：`6611.572266`
- improvement ratio：`0.11491308216901316`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.000712 -> 0.44105 (+61845.22%)
- aiv_utilization.vec_ratio_mean: 0.034653 -> 0.551312 (+1490.95%)
- aiv_utilization.scalar_ratio_mean: 0.06002 -> 0.466908 (+677.92%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.033941 -> 0.110262 (+224.86%)
- arithmetic.aiv_vec_fops: 1.13391e+07 -> 3.36646e+07 (+196.89%)

### Affected metric groups

- arithmetic
- aiv_utilization
- pipeline_stalls
- critical_path

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_misc_ratio_mean",
  "aiv_utilization.vec_ratio_mean",
  "aiv_utilization.scalar_ratio_mean",
  "arithmetic.aiv_vec_fp32_ratio_mean",
  "arithmetic.aiv_vec_fops",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "aiv_utilization.mte2_ratio_mean"
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

- In foreach_minimum_scalar.cpp:294-318, replace chunked computation with direct operation
- Remove the CHUNK_SIZE constant and loop structure around op() calls
- Eliminate PipeBarrier<PIPE_V>() calls before and after each computation segment
- Call op(outLocal, dataLocal, scalarVal, dataCount) directly without artificial chunking
- Apply this pattern to similar foreach operators using InnerComputer::Compute template

## 检索线索

- foreach_minimum_scalar
- InnerComputer::Compute
- artificial_chunking
- pipeline_stall
- PipeBarrier
- vector_utilization
- MTE_wait
- non-bfloat16
- scalar_vector_operation
- computation_loop
