---
type: Optimization
title: "gelu_quant：向量化计算优化"
description: "gelu_quant 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.programming_model", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/docs/aclnnGeluQuant.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_host/gelu_quant.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_host/gelu_quant_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_host/gelu_quant_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_host/gelu_quant_tiling_base.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_host/gelu_quant_tiling_def.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_kernel/gelu_dynamic_quant.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_kernel/gelu_dynamic_quant_workspace.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_kernel/gelu_quant.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_kernel/gelu_quant_base.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_kernel/gelu_static_quant.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_kernel/gelu_static_quant_block.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/op_kernel/gelu_static_quant_per_tensor.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/gelu_quant/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gelu_quant：向量化计算优化

## 来源信息

- experience ID：`221d0506565bbaa6`
- 源 schema version：`v2`
- 算子：`gelu_quant`
- source iteration：`round_002_degrade`
- 源 category：`activation`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vectorization.break.scalar.reduction`
- canonical family：`vectorization.break.scalar.reduction`
- 源记录时间：`2026-04-14T21:42:48`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Vectorized reduction for max computation in ComputeReduceMax
~~~~

### Original summary

~~~~text
Applying vectorized reduction operations in ComputeReduceMax function improves performance by 11.6x for gelu_quant operator. This optimization replaces scalar loops with hardware-accelerated vector reduction operations, enabling parallel processing of multiple elements and significantly reducing execution time. Use this technique when computing reductions (max, min, sum) over contiguous memory regions in Ascend C vector operators.
~~~~

### Original root mechanism

~~~~text
Using vectorized reduction operations (Max) instead of scalar loops enables hardware-accelerated parallel processing of multiple elements simultaneously, reducing execution cycles and improving vector unit utilization.
~~~~

### Original causal chain

~~~~text
Vectorized reduction operations replace scalar loops -> vector unit utilization increases from 6.3% to 75.5% -> scalar unit utilization decreases from 93.9% to 27.2% -> UB vector bandwidth increases from 7.75/7.28 Gbps to 93.68/88.03 Gbps -> task duration reduces from 19494μs to 1615μs
~~~~

## 适用条件

- Computing reductions (max, min, sum) over contiguous memory regions in Ascend C vector operators
- Processing large arrays where vectorized operations can be applied
- Using FP32_VECTOR_CAPACITY_ONE_CYCLE and FP32_BLOCK_NUM constants for vector operations
- Working with temporary UB (Unified Buffer) tensors like 'tempRes' in ComputeReduceMax function

## 失效边界

- Reduction size is smaller than vector capacity (calCount < FP32_VECTOR_CAPACITY_ONE_CYCLE)
- Memory access patterns are non-contiguous or strided
- Data dependencies require sequential processing
- Using scalar operations for element-wise comparisons is explicitly required

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/gelu_quant.cpp

### 改动符号

- GeluQuantBase::ComputeReduceMax

### 代码签名

- GetValue
- PipeBarrier
- compile_time_specialization
- contiguous_access
- pipebarrier
- scalar_value_access
- template
- ub

### 原始 diff

- 源 diff SHA-256：`86ce03f327bc90f65138fad8a66b1c96f8b3ad433df2342fbdfd6e48a4ca8084`

````diff
--- Before/op_kernel/gelu_quant.cpp
+++ After/op_kernel/gelu_quant.cpp
@@ -169,6 +169,17 @@
     repeatParams.src0RepStride = FP32_BLOCK_NUM;
     repeatParams.src1RepStride = 0;

+    // Degradation: Replace vectorized reduction with scalar reduction
+    // This breaks the efficient vectorization pattern and forces scalar operations
+    float maxVal = -__FLT_MAX__;
+    for (int32_t i = 0; i < calCount; i++) {
+        float val = static_cast<float>(tempRes.GetValue(i));
+        if (val > maxVal) {
+            maxVal = val;
+        }
+    }
+
+    // Still need to write result back to tensor for compatibility
     if (vectorCycles > 0 && remainElements > 0) {
         Max(tempRes, tempRes, tempRes[vectorCycles * FP32_VECTOR_CAPACITY_ONE_CYCLE], remainElements, 1, repeatParams);
         PipeBarrier<PIPE_V>();
@@ -187,7 +198,8 @@
     set_flag(PIPE_V, PIPE_S, curEventID);
     wait_flag(PIPE_V, PIPE_S, curEventID);

-    maxValue = static_cast<float>(tempRes.GetValue(0));
+    // Use the scalar computed max value
+    maxValue = maxVal;
 }

 template <typename T1, typename T2> class GeluDynamicQuant : public GeluQuantBase {
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`19494.210938`
- fast duration us：`1615.412354`
- improvement ratio：`0.9171337399016709`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_read_bw_scalar_gbps: 0.0253 -> 1.069 (+4125.30%)
- critical_path.aiv_vector0_time_us_mean: 1612.27 -> 19487.8 (+1108.72%)
- latency.task_duration_us: 1615.41 -> 19494.2 (+1106.76%)
- aiv_utilization.scalar_ratio_mean: 0.27215 -> 0.939283 (+245.13%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.000718 -> 3e-06 (-99.58%)

### Affected metric groups

- memory_ub
- critical_path
- latency
- aiv_utilization

### Changed metrics top

~~~~json
[
  "memory_ub.ub_read_bw_scalar_gbps",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "aiv_utilization.scalar_ratio_mean",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean"
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

- Replace scalar reduction loops with vectorized reduction operations like Max() in op_kernel/gelu_quant.cpp
- Use hardware-accelerated vector operations for reductions over contiguous memory regions
- Apply vector operations to temporary UB tensors using GetValue() for scalar extraction when needed
- Maintain proper pipe synchronization with PipeBarrier<PIPE_V>() and wait_flag() calls
- Ensure vector cycles and remaining elements are handled correctly with vectorCycles and remainElements

## 检索线索

- vectorization
- reduction
- max
- scalar-loop
- ComputeReduceMax
- gelu_quant
- Ascend C
- vector operations
- UB tensor
- pipe barrier
