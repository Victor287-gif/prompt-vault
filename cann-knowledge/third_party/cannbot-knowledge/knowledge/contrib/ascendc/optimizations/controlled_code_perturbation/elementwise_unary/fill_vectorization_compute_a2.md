---
type: Optimization
title: "fill：向量化计算优化"
description: "fill 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/docs/Fill.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/framework/tf_plugin/tensorflow_sqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/op_host/fill.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/op_host/fill_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/op_kernel/fill.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/atk/aclnn_fill.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/atk/generate_fill.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/atk/op_fill.yaml
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/Fill_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/test_fill.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# fill：向量化计算优化

## 来源信息

- experience ID：`b506317f24d9beac`
- 源 schema version：`v2`
- 算子：`fill`
- source iteration：`round_011_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vectorization.break.scalar.loop`
- canonical family：`vectorization.break.scalar.loop`
- 源记录时间：`2026-04-20T14:49:09`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use AscendC::Duplicate for vectorized fill operations
~~~~

### Original summary

~~~~text
When implementing fill operations in Ascend C kernels, use AscendC::Duplicate for vectorized data duplication instead of scalar element-wise loops. This optimization leverages SIMD parallelism to process multiple elements per instruction, significantly reducing execution time. Apply this technique when filling local tensors with constant values in Compute functions.
~~~~

### Original root mechanism

~~~~text
AscendC::Duplicate enables SIMD vectorization by processing multiple elements per instruction, reducing loop iteration overhead and improving memory write bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Replace scalar loops with AscendC::Duplicate -> Increases vector floating-point operations from 128 to 4,392,128 -> Reduces vector wait ratio from 99.37% to 0.57% -> Decreases task duration from 39,102μs to 1,035μs (37.8x speedup)
~~~~

## 适用条件

- Implementing fill operations with constant values in Ascend C kernels
- Filling AscendC::LocalTensor instances with repeated scalar values
- Processing contiguous memory regions with identical data patterns
- Using AscendC queue APIs (TQue) for tensor allocation and enqueuing
- Working with AscendC::Duplicate API for vectorized data duplication

## 失效边界

- Filling tensors with non-uniform or pattern-based values that cannot use Duplicate
- When element-wise computation requires per-element conditional logic
- When data dependencies prevent vectorized memory operations
- When working with non-contiguous memory access patterns

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/fill.cpp

### 改动符号

- KernelFill::Compute
- KernelFill1_INT64::Compute

### 代码签名

- AllocTensor
- EnQue
- SetValue
- TQue
- contiguous_access
- queue_api
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`d504c890dcdd950d8fe12c2806f25a57d9d99b70a4bc758b335fcfa30ed15d9a`

````diff
--- Before/op_kernel/fill.cpp
+++ After/op_kernel/fill.cpp
@@ -66,7 +66,10 @@
             outQueueOUT.EnQue<int8_t>(outLocal);
         } else {
             AscendC::LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
-            AscendC::Duplicate<T>(outLocal, this->value, this->processDataNum);
+            // Degradation: Replace vectorized Duplicate with scalar loop
+            for (uint32_t i = 0; i < this->processDataNum; ++i) {
+                outLocal.SetValue(i, this->value);
+            }
             outQueueOUT.EnQue<T>(outLocal);
         }
     }
@@ -150,11 +153,16 @@
     __aicore__ inline void Compute(uint32_t progress) {
         AscendC::LocalTensor<int32_t> outLocal = outQueueOUT.AllocTensor<int32_t>();

-        uint64_t mask2[2] = {0xAAAAAAAAAAAAAAAA, 0};
-        uint64_t mask1[2] = {0x5555555555555555, 0};
-
-        AscendC::Duplicate(outLocal, low, mask1, this->repeatTimes, 1, 8);
-        AscendC::Duplicate(outLocal, high, mask2, this->repeatTimes, 1, 8);
+        // Degradation: Replace vectorized Duplicate with scalar loops
+        // Process low and high parts separately in scalar fashion
+        for (uint32_t i = 0; i < this->processDataNum; i += 2) {
+            if (i < this->processDataNum) {
+                outLocal.SetValue(i, low);
+            }
+            if (i + 1 < this->processDataNum) {
+                outLocal.SetValue(i + 1, high);
+            }
+        }

         outQueueOUT.EnQue<int32_t>(outLocal);
     }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`39102.320312`
- fast duration us：`1034.78064`
- improvement ratio：`0.9735365924133552`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.017483 -> 0.994279 (+5587.12%)
- critical_path.aiv_vector0_time_us_mean: 1016.45 -> 39076.4 (+3744.41%)
- latency.task_duration_us: 1034.78 -> 39102.3 (+3678.80%)
- memory_ub.ub_write_bw_vector_gbps: 32.03 -> 0 (-100.00%)
- memory_ub.ub_write_bw_scalar_gbps: 0 -> 3.3321 (+100.00%)

### Affected metric groups

- aiv_utilization
- critical_path
- latency
- arithmetic

### Changed metrics top

~~~~json
[
  "aiv_utilization.scalar_ratio_mean: 0.017483 -> 0.994279 (+5587.12%)",
  "critical_path.aiv_vector0_time_us_mean: 1016.449 -> 39076.434 (+3744.41%)",
  "latency.task_duration_us: 1034.781 -> 39102.32 (+3678.80%)",
  "memory_ub.ub_write_bw_vector_gbps: 32.03 -> 0 (-100.00%)",
  "memory_ub.ub_write_bw_scalar_gbps: 0 -> 3.3321 (+100.00%)"
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

- In fill.cpp KernelFill::Compute function, replace scalar loops with AscendC::Duplicate<T>(outLocal, value, processDataNum)
- For bitmask patterns in KernelFill1_INT64::Compute, use AscendC::Duplicate with mask parameters instead of manual scalar loops
- Ensure outLocal is allocated via outQueueOUT.AllocTensor<T>() before calling Duplicate
- Use the appropriate Duplicate overload based on data type and pattern requirements
- Verify that processDataNum or repeatTimes parameters match the tensor size being filled

## 检索线索

- AscendC
- Duplicate
- vectorization
- fill
- scalar-loop
- SIMD
- LocalTensor
- AllocTensor
- EnQue
- TQue
- pipeline-stall
- memory-bandwidth
