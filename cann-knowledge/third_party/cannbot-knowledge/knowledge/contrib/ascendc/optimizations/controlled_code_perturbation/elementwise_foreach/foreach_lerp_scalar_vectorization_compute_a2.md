---
type: Optimization
title: "foreach_lerp_scalar：向量化计算优化"
description: "foreach_lerp_scalar 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/docs/aclnnForeachLerpScalar.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/op_host/foreach_lerp_scalar.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_lerp_scalar：向量化计算优化

## 来源信息

- experience ID：`972ebb03dfa7e11c`
- 源 schema version：`v2`
- 算子：`foreach_lerp_scalar`
- source iteration：`round_001_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vectorization.break`
- canonical family：`vectorization.break`
- 源记录时间：`2026-04-14T21:17:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use vectorized Sub and Axpy operations instead of scalar loops for float data processing
~~~~

### Original summary

~~~~text
Applying vectorized Sub and Axpy operations instead of scalar loops in Ascend C kernels significantly improves performance by enabling SIMD processing. This optimization reduces vector wait ratios from 0.95 to 0.003 and increases vector unit utilization from 0% to 8.35%, while improving memory bandwidth from 0.18/0.09 Gbps to 16.13/8.06 Gbps for read/write operations. Use this pattern when processing contiguous float data in InnerComputer specializations.
~~~~

### Original root mechanism

~~~~text
Vectorized operations (Sub and Axpy) enable SIMD processing of float data type in InnerComputer<float>::Compute, reducing pipeline stalls and improving memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Replacing scalar loops with vectorized Sub/Axpy operations → increases vector unit utilization from 0% to 8.35% and reduces vector wait ratio from 95.25% to 0.30% → eliminates pipeline stalls and improves memory bandwidth utilization → reduces task duration from 228,658μs to 2,614μs (87x speedup).
~~~~

## 适用条件

- Processing float data type in Ascend C kernels
- Using InnerComputer<T>::Compute pattern with contiguous memory access
- Operations involve element-wise arithmetic (Sub, Axpy) on vector data
- Working with UB (Unified Buffer) memory operations
- Performance bottleneck shows high scalar ratio (>0.99) and low vector ratio (<0.01)

## 失效边界

- Data dependencies prevent vectorization
- Memory access patterns are non-contiguous
- Working with data types not supported by vector units
- Operation count is too small to benefit from vectorization overhead

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/foreach_lerp_scalar.cpp

### 改动符号

- InnerComputer<float>::Compute

### 代码签名

- GetValue
- PipeBarrier
- SetValue
- contiguous_access
- pipebarrier
- scalar_value_access
- ub

### 原始 diff

- 源 diff SHA-256：`3b114624b41619cd1772d9b9021539c4a38ecb94adf53a2158e4dab13fbdaf93`

````diff
--- Before/op_kernel/foreach_lerp_scalar.cpp
+++ After/op_kernel/foreach_lerp_scalar.cpp
@@ -91,16 +91,28 @@
         PipeBarrier<PIPE_V>();

         if (weightVal < FLOAT_NUM_POS && weightVal > FLOAT_NUM_NEG) {
-            Sub(x2Local, x2Local, x1Local, dataCount);
-            PipeBarrier<PIPE_V>();
-            Axpy(x1Local, x2Local, weightVal, dataCount);
+            // Break vectorization by using scalar loop instead of vector Sub
+            for (int64_t i = 0; i < dataCount; ++i) {
+                x2Local.SetValue(i, x2Local.GetValue(i) - x1Local.GetValue(i));
+            }
+            PipeBarrier<PIPE_V>();
+            // Break vectorization by using scalar loop instead of vector Axpy
+            for (int64_t i = 0; i < dataCount; ++i) {
+                x1Local.SetValue(i, x1Local.GetValue(i) + weightVal * x2Local.GetValue(i));
+            }
             PipeBarrier<PIPE_V>();
         } else {
-            Sub(x1Local, x2Local, x1Local, dataCount);
+            // Break vectorization by using scalar loop instead of vector Sub
+            for (int64_t i = 0; i < dataCount; ++i) {
+                x1Local.SetValue(i, x2Local.GetValue(i) - x1Local.GetValue(i));
+            }
             PipeBarrier<PIPE_V>();
             weightVal = weightVal - FLOAT_NUM_ONE;
             PipeBarrier<PIPE_V>();
-            Axpy(x2Local, x1Local, weightVal, dataCount);
+            // Break vectorization by using scalar loop instead of vector Axpy
+            for (int64_t i = 0; i < dataCount; ++i) {
+                x2Local.SetValue(i, x2Local.GetValue(i) + weightVal * x1Local.GetValue(i));
+            }
             PipeBarrier<PIPE_V>();
         }
     }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`457312.40625`
- fast duration us：`5231.364502`
- improvement ratio：`0.988560632883552`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_read_bw_scalar_gbps: 0.0005 -> 1.4581 (+291520.00%)
- critical_path.aiv_vector0_time_us_mean: 2591 -> 228604 (+8723.01%)
- latency.task_duration_us: 2614.27 -> 228658 (+8646.52%)
- aiv_utilization.scalar_ratio_mean: 0.052381 -> 0.997116 (+1803.58%)
- memory_ub.ub_read_bw_vector_gbps: 32.26 -> 0 (-100.00%)

### Affected metric groups

- memory_ub
- critical_path
- latency
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_read_bw_scalar_gbps": {
      "before": 0.0005,
      "after": 1.4581,
      "delta": 1.4576,
      "rel": 2915.2
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2591.0,
      "after": 228604.061,
      "delta": 226013.061,
      "rel": 87.23005055962948
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2614.272,
      "after": 228657.953,
      "delta": 226043.681,
      "rel": 86.46524959912358
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.052381,
      "after": 0.997116,
      "delta": 0.944735,
      "rel": 18.035833603787633
    }
  },
  {
    "memory_ub.ub_read_bw_vector_gbps": {
      "before": 32.26,
      "after": 0.0,
      "delta": -32.26,
      "rel": -1.0
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 16.11,
      "after": 0.0,
      "delta": -16.11,
      "rel": -1.0
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 0.083501,
      "after": 0.0,
      "delta": -0.083501,
      "rel": -1.0
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.082203,
      "after": 0.0,
      "delta": -0.082203,
      "rel": -1.0
    }
  },
  {
    "pipeline_stalls.aiv_vec_total_cflt_ratio_mean": {
      "before": 0.071841,
      "after": 0.0,
      "delta": -0.071841,
      "rel": -1.0
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.037622,
      "after": 0.0,
      "delta": -0.037622,
      "rel": -1.0
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

- In op_kernel/foreach_lerp_scalar.cpp, replace scalar loops with vectorized operations in InnerComputer<float>::Compute
- Use Sub(dest, src1, src2, count) instead of for-loop with GetValue/SetValue for subtraction
- Use Axpy(dest, src, scalar, count) instead of for-loop with GetValue/SetValue for fused multiply-add
- Maintain PipeBarrier<PIPE_V>() calls after vector operations for synchronization
- Apply to similar patterns in other InnerComputer specializations for float data

## 检索线索

- vectorization
- SIMD
- Ascend C
- InnerComputer
- float
- Sub
- Axpy
- foreach_lerp_scalar
- pipeline_stall
- memory_bandwidth
- UB
- vector_unit
