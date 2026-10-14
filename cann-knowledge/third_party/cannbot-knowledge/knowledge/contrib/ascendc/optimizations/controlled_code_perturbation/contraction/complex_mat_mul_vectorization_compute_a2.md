---
type: Optimization
title: "complex_mat_mul：向量化计算优化"
description: "complex_mat_mul 算子的向量化计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/docs/ComplexMatMul.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/op_host/complex_mat_mul.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/op_host/complex_mat_mul_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/op_kernel/complex_mat_mul.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# complex_mat_mul：向量化计算优化

## 来源信息

- experience ID：`9c70bb02c827a914`
- 源 schema version：`v2`
- 算子：`complex_mat_mul`
- source iteration：`round_012_degrade`
- 源 category：`matmul`
- 计算分区：`contraction`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vector.operation.count.double`
- canonical family：`vector.operation.count.double`
- 源记录时间：`2026-04-21T02:02:26`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Optimize GatherMask vector operation count for complex data separation
~~~~

### Original summary

~~~~text
For complex matrix multiplication operators that separate real and imaginary components using GatherMask, adjusting the stride parameter from count/32 to count/16 reduces the number of vector operations required by half. This optimization improves vector unit utilization from 18.49% to 22.26% and reduces overall latency by 5.2%. Apply this technique when complex data separation is performed using GatherMask with stride-based access patterns.
~~~~

### Original root mechanism

~~~~text
Reducing GatherMask stride from count/32 to count/16 halves the vector operations needed to separate real and imaginary components in complex matrix multiplication, improving vector unit utilization.
~~~~

### Original causal chain

~~~~text
Changed GatherMask stride from count/32 to count/16 → Vector operation count reduced from 15.9M to 10.7M → Vector unit utilization increased from 18.49% to 22.26% → UB vector bandwidth increased by 41.7-52.6% → Overall latency reduced by 116.2μs (5.2%)
~~~~

## 适用条件

- Complex matrix multiplication operators using GatherMask for data separation
- When separating real and imaginary components from complex tensors
- When vector unit utilization is below 20% and shows scalar overhead bound
- When GatherMask is called with stride parameters based on element count

## 失效边界

- Non-complex operators or operators not using GatherMask
- When memory bandwidth is already saturated (>80% utilization)
- When vector unit utilization is already high (>40%)
- When using different data access patterns than stride-based separation

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- complex_mat_mul/op_kernel/complex_mat_mul.cpp

### 改动符号

- KernelMatMulComplex::Split

### 代码签名

- TBuf
- bool_control
- contiguous_access
- tbuf
- ub

### 原始 diff

- 源 diff SHA-256：`b246dfce6706eb9945f621193ee211affa442dec4b6c844648d34ee82bc27ad2`

````diff
--- Before/op_kernel/complex_mat_mul.cpp
+++ After/op_kernel/complex_mat_mul.cpp
@@ -261,14 +261,14 @@
     }

     __aicore__ inline void Split(const LocalTensor<float> &xComplex,
-                                 const LocalTensor<float> &xReal,
-                                 const LocalTensor<float> &xImag,
-                                 uint32_t count)
+                                  const LocalTensor<float> &xReal,
+                                  const LocalTensor<float> &xImag,
+                                  uint32_t count)
     {
         LocalTensor<uint32_t> offset = tBufOffset.Get<uint32_t>();
         uint64_t rsvdCnt;
-        GatherMask(xReal, xComplex, 1, false, 0, {1, static_cast<uint16_t>(count / 32), 8, 8}, rsvdCnt);
-        GatherMask(xImag, xComplex, 2, false, 0, {1, static_cast<uint16_t>(count / 32), 8, 8}, rsvdCnt);
+        GatherMask(xReal, xComplex, 1, false, 0, {1, static_cast<uint16_t>(count / 16), 8, 8}, rsvdCnt);
+        GatherMask(xImag, xComplex, 2, false, 0, {1, static_cast<uint16_t>(count / 16), 8, 8}, rsvdCnt);
     }

     __aicore__ inline void Combine(const LocalTensor<float> &xComplex,
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2234.844727`
- fast duration us：`2118.622314`
- improvement ratio：`0.052004692583727745`
- machine confidence：`0.95`

### Profile evidence

- parallelism.block_time_cv: 0.0001 -> 0 (-100.00%)
- memory_ub.ub_write_bw_vector_gbps: 29.36 -> 44.79 (+52.55%)
- arithmetic.aiv_vec_fops: 1.07409e+07 -> 1.59019e+07 (+48.05%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.01567 -> 0.02273 (+45.05%)
- memory_ub.ub_read_bw_vector_gbps: 43.37 -> 61.45 (+41.69%)

### Affected metric groups

- parallelism
- memory_ub
- arithmetic
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "parallelism.block_time_cv": {
      "before": 0.0001,
      "after": 0.0,
      "delta": -0.0001,
      "rel": -1.0
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 29.36,
      "after": 44.79,
      "delta": 15.43,
      "rel": 0.5255449591280654
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 10740928.0,
      "after": 15901888.0,
      "delta": 5160960.0,
      "rel": 0.48049479523557
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.01567,
      "after": 0.02273,
      "delta": 0.00706,
      "rel": 0.45054243777919595
    }
  },
  {
    "memory_ub.ub_read_bw_vector_gbps": {
      "before": 43.37,
      "after": 61.45,
      "delta": 18.080000000000005,
      "rel": 0.4168780262854509
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.097151,
  "overall_deviation": 0.03886,
  "metrics_checked": 1,
  "metrics_wrong": 0,
  "performance_metrics_checked": 1,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- Locate GatherMask calls in complex operator kernels (e.g., complex_mat_mul/op_kernel/complex_mat_mul.cpp)
- Identify calls separating real/imaginary components with stride parameters
- Change stride from count/32 to count/16 in GatherMask parameter {1, static_cast<uint16_t>(count/16), 8, 8}
- Verify vector operation count (aiv_vec_fops) decreases by approximately 30-50%
- Monitor vector unit utilization (aiv_utilization.vec_ratio_mean) increases by 3-5 percentage points

## 检索线索

- GatherMask
- complex_mat_mul
- vector_operation_count
- stride_optimization
- complex_data_separation
- Ascend_C
- vector_unit_utilization
- scalar_overhead_bound
