---
type: Optimization
title: "lerp：循环计算优化"
description: "lerp 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.synchronization", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/docs/Lerp.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/tf_plugin/tensorflow_lerp_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_host/lerp.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_host/lerp_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_kernel/lerp.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/Lerp_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/test_lerp.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# lerp：循环计算优化

## 来源信息

- experience ID：`1601034ee6e62c18`
- 源 schema version：`v2`
- 算子：`lerp`
- source iteration：`round_006_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`loop.parallelization`
- canonical family：`loop.serial.dependency`
- 源记录时间：`2026-04-20T19:49:03`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate serial loop dependency in vector element-wise computation
~~~~

### Original summary

~~~~text
Applying parallel element-wise computation instead of sequential dependency chains improves performance by enabling SIMD vectorization. This optimization eliminates serial dependencies between consecutive elements in vector operations, allowing the AscendC compiler to generate efficient vector instructions. Use this pattern when implementing element-wise operations that can be computed independently across vector lanes.
~~~~

### Original root mechanism

~~~~text
Removing sequential loop dependency where each output element depends on previous element's value enables SIMD parallelism and eliminates data dependency chains that prevent vectorization.
~~~~

### Original causal chain

~~~~text
Removing serial loop dependency -> reduces scalar utilization from 0.99277 to 0.144454 -> increases vector utilization from 0.005047 to 0.34443 -> reduces pipeline stalls (aiv_vec_wait_ratio_mean from 0.002952 to 0.664321) -> improves memory bandwidth utilization (ub_read_bw_vector_gbps from 1.93 to 132.77 Gbps) -> reduces latency from 135004 us to 1729 us.
~~~~

## 适用条件

- Implementing element-wise vector operations in AscendC kernels
- Processing contiguous data in Unified Buffer (UB) with independent computations
- Using AscendC vector APIs like Add, Sub, Mul with TYPE_START and TYPE_Y templates
- Working with lerp.cpp:115-135 patterns in KernelLerp::Compute function

## 失效边界

- When actual data dependencies exist between consecutive vector elements
- When using reduction operations that require sequential processing
- When working with scan or prefix-sum patterns that inherently require sequential dependencies
- When implementing algorithms with temporal dependencies across iterations

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- lerp/op_kernel/lerp.cpp

### 改动符号

- KernelLerp::Compute

### 代码签名

- EnQue
- FreeTensor
- GetValue
- SetValue
- TQue
- compile_time_specialization
- constexpr
- contiguous_access
- queue_api
- scalar_value_access
- ub

### 原始 diff

- 源 diff SHA-256：`bd1a572286cd9d5733923964ab9e27187d4b6c353cd29c448d04a06f6e5bb08e`

````diff
--- Before/op_kernel/lerp.cpp
+++ After/op_kernel/lerp.cpp
@@ -115,6 +115,12 @@
         AscendC::Sub(yLocal, endLocal, startLocal, this->processDataNum);
         AscendC::Mul(yLocal, weightLocal, yLocal, this->processDataNum);
         AscendC::Add(yLocal, yLocal, startLocal, this->processDataNum);
+
+        // Introduce serial dependency: make each element depend on previous element
+        // This breaks SIMD parallelism and creates sequential dependency chain
+        for (uint64_t i = 1; i < this->processDataNum; ++i) {
+          yLocal.SetValue(i, yLocal.GetValue(i) + yLocal.GetValue(i-1) * 0.0001f);
+        }
       }
       else if constexpr (std::is_same_v<TYPE_START, half> || std::is_same_v<TYPE_START, bfloat16_t>)
       {
@@ -135,6 +141,15 @@

         // Convert back to FP16 with rounding to nearest even
         AscendC::Cast(yLocal, fend, AscendC::RoundMode::CAST_RINT, this->processDataNum);
+
+        // Introduce serial dependency for half/bfloat16 types too
+        // Create temporary float buffer for dependency chain
+        auto tempBuf = tmp1.Get<float>();
+        AscendC::Cast(tempBuf, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
+        for (uint64_t i = 1; i < this->processDataNum; ++i) {
+          tempBuf.SetValue(i, tempBuf.GetValue(i) + tempBuf.GetValue(i-1) * 0.0001f);
+        }
+        AscendC::Cast(yLocal, tempBuf, AscendC::RoundMode::CAST_RINT, this->processDataNum);
       }
       outQueueY.EnQue<TYPE_Y>(yLocal);
       inQueueStart.FreeTensor(startLocal);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`135004.4375`
- fast duration us：`1729.634644`
- improvement ratio：`0.9871883126508342`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1723.88 -> 134991 (+7730.64%)
- latency.task_duration_us: 1729.63 -> 135004 (+7705.37%)
- aiv_utilization.scalar_ratio_mean: 0.144454 -> 0.99277 (+587.26%)
- memory_ub.ub_read_bw_scalar_gbps: 0 -> 1.2345 (+100.00%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.996968 -> 0 (-100.00%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- memory_ub

### Changed metrics top

~~~~json
[
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "aiv_utilization.scalar_ratio_mean",
  "memory_ub.ub_read_bw_scalar_gbps",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean"
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

- Replace sequential for-loops that access GetValue(i-1) in AscendC kernels with parallel vector operations
- Use AscendC::Add, AscendC::Sub, AscendC::Mul with this->processDataNum for element-wise computations
- Ensure all vector element computations are independent and can use SIMD parallelism
- Check for scalar value access patterns (GetValue/SetValue) in performance-critical loops
- Verify that vector utilization (aiv_utilization.vec_ratio_mean) increases after optimization

## 检索线索

- AscendC
- vector
- SIMD
- parallelization
- loop_dependency
- lerp
- element_wise
- UnifiedBuffer
- UB
- GetValue
- SetValue
- processDataNum
- KernelLerp
- scalar_ratio
- vector_ratio
- pipeline_stall
