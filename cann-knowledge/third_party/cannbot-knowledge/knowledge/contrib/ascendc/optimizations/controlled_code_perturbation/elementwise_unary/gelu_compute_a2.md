---
type: Optimization
title: "gelu：计算优化"
description: "gelu 算子的计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/docs/Gelu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/tf_plugin/tensorflow_gelu_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_host/gelu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_host/gelu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_kernel/gelu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/Gelu_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/test_gelu.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gelu：计算优化

## 来源信息

- experience ID：`6caaefabb3551d02`
- 源 schema version：`v2`
- 算子：`gelu`
- source iteration：`round_003_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`computation.redundant.arithmetic`
- canonical family：`computation.redundant.arithmetic`
- 源记录时间：`2026-04-20T19:30:04`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant arithmetic operations in GELU kernel
~~~~

### Original summary

~~~~text
Removing redundant Cast operations in the non-float32 path and redundant Muls operations in the float32 path reduces vector wait ratio by 74% and improves overall performance by 5.4%. This optimization is applicable to Ascend C kernels where unnecessary arithmetic operations create pipeline stalls and increase scalar overhead.
~~~~

### Original root mechanism

~~~~text
Redundant Cast and Muls operations in the Compute() function increase vector pipeline stalls and scalar overhead without changing computation results.
~~~~

### Original causal chain

~~~~text
Removing redundant Cast(p2, srcLocal, RoundMode::CAST_NONE, length) and Muls(dstLocal, dstLocal, (T)1.0, length) operations → reduces aiv_vec_fops by 6.3M operations → decreases aiv_vec_wait_ratio_mean by 74.09% → reduces task_duration_us by 57.26µs (5.4% improvement)
~~~~

## 适用条件

- Ascend C kernels performing element-wise mathematical operations like GELU
- Kernels with type conversion paths for non-float32 data types using Cast operations
- Kernels with scalar multiplication operations (Muls) that multiply by identity value 1.0
- Operators showing high aiv_vec_wait_ratio_mean (>0.004) and scalar_ratio_mean (>0.15)
- Vector compute kernels with redundant arithmetic operations in Compute() function

## 失效边界

- Kernels where Cast operations are required for actual data type conversion
- Operations where Muls with 1.0 serves a legitimate purpose (e.g., scaling by dynamic coefficient)
- Kernels already optimized for minimal arithmetic operations
- When aiv_vec_wait_ratio_mean is already low (<0.001)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- gelu/op_kernel/gelu.cpp

### 改动符号

- KernelGelu::Compute

### 原始 diff

- 源 diff SHA-256：`9a657544d95ab514eb2bee32318df4a9925b994f561301a5d87dd460835c6e67`

````diff
--- Before/op_kernel/gelu.cpp
+++ After/op_kernel/gelu.cpp
@@ -67,6 +67,8 @@
             LocalTensor<float> p2 = tmp2.Get<float>();
             Cast(p1, srcLocal, RoundMode::CAST_NONE, length);
             Cast(p2, srcLocal, RoundMode::CAST_NONE, length);
+            // Redundant cast operation to degrade performance
+            Cast(p2, srcLocal, RoundMode::CAST_NONE, length);
             Mul(p2, p1, p1, length);
             Mul(p2, p2, p1, length);
             Muls(p2, p2, (float)this->param1, length);
@@ -83,6 +85,8 @@
         {
             Mul(dstLocal, srcLocal, srcLocal, length);
             Mul(dstLocal, dstLocal, srcLocal, length);
+            // Redundant scalar multiplication to degrade performance
+            Muls(dstLocal, dstLocal, (T)1.0, length);
             Muls(dstLocal, dstLocal, (T)this->param1, length);
             Add(dstLocal, dstLocal, srcLocal, length);
             Muls(dstLocal, dstLocal, (T)this->param2, length);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1065.081299`
- fast duration us：`1007.820129`
- improvement ratio：`0.05376225275362759`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.004608 -> 0.001194 (-74.09%)
- aiv_utilization.mte3_ratio_mean: 0.236909 -> 0.158951 (-32.91%)
- aiv_utilization.mte2_ratio_mean: 0.586184 -> 0.462171 (-21.16%)
- aiv_utilization.scalar_ratio_mean: 0.153418 -> 0.122138 (-20.39%)
- arithmetic.aiv_vec_fops: 7.58586e+07 -> 8.21718e+07 (+8.32%)

### Affected metric groups

- pipeline_stalls
- aiv_utilization
- arithmetic
- critical_path

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "aiv_utilization.mte2_ratio_mean",
  "aiv_utilization.scalar_ratio_mean",
  "arithmetic.aiv_vec_fops"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.136337,
  "overall_deviation": 0.054535,
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

- Inspect Compute() function in gelu/op_kernel/gelu.cpp for redundant Cast operations in non-float32 path
- Check for Muls operations multiplying by identity value 1.0 in float32 path
- Profile kernel with aiv_vec_wait_ratio_mean metric to identify pipeline stalls
- Monitor scalar_ratio_mean to detect excessive scalar operation overhead
- Compare aiv_vec_fops before and after removing redundant operations
- Verify computation correctness after removing redundant operations

## 检索线索

- gelu
- ascendc
- redundant_arithmetic
- cast_operation
- muls
- vector_wait
- scalar_overhead
- pipeline_stalls
- aiv_utilization
- computation_optimization
