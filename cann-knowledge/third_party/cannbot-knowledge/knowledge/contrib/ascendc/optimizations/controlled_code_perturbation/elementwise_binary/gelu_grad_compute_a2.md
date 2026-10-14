---
type: Optimization
title: "gelu_grad：计算优化"
description: "gelu_grad 算子的计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/docs/GeluGrad.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/tf_plugin/tensorflow_gelu_grad_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_host/gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_host/gelu_grad_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_kernel/gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/GeluGrad_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/test_gelu_grad.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gelu_grad：计算优化

## 来源信息

- experience ID：`7bc56b29ec96c173`
- 源 schema version：`v2`
- 算子：`gelu_grad`
- source iteration：`round_010_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`arithmetic.pipeline.stall`
- canonical family：`arithmetic.pipeline.stall`
- 源记录时间：`2026-04-20T20:58:17`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove redundant identity operations to reduce arithmetic pipeline stalls
~~~~

### Original summary

~~~~text
This optimization reduces arithmetic pipeline stalls by eliminating mathematically redundant identity operations (multiply by 1.0, add 0.0) that were inserted between dependent computations. These redundant operations force extra scalar arithmetic cycles that break pipeline parallelism, increasing vector wait time and bank conflict stalls. Apply this technique when profiling shows high aiv_vec_wait_ratio_mean and aiv_vec_bank_cflt_ratio_mean with excessive scalar operations.
~~~~

### Original root mechanism

~~~~text
Removing redundant multiply-by-1.0 and add-0.0 operations reduces arithmetic pipeline stalls by eliminating unnecessary scalar operations that break instruction-level parallelism between dependent computations.
~~~~

### Original causal chain

~~~~text
Removing redundant identity operations reduces aiv_vec_fops by 19.5% and scalar_ratio_mean by 19.4%, which decreases vector wait ratio by 12.6% and bank conflict ratio by 35.1%, resulting in 9.8% faster execution time.
~~~~

## 适用条件

- Profiling shows high aiv_vec_wait_ratio_mean (>0.2) combined with high scalar_ratio_mean (>0.1)
- Operator has dependent arithmetic operations that could be pipelined but show pipeline stalls
- Code contains mathematically redundant identity operations (multiply by 1.0, add 0.0) between dependent computations
- Vector operations show bank conflict stalls (aiv_vec_bank_cflt_ratio_mean > 0.005)

## 失效边界

- Operator is already memory-bound with low arithmetic utilization (<0.5)
- Scalar operations are necessary for correctness or numerical stability
- Profile shows low aiv_vec_wait_ratio_mean (<0.1) and low scalar_ratio_mean (<0.05)
- Operator has minimal arithmetic operations or is dominated by memory accesses

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- gelu_grad/op_kernel/gelu_grad.cpp

### 改动符号

- KernelGeluGrad::Compute

### 原始 diff

- 源 diff SHA-256：`0182f8e821dc38fb72290fdac596cc90d9097bdb47609f8d608169fcf706df78`

````diff
--- Before/op_kernel/gelu_grad.cpp
+++ After/op_kernel/gelu_grad.cpp
@@ -103,19 +103,27 @@
             LocalTensor<float> p1 = tmp1.Get<float>(); // xlocal
             LocalTensor<float> p2 = tmp2.Get<float>(); // dylocal
             LocalTensor<float> p3 = tmp3.Get<float>(); // zlocal
-            Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
-            Cast(p2, dyLocal, RoundMode::CAST_NONE, this->processDataNum);
-
-            Mul(p3, p1, p1, this->processDataNum);      // x*2
-            Muls(p3, p3, COEFF0, this->processDataNum); // x^2*COEFF0
+             Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
+             Cast(p2, dyLocal, RoundMode::CAST_NONE, this->processDataNum);
+             // REDUNDANT: add 0.0 to dy (adds arithmetic overhead without changing value)
+             Adds(p2, p2, 0.0f, this->processDataNum);
+
+             Mul(p3, p1, p1, this->processDataNum);      // x*2
+             // REDUNDANT: multiply by 1.0 (adds computational overhead)
+             Muls(p3, p3, 1.0f, this->processDataNum);
+             Muls(p3, p3, COEFF0, this->processDataNum); // x^2*COEFF0
             Adds(p3, p3, COEFF1, this->processDataNum); // x^2*COEFF0+COEFF1
-            Mul(p3, p3, p1, this->processDataNum);      //( x^2*COEFF0+COEFF1 )*x
-            Exp(p3, p3, this->processDataNum);          // exp()    px
-
-            Mul(tmp, p1, p1, this->processDataNum);
-            Muls(tmp, tmp, COEFF2, this->processDataNum); // x^2*COEFF2
-            Adds(tmp, tmp, COEFF3, this->processDataNum); // x^2*COEFF2 +COEFF3
-            Mul(tmp, tmp, p1, this->processDataNum);      // (x^2*COEFF2 +COEFF3)*x
+             Mul(p3, p3, p1, this->processDataNum);      //( x^2*COEFF0+COEFF1 )*x
+             // REDUNDANT: multiply by 1.0 between dependent operations (forces pipeline stall)
+             Muls(p3, p3, 1.0f, this->processDataNum);
+             Exp(p3, p3, this->processDa... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2390.787842`
- fast duration us：`2156.843018`
- improvement ratio：`0.09785260736657207`
- machine confidence：`0.85`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.007083 -> 0.009569 (+35.10%)
- arithmetic.aiv_vec_fops: 1.15178e+08 -> 1.37606e+08 (+19.47%)
- aiv_utilization.scalar_ratio_mean: 0.132327 -> 0.106706 (-19.36%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.038526 -> 0.033555 (-12.90%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.227171 -> 0.198551 (-12.60%)

### Affected metric groups

- pipeline_stalls
- arithmetic
- aiv_utilization
- critical_path

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.007083,
      "after": 0.009569,
      "delta": 0.0024859999999999995,
      "rel": 0.3509812226457715
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 115178048.0,
      "after": 137606208.0,
      "delta": 22428160.0,
      "rel": 0.19472599500904894
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.132327,
      "after": 0.106706,
      "delta": -0.025621000000000005,
      "rel": -0.19361883818117243
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.038526,
      "after": 0.033555,
      "delta": -0.004970999999999996,
      "rel": -0.12902974614546012
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.227171,
      "after": 0.198551,
      "delta": -0.028620000000000007,
      "rel": -0.12598439061323852
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.355829,
  "overall_deviation": 0.142332,
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

- 1. Examine the GeluGrad kernel in gelu_grad/op_kernel/gelu_grad.cpp, specifically the KernelGeluGrad::Compute function
- 2. Look for mathematically redundant identity operations like Adds(dst, src, 0.0f) or Muls(dst, src, 1.0f) between dependent computations
- 3. Remove these redundant operations while preserving mathematical equivalence
- 4. Verify the optimization reduces aiv_vec_fops count and scalar_ratio_mean in profiling metrics
- 5. Monitor pipeline stall metrics: aiv_vec_wait_ratio_mean should decrease by 10-15%, aiv_vec_bank_cflt_ratio_mean should decrease significantly
- 6. Ensure critical path time (aiv_vector0_time_us_mean) improves by 5-10%

## 检索线索

- gelu_grad
- arithmetic_pipeline_stall
- redundant_operations
- scalar_overhead
- identity_operations
- pipeline_optimization
- vector_wait_ratio
- bank_conflict
- Ascend_C
- KernelGeluGrad
