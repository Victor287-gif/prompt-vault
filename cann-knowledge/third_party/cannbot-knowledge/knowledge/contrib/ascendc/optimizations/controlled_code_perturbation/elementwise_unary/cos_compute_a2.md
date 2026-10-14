---
type: Optimization
title: "cos：计算优化"
description: "cos 算子的计算优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/docs/Cos.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/framework/tf_plugin/tensorflow_cos_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/op_host/cos.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/op_host/cos_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/op_kernel/cos.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/Cos_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/test_cos.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# cos：计算优化

## 来源信息

- experience ID：`c70f96d62bd35695`
- 源 schema version：`v2`
- 算子：`cos`
- source iteration：`round_015_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`computational.overhead.redundant.ops`
- canonical family：`computational.overhead.redundant.ops`
- 源记录时间：`2026-04-20T15:53:46`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant identity arithmetic operations in computation kernel
~~~~

### Original summary

~~~~text
In Ascend C math operators, avoid inserting redundant identity operations (multiplying by 1.0 or adding 0.0) in computation kernels. These operations increase floating-point operation count (FLOPS) without changing mathematical results, leading to unnecessary computational overhead and pipeline stalls. Apply this optimization to polynomial evaluation sections and other compute-intensive regions where identity operations may be inadvertently introduced.
~~~~

### Original root mechanism

~~~~text
Removing unnecessary multiplication-by-1.0 and addition-of-0.0 operations in polynomial evaluation reduces floating-point operations and improves pipeline efficiency.
~~~~

### Original causal chain

~~~~text
apply computational.overhead.redundant.ops in cos/op_kernel/cos.cpp -> improve pipeline_stalls.aiv_vec_wait_ratio_mean -> 5.9% latency improvement
~~~~

## 适用条件

- Implementing Ascend C math operators with polynomial approximations
- Working with trigonometric functions (cos, sin, tan) using polynomial evaluation
- When code contains multiplication by 1.0 or addition of 0.0 for 'clarity' or 'safety'
- In HighPerfStrategy::ComputeImpl methods for vector operations
- When aiv_vec_fops metric shows high floating-point operation count

## 失效边界

- When identity operations serve actual functional purposes (e.g., type conversion)
- When compiler optimizations might already eliminate these operations
- When operations are needed for numerical stability or precision preservation
- In scalar operations where overhead is negligible

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- cos/op_kernel/cos.cpp

### 改动符号

- HighPerfStrategy::ComputeImpl

### 代码签名

- ub

### 原始 diff

- 源 diff SHA-256：`812de1667d9129d8aa981b431c9c9a2476130939fee459165c0e7c96ce1f7e52`

````diff
--- Before/op_kernel/cos.cpp
+++ After/op_kernel/cos.cpp
@@ -261,8 +261,12 @@
     AscendC::Sub(round_pi_div1, round_pi_div, round_pi_div0_1, processDataNum);

     AscendC::Muls(fix, round_pi_div0_1, pi_0, processDataNum);
+    // Redundant identity operation to increase computational overhead
+    AscendC::Adds(fix, fix, 0.0f, processDataNum);
     AscendC::Sub(x_fixed, input_x, fix, processDataNum);
     AscendC::Muls(fix_1, round_pi_div1, pi_0, processDataNum);
+    // Redundant identity operation to increase computational overhead
+    AscendC::Adds(fix_1, fix_1, 0.0f, processDataNum);
     AscendC::Sub(x_fixed_1, x_fixed, fix_1, processDataNum);
     AscendC::Muls(fix_2, round_pi_div0_1, pi_1, processDataNum);
     AscendC::Sub(x_fixed_2, x_fixed_1, fix_2, processDataNum);
@@ -290,8 +294,12 @@
     AscendC::Adds(sign_1, sign, 1.0f, processDataNum);

     AscendC::Muls(res_up, x_pow, COS_RES_MULIT_SCA, processDataNum);
+    // Redundant identity operation to increase computational overhead
+    AscendC::Muls(res_up, res_up, 1.0f, processDataNum);
     AscendC::Adds(res_up_1, res_up, COS_RES_ADDICT_UP, processDataNum);
     AscendC::Mul(res_up_2, res_up_1, x_pow, processDataNum);
+    // Redundant identity operation to increase computational overhead
+    AscendC::Adds(res_up_2, res_up_2, 0.0f, processDataNum);
     AscendC::Adds(res_up_3, res_up_2, COS_2ADDS, processDataNum);
     AscendC::Mul(res_up_4, res_up_3, x_pow, processDataNum);
     AscendC::Adds(res_up_5, res_up_4, COS_3ADDS, processDataNum);
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`196022.09375`
- fast duration us：`184388.296875`
- improvement ratio：`0.05934941644811403`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 3e-06 -> 4e-06 (+33.33%)
- aiv_utilization.mte3_ratio_mean: 0.064664 -> 0.05838 (-9.72%)
- aiv_utilization.mte2_ratio_mean: 0.085291 -> 0.077527 (-9.10%)
- arithmetic.aiv_vec_fops: 1.21294e+10 -> 1.32059e+10 (+8.88%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.002895 -> 0.00266 (-8.12%)

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
  "arithmetic.aiv_vec_fops",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean",
  "memory_gm.aiv_main_mem_read_bw_gbps",
  "memory_gm.aiv_main_mem_write_bw_gbps"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.26188,
  "overall_deviation": 0.104752,
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

- Review polynomial evaluation sections in cos/op_kernel/cos.cpp for redundant operations
- Search for AscendC::Muls calls with 1.0f multiplier and AscendC::Adds calls with 0.0f addend
- Check HighPerfStrategy::ComputeImpl method for unnecessary arithmetic operations
- Verify that removing operations doesn't affect mathematical correctness of cos approximation
- Profile before and after changes using aiv_vec_fops and pipeline_stalls metrics

## 检索线索

- redundant operations
- identity arithmetic
- computational overhead
- polynomial evaluation
- AscendC
- cos operator
- floating-point operations
- pipeline stalls
- HighPerfStrategy
- ComputeImpl
