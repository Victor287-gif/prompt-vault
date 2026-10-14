---
type: Optimization
title: "gcd：冗余消除优化"
description: "gcd 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/Gcd.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/tensorflow_sqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd_common.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gcd：冗余消除优化

## 来源信息

- experience ID：`31a23694ef8b9d9d`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_006_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.computation`
- canonical family：`remove.redundant.computation`
- 源记录时间：`2026-04-20T20:12:08`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing redundant Muls and Adds operations in GCD kernel
~~~~

### Original summary

~~~~text
In GCD kernel implementations using tile-based processing, removing redundant Muls and Adds operations that compute intermediate values immediately overwritten by subsequent operations can improve performance by 6.4%. This optimization reduces scalar instruction overhead and decreases vector floating point operations (FLOPS) by 12.9%, particularly beneficial for vector operations with high arithmetic intensity.
~~~~

### Original root mechanism

~~~~text
Removing redundant Muls and Adds operations that compute values immediately overwritten by subsequent ShiftRight and Sub operations eliminates unnecessary arithmetic instructions and reduces scalar overhead
~~~~

### Original causal chain

~~~~text
Removing redundant Muls and Adds operations → reduces scalar_ratio_mean by 14.5% and aiv_vec_fops by 12.9% → decreases task_duration_us by 6.4% from 7360.9μs to 6891.9μs
~~~~

## 适用条件

- When implementing tile-based GCD kernels using Ascend C vector operations
- When code contains Muls or Adds operations whose results are immediately overwritten by subsequent operations like ShiftRight or Sub
- When scalar_ratio_mean is above 0.06 indicating significant scalar overhead
- When aiv_vec_fops shows high floating point operation count relative to workload size

## 失效边界

- When the intermediate values are actually used elsewhere in the computation
- When the operations have side effects beyond value computation
- When the code pattern is required for correctness or algorithm stability

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- gcd/op_kernel/gcd.cpp

### 改动符号

- KernelGcd::BinaryGcd

### 代码签名

- template
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`8b8e9ffd1bee83eece5248112a111875bb142868dfae205ff65b3254bc75baa7`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -224,11 +224,6 @@
         for (int i = 0; i < REP; i++) {
             Max(a0, a, b, TILE_SIZE);
             Min(b0, a, b, TILE_SIZE);
-
-            // Redundant computations that will be overwritten - degrade performance
-            Muls(a_div2, a, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
-            Muls(b_div2, b, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
-            Adds(ab_diff, a, (U)0, TILE_SIZE); // Will be overwritten by Sub and ShiftRight

             And(a_u16, a0_u16, ones_u16, num_uint16 * TILE_SIZE);
             And(b_u16, b0_u16, ones_u16, num_uint16 * TILE_SIZE);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`7360.947266`
- fast duration us：`6891.857422`
- improvement ratio：`0.06372683121460634`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.071695 -> 0.061295 (-14.51%)
- arithmetic.aiv_vec_fops: 2.52457e+08 -> 2.85217e+08 (+12.98%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.465752 -> 0.416837 (-10.50%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.141971 -> 0.127061 (-10.50%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.1 -> 0.09 (-10.00%)

### Affected metric groups

- aiv_utilization
- arithmetic
- memory_gm
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.071695,
      "after": 0.061295,
      "delta": -0.010399999999999993,
      "rel": -0.14505893019038976
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 252456960.0,
      "after": 285216768.0,
      "delta": 32759808.0,
      "rel": 0.12976393283037235
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.465752,
      "after": 0.416837,
      "delta": -0.048914999999999986,
      "rel": -0.10502370360191687
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.141971,
      "after": 0.127061,
      "delta": -0.014910000000000007,
      "rel": -0.10502144804220584
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 0.1,
      "after": 0.09,
      "delta": -0.010000000000000009,
      "rel": -0.10000000000000009
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

- Examine the KernelGcd::BinaryGcd function in gcd/op_kernel/gcd.cpp for redundant Muls and Adds operations
- Look for Muls operations that compute values immediately overwritten by ShiftRight operations
- Look for Adds operations that compute values immediately overwritten by Sub operations
- Remove lines like 'Muls(a_div2, a, (U)1, TILE_SIZE)' when a_div2 is overwritten by subsequent ShiftRight
- Remove lines like 'Adds(ab_diff, a, (U)0, TILE_SIZE)' when ab_diff is overwritten by subsequent Sub and ShiftRight
- Verify the removed operations don't affect algorithm correctness by checking data flow dependencies

## 检索线索

- gcd
- remove.redundant.computation
- scalar_overhead
- Muls
- Adds
- ShiftRight
- Sub
- tile
- ub
- vector_operations
- arithmetic_intensity
- performance_sensitivity
