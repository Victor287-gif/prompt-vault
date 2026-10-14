---
type: Optimization
title: "cross：冗余消除优化"
description: "cross 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/docs/Cross.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/framework/tf_plugin/tensorflow_cross_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/op_host/cross.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/op_host/cross_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/op_kernel/cross.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/tests/st/Cross_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/tests/st/test_cross.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cross/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# cross：冗余消除优化

## 来源信息

- experience ID：`2dfa4720ed1c4b11`
- 源 schema version：`v2`
- 算子：`cross`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`redundant.bounds.check`
- canonical family：`redundant.bounds.check`
- 源记录时间：`2026-04-14T21:03:04`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant bounds checking in modulo index calculation
~~~~

### Original summary

~~~~text
Removing unnecessary boundary checks in modulo operations for tensor index calculations eliminates conditional branches from the hot path, improving instruction cache efficiency and reducing scalar operation overhead. This optimization applies to operators performing modulo-based index calculations where inputs are guaranteed to be non-negative and within bounds.
~~~~

### Original root mechanism

~~~~text
Redundant boundary condition checks in modulo operations introduce unnecessary conditional branches in tensor index computation hot paths.
~~~~

### Original causal chain

~~~~text
Removing redundant bounds checks → Reduced instruction cache misses (icache_miss_rate_mean: 7.6e-05 → 5.6e-05, -26.3%) and MTE2 utilization (mte2_ratio_mean: 0.000258 → 0.00021, -18.6%) → Lower scalar operation ratio (scalar_ratio_mean: 0.999683 → 0.992927, -0.68%) → Reduced task duration (1922.858μs → 1738.335μs, -9.6%)
~~~~

## 适用条件

- Tensor index calculations use modulo operations with non-negative inputs
- get_index functions perform bounds checking on modulo results
- Operators have scalar_ratio_mean > 0.99 indicating scalar-heavy computation
- Performance profiling shows high icache_miss_rate_mean (> 5e-05)

## 失效边界

- Input indices may be negative or exceed tensor bounds
- Modulo divisor may be zero or variable at runtime
- Bounds checking is required for safety-critical applications

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- cross/op_kernel/cross.cpp

### 改动符号

- KernelCross::get_index

### 原始 diff

- 源 diff SHA-256：`f822bd60ac9ea4e77cbd4decb4f78b758c8a34f526cea6bd12c0514932b7648f`

````diff
--- Before/op_kernel/cross.cpp
+++ After/op_kernel/cross.cpp
@@ -16,7 +16,11 @@
         for (int64_t k = numshapes - 1; k >= 0; --k) {
             const int64_t idx = i / step % outshape[k];
             step *= outshape[k];
-            index += idx % shape[j][k] * offset;
+            // Add redundant bounds checking and computation
+            int64_t mod_result = idx % shape[j][k];
+            if (mod_result < 0) mod_result = 0;  // Unnecessary check since idx and shape[j][k] are non-negative
+            if (mod_result >= shape[j][k]) mod_result = shape[j][k] - 1;  // Also unnecessary
+            index += mod_result * offset;
             offset *= shape[j][k];
         }
         return index;
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1922.858398`
- fast duration us：`1738.334717`
- improvement ratio：`0.09596321871227052`
- machine confidence：`0.9`

### Profile evidence

- memory_gm.aiv_main_mem_write_bw_gbps: 0.03 -> 0.02 (-33.33%)
- aiv_utilization.icache_miss_rate_mean: 7.6e-05 -> 5.6e-05 (-26.32%)
- aiv_utilization.mte2_ratio_mean: 0.000258 -> 0.00021 (-18.60%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0006 -> 0.0005 (-16.67%)
- memory_gm.aiv_main_mem_read_bw_gbps: 0.08 -> 0.07 (-12.50%)

### Affected metric groups

- memory_gm
- aiv_utilization
- memory_ub
- critical_path

### Changed metrics top

~~~~json
[
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 0.03,
      "after": 0.02,
      "delta": -0.009999999999999998,
      "rel": -0.3333333333333333
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 7.6e-05,
      "after": 5.6e-05,
      "delta": -2.0000000000000005e-05,
      "rel": -0.26315789473684215
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.000258,
      "after": 0.00021,
      "delta": -4.7999999999999974e-05,
      "rel": -0.1860465116279069
    }
  },
  {
    "memory_ub.ub_read_bw_scalar_gbps": {
      "before": 0.0006,
      "after": 0.0005,
      "delta": -9.999999999999994e-05,
      "rel": -0.16666666666666657
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 0.08,
      "after": 0.07,
      "delta": -0.009999999999999995,
      "rel": -0.12499999999999993
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1737.728,
      "after": 1922.311,
      "delta": 184.58299999999986,
      "rel": 0.10622088151885672
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1738.335,
      "after": 1922.858,
      "delta": 184.5229999999999,
      "rel": 0.10614927502466436
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.999683,
      "after": 0.992927,
      "delta": -0.006755999999999984,
      "rel": -0.006758142331118949
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 86.77,
      "after": 86.85,
      "delta": 0.0799999999999983,
      "rel": 0.0009219776420421609
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 90.01,
      "after": 90.07,
      "delta": 0.05999999999998806,
      "rel": 0.0006665926008219982
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

- Inspect get_index functions in op_kernel files for redundant bounds checks after modulo operations
- Remove unnecessary conditional branches like 'if (mod_result < 0) mod_result = 0' when inputs are guaranteed non-negative
- Remove unnecessary upper bound checks like 'if (mod_result >= shape[j][k]) mod_result = shape[j][k] - 1' when modulo operation already ensures result < divisor
- Profile the operator to verify reduced icache_miss_rate_mean and scalar_ratio_mean metrics
- Apply to KernelCross::get_index and similar index calculation patterns in other operators

## 检索线索

- modulo
- bounds_check
- index_calculation
- get_index
- conditional_branch
- scalar_ratio
- icache_miss
- tensor_indexing
- redundant_check
- performance_optimization
