---
type: Optimization
title: "heaviside：循环计算优化"
description: "heaviside 算子的循环计算优化经验，关注流水启动开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/docs/Heaviside.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/tf_plugin/tensorflow_add_custom_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_host/heaviside.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_host/heaviside_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_kernel/heaviside.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# heaviside：循环计算优化

## 来源信息

- experience ID：`e65e3526e69cb1b9`
- 源 schema version：`v2`
- 算子：`heaviside`
- source iteration：`round_004_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`loop.iteration.size.increase`
- canonical family：`loop.iteration.size.reduction`
- 源记录时间：`2026-04-20T19:27:05`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase loop iteration size to reduce pipeline overhead
~~~~

### Original summary

~~~~text
Increasing the loop iteration size from 50*512/sizeof(DTYPE_VALUES)/BufferNum to 100*512/sizeof(DTYPE_VALUES)/BufferNum reduces pipeline startup overhead by processing twice as many elements per iteration. This optimization improves performance by 5.4% for vector operations with high MTE wait ratios, particularly when scalar unit utilization increases significantly with smaller iteration sizes.
~~~~

### Original root mechanism

~~~~text
Larger loop iteration sizes reduce pipeline startup/shutdown overhead by processing more elements per iteration
~~~~

### Original causal chain

~~~~text
Increased elements per iteration from 50*512 to 100*512 → reduced loop iterations by half → decreased pipeline startup/shutdown overhead → reduced scalar unit utilization from 16.06% to 8.52% → decreased MTE2 wait ratio from 99.30% to 99.24% → improved vector wait ratio from 81.11% to 83.21% → reduced task duration from 1464.37μs to 1385.21μs
~~~~

## 适用条件

- Vector operators with loop-based processing in heaviside.cpp
- MTE wait ratios above 99% indicating memory transfer bottlenecks
- Scalar unit utilization increases significantly with smaller loop iterations
- Operators using fixed-size buffer processing with DTYPE_VALUES and BufferNum parameters

## 失效边界

- Memory-bound operators where larger iterations exceed UB capacity
- Operators with register pressure issues that worsen with larger iteration sizes
- When loop body contains complex control flow that doesn't benefit from larger iterations

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- heaviside/op_kernel/heaviside.cpp

### 改动符号

- KernelHeaviside::Process

### 代码签名

- ub

### 原始 diff

- 源 diff SHA-256：`81292306358645b32d07a8d0b7c8ab3526395cf9e1207b85b0c48932a26ddcad`

````diff
--- Before/op_kernel/heaviside.cpp
+++ After/op_kernel/heaviside.cpp
@@ -81,7 +81,7 @@
                 Process_iter(0, size);
             }
         } else {
-            uint32_t n_elements_per_iter = 100 * 512 / sizeof(DTYPE_VALUES) / BufferNum;
+            uint32_t n_elements_per_iter = 50 * 512 / sizeof(DTYPE_VALUES) / BufferNum;
             uint16_t repeatTimes = size / n_elements_per_iter;
             uint16_t remainSize = size % n_elements_per_iter;
             for (uint16_t i = 0; i < repeatTimes; i++) {
````

## 观测记录

- bottleneck type：`pipeline_startup_overhead`
- slow duration us：`1464.369263`
- fast duration us：`1385.207764`
- improvement ratio：`0.05405842706492266`
- machine confidence：`0.85`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.002913 -> 0.006235 (+114.04%)
- aiv_utilization.scalar_ratio_mean: 0.085182 -> 0.160621 (+88.56%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.016552 -> 0.020846 (+25.94%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.020584 -> 0.025256 (+22.70%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.004032 -> 0.00441 (+9.37%)

### Affected metric groups

- arithmetic
- aiv_utilization
- pipeline_stalls
- critical_path

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.002913,
      "after": 0.006235,
      "delta": 0.0033220000000000003,
      "rel": 1.140405080672846
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.085182,
      "after": 0.160621,
      "delta": 0.07543900000000002,
      "rel": 0.8856213754079504
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.016552,
      "after": 0.020846,
      "delta": 0.004293999999999999,
      "rel": 0.25942484291928464
    }
  },
  {
    "pipeline_stalls.aiv_vec_total_cflt_ratio_mean": {
      "before": 0.020584,
      "after": 0.025256,
      "delta": 0.004671999999999999,
      "rel": 0.22697240575204036
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.004032,
      "after": 0.00441,
      "delta": 0.00037799999999999986,
      "rel": 0.09374999999999997
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.142955,
  "overall_deviation": 0.057182,
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

- In heaviside.cpp KernelHeaviside::Process function, increase n_elements_per_iter calculation from 50*512 to 100*512
- Ensure total elements per iteration (n_elements_per_iter * sizeof(DTYPE_VALUES) * BufferNum) fits within available UB capacity
- Verify that increased iteration size doesn't cause register spilling by monitoring scalar ratio changes
- Apply to similar vector operators with high MTE wait ratios and low vector unit utilization

## 检索线索

- loop_iteration_size
- pipeline_overhead
- MTE_wait
- scalar_utilization
- heaviside
- vector_operator
- ascend_c
- performance_optimization
- buffer_processing
