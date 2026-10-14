---
type: Optimization
title: "arange：Tiling优化"
description: "arange 算子的Tiling优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/docs/Arange.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/tf_plugin/tensorflow_arange_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_host/arange.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_host/arange_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_kernel/arange.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/Arange_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/test_arange.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# arange：Tiling优化

## 来源信息

- experience ID：`16f36259a7fe6070`
- 源 schema version：`v2`
- 算子：`arange`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.inefficient.ub.utilization`
- canonical family：`tiling.inefficient.ub.utilization`
- 源记录时间：`2026-04-14T20:59:47`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase UB allocation divisor to reduce per-iteration memory usage
~~~~

### Original summary

~~~~text
For Ascend C operators using UB memory for tiling, increasing the divisor in DIVIDE_AND_ALIGN calls reduces per-iteration UB memory allocation, forcing more loop iterations but improving vector utilization from 70.26% to 89.52%. This optimization reduces pipeline wait ratios from 42.63% to 17.43% and improves GM write bandwidth from 40.61 Gbps to 64.99 Gbps. Apply this technique when pipeline stalls (aiv_vec_wait_ratio_mean) are high and vector utilization is below optimal levels.
~~~~

### Original root mechanism

~~~~text
Increasing the UB allocation divisor from 10 to 20 reduces the amount of UB memory used per iteration, which increases the number of loop iterations required to process the same total data, improving vector utilization and reducing pipeline stalls.
~~~~

### Original causal chain

~~~~text
Increased UB allocation divisor from 10 to 20 in DIVIDE_AND_AND_ALIGN(ub_size, divisor, BLOCK_SIZE) -> Reduced per-iteration UB memory usage -> Increased loop iteration count -> Improved vector utilization (70.26% → 89.52%) and reduced pipeline stalls (aiv_vec_wait_ratio_mean 42.63% → 17.43%) → Reduced latency from 9262.92μs to 5751.09μs (38% improvement)
~~~~

## 适用条件

- Operator uses UB memory for tiling with DIVIDE_AND_ALIGN pattern
- Pipeline stalls (aiv_vec_wait_ratio_mean) exceed 30%
- Vector utilization (aiv_utilization.vec_ratio_mean) is below 80%
- UB to GM bandwidth usage rate (memory_ub.ub_to_gm_bw_usage_rate_pct) is below 30%
- Operator has sufficient total work to benefit from increased loop iterations

## 失效边界

- Operator already has high vector utilization (>90%)
- Pipeline stalls are already low (<10%)
- Memory bandwidth is already saturated (>80% usage rate)
- Total data size is small (<1MB)
- Loop overhead would dominate computation

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/arange.cpp

### 改动符号

- optiling::TilingFunc

### 代码签名

- ub

### 原始 diff

- 源 diff SHA-256：`b0c41374d50e91e9eeaf934a7f1a2ca983a7c5075b241105b93c44987f119754`

````diff
--- Before/op_host/arange.cpp
+++ After/op_host/arange.cpp
@@ -40,8 +40,8 @@
         uint64_t ub_size;
         auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
         ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
-        /*单次api计算大小：将ub 10等份后并按BLOCK_SIZE对齐*/
-        uint64_t ub_unit_size = DIVIDE_AND_ALIGN(ub_size, 10, BLOCK_SIZE);
+        /*单次api计算大小：将ub 20等份后并按BLOCK_SIZE对齐*/
+        uint64_t ub_unit_size = DIVIDE_AND_ALIGN(ub_size, 20, BLOCK_SIZE);
         uint32_t totalNum = totalLength;
         uint32_t unitNum  = ub_unit_size / dtype_size;
         uint32_t unitLoops = totalNum / unitNum;
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`9262.924805`
- fast duration us：`5751.094727`
- improvement ratio：`0.3791275598074987`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.174277 -> 0.426273 (+144.60%)
- cache_l2.aiv_total_hit_rate_pct: 5.25 -> 11.78 (+124.38%)
- cache_l2.aiv_write_hit_rate_pct: 5.25 -> 11.78 (+124.38%)
- memory_ub.ub_write_bw_scalar_gbps: 0.0127 -> 0.004 (-68.50%)
- critical_path.aiv_vector0_time_us_mean: 5750.47 -> 9262.16 (+61.07%)

### Affected metric groups

- pipeline_stalls
- cache_l2
- memory_ub
- critical_path

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.174277,
      "after": 0.426273,
      "delta": 0.251996,
      "rel": 1.445950986073894
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 5.25,
      "after": 11.78,
      "delta": 6.529999999999999,
      "rel": 1.2438095238095237
    }
  },
  {
    "cache_l2.aiv_write_hit_rate_pct": {
      "before": 5.25,
      "after": 11.78,
      "delta": 6.529999999999999,
      "rel": 1.2438095238095237
    }
  },
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.0127,
      "after": 0.004,
      "delta": -0.0087,
      "rel": -0.6850393700787402
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 5750.471,
      "after": 9262.158,
      "delta": 3511.687,
      "rel": 0.610678151407076
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

- Locate UB allocation divisor in op_host/*.cpp files using DIVIDE_AND_ALIGN(ub_size, divisor, BLOCK_SIZE)
- Increase divisor value (e.g., from 10 to 20) to reduce per-iteration UB memory allocation
- Verify UB unit size calculation: ub_unit_size = DIVIDE_AND_AND_ALIGN(ub_size, divisor, BLOCK_SIZE)
- Monitor aiv_vec_wait_ratio_mean reduction and aiv_utilization.vec_ratio_mean improvement
- Check that ub_to_gm_bw_usage_rate_pct increases and task_duration_us decreases

## 检索线索

- DIVIDE_AND_ALIGN
- UB
- tiling
- pipeline_stalls
- vector_utilization
- ub_allocation
- arange
- ascendc
- memory_bandwidth
- loop_iterations
