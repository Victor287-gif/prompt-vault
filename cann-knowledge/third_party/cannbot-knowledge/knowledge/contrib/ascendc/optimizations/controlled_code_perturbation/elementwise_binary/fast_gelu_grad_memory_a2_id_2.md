---
type: Optimization
title: "fast_gelu_grad：内存优化"
description: "fast_gelu_grad 算子的内存优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/docs/FastGeluGrad.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/tf_plugin/tensorflow_fast_gelu_grad_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_host/fast_gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_host/fast_gelu_grad_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_kernel/fast_gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/FastGeluGrad_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/test_fast_gelu_grad.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# fast_gelu_grad：内存优化

## 来源信息

- experience ID：`9e1e41ec8cb4f97b`
- 源 schema version：`v2`
- 算子：`fast_gelu_grad`
- source iteration：`round_006_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.buffer.size`
- canonical family：`memory.buffer.size`
- 源记录时间：`2026-04-20T14:51:43`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Tile size reduction via memory divisor increase
~~~~

### Original summary

~~~~text
Increasing the NUM divisor constant in FastGeluGrad's TilingFunc reduces tile size, which improves vector unit utilization from 49.7% to 58.0% and reduces pipeline stalls. This optimization is effective when memory bandwidth to UB is underutilized and vector wait ratios are high. Apply this pattern to vector operators with similar tiling calculations to balance memory access patterns with computation efficiency.
~~~~

### Original root mechanism

~~~~text
Doubling the NUM constant from 7 to 14 in tiling calculation reduces tile size by half, decreasing vector wait ratio from 0.516 to 0.445 and improving vector utilization from 0.497 to 0.580.
~~~~

### Original causal chain

~~~~text
NUM constant increased from 7 to 14 → tile size halved → vector wait ratio decreased from 0.516 to 0.445 → vector utilization increased from 0.497 to 0.580 → task duration reduced from 1772μs to 1540μs
~~~~

## 适用条件

- Vector operators with TilingFunc using NUM divisor constants
- UB read bandwidth is underutilized (< 200 Gbps)
- Vector wait ratio is high (> 0.5)
- Vector utilization is below 60%
- Cache hit rates are near zero

## 失效边界

- Memory bandwidth to UB is already saturated (> 250 Gbps)
- Vector utilization is already high (> 80%)
- Tile size reduction would cause excessive loop overhead
- NUM divisor would exceed hardware memory alignment constraints

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- fast_gelu_grad/op_host/fast_gelu_grad.cpp

### 改动符号

- TilingFunc
- NUM

### 代码签名

- compile_time_specialization
- constexpr
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`83bb506aa0c9be0ac36667db130552b460ea6ba80dee4bf2739b784f8277c5bd`

````diff
--- Before/op_host/fast_gelu_grad.cpp
+++ After/op_host/fast_gelu_grad.cpp
@@ -11,7 +11,7 @@
 static ge::graphStatus TilingFunc(gert::TilingContext* context)
 {
     FastGeluGradTilingData tiling;
-    constexpr int32_t NUM = 7;
+    constexpr int32_t NUM = 14;
     uint32_t sizeofdatatype;
     uint32_t totalLengthAligned;
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1772.355469`
- fast duration us：`1539.570801`
- improvement ratio：`0.1313419751689778`
- machine confidence：`0.9`

### Profile evidence

- cache_l2.aiv_write_hit_rate_pct: 0 -> 3.57 (+100.00%)
- cache_l2.aiv_total_hit_rate_pct: 0 -> 2.05 (+100.00%)
- cache_l2.aiv_read_hit_rate_pct: 0 -> 1.85 (+100.00%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.171359 -> 0.062558 (-63.49%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.014107 -> 0.02256 (+59.92%)

### Affected metric groups

- cache_l2
- pipeline_stalls
- aiv_utilization
- memory_ub

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_write_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "cache_l2.aiv_read_hit_rate_pct",
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean",
  "aiv_utilization.scalar_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "memory_ub.ub_read_bw_vector_gbps",
  "aiv_utilization.mte2_ratio_mean"
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

- 1. Locate TilingFunc in op_host/fast_gelu_grad.cpp or similar vector operator files
- 2. Identify NUM or similar divisor constants in tile size calculations
- 3. Increase the divisor value (e.g., from 7 to 14) to reduce tile size
- 4. Verify tile size reduction improves vector utilization and reduces wait ratios
- 5. Monitor cache hit rates and UB bandwidth changes after optimization

## 检索线索

- fast_gelu_grad
- tiling
- NUM
- tile_size
- vector_utilization
- ub_bandwidth
- cache_hit
- pipeline_stalls
- memory_buffer
- ascendc
