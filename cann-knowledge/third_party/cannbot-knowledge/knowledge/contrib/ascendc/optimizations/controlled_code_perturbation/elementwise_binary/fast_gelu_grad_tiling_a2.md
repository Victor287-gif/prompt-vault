---
type: Optimization
title: "fast_gelu_grad：Tiling优化"
description: "fast_gelu_grad 算子的Tiling优化经验，关注Tile 数量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling"]
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
# fast_gelu_grad：Tiling优化

## 来源信息

- experience ID：`81557f8ccaeddd5d`
- 源 schema version：`v2`
- 算子：`fast_gelu_grad`
- source iteration：`round_019_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.constexpr_size`
- canonical family：`tiling.data.type.dependent`
- 源记录时间：`2026-04-20T16:50:28`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Compile-time constant tile sizing for Ascend C operators
~~~~

### Original summary

~~~~text
Applying compile-time constant tile sizes (constexpr NUM = 14) rather than data-type dependent tile calculations (NUM = BASE_NUM * sizeofdatatype) reduces tile count by ~2.1x and improves performance by 46.8%. This optimization is effective for Ascend C vector operators where tile size calculations are performed at compile time and data-type dependent sizing creates unnecessary tile fragmentation.
~~~~

### Original root mechanism

~~~~text
Using compile-time constant tile sizes instead of data-type dependent calculations reduces tile count and avoids aggressive rounding that disproportionately increases loop overhead for larger data types.
~~~~

### Original causal chain

~~~~text
Using constexpr tile size -> reduces tile count from data-type dependent fragmentation -> decreases aiv_vector0_time_us_mean from 5729.75μs to 2699.01μs (-53.0%) -> reduces task_duration_us from 5838.32μs to 2734.74μs (-53.2%)
~~~~

## 适用条件

- Ascend C vector operators with compile-time tile size calculations
- Operators using TilingFunc with ub_size/BLOCK_SIZE calculations
- When sizeofdatatype influences tile size in tiling_size = ((ub_size) / BLOCK_SIZE / 4) / NUM patterns
- Operators showing high aiv_vector0_time_us_mean with disproportionate tile count overhead

## 失效边界

- When data-type dependent tile sizing is required for memory alignment constraints
- When tile size must adapt dynamically to different data type sizes at runtime
- When using variable data types that cannot use compile-time constants

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- fast_gelu_grad/op_host/fast_gelu_grad.cpp

### 改动符号

- TilingFunc
- NUM
- tiling_size
- round_factor

### 代码签名

- compile_time_specialization
- constexpr
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`d551fc5a2457f04dc9a34a7ab04f3d95e9074260168b1db62c9556ca5cf43bd6`

````diff
--- Before/op_host/fast_gelu_grad.cpp
+++ After/op_host/fast_gelu_grad.cpp
@@ -11,7 +11,7 @@
 static ge::graphStatus TilingFunc(gert::TilingContext* context)
 {
     FastGeluGradTilingData tiling;
-    constexpr int32_t NUM = 14;
+    constexpr int32_t BASE_NUM = 14;
     uint32_t sizeofdatatype;
     uint32_t totalLengthAligned;
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
@@ -31,8 +31,12 @@
     }

     uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
+    // Degradation: Make tile size data-type dependent, reducing tile size for larger data types
+    uint32_t NUM = BASE_NUM * sizeofdatatype;
     uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 4) / NUM;
-    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
+    // Data-type dependent rounding: round down to multiple of 8*sizeofdatatype for larger types
+    uint32_t round_factor = 8 * sizeofdatatype;
+    tiling_size = tiling_size <= round_factor ? tiling_size : tiling_size / round_factor * round_factor;

     uint32_t block_size = tiling_size * ALIGN_NUM;
     aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
````

## 观测记录

- bottleneck type：`tile_count_overhead`
- slow duration us：`5838.316895`
- fast duration us：`2734.734619`
- improvement ratio：`0.5315885265936734`
- machine confidence：`0.9`

### Profile evidence

- latency.task_duration_us: 2734.74 -> 5838.32 (+113.49%)
- critical_path.aiv_vector0_time_us_mean: 2699.01 -> 5729.75 (+112.29%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.005246 -> 0 (-100.00%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.098927 -> 0.042364 (-57.18%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.093681 -> 0.042364 (-54.78%)

### Affected metric groups

- latency
- critical_path
- pipeline_stalls
- memory_ub

### Changed metrics top

~~~~json
[
  "latency.task_duration_us",
  "critical_path.aiv_vector0_time_us_mean",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean",
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean"
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

- In TilingFunc implementations, replace data-type dependent NUM calculations with constexpr constants
- Use constexpr int32_t NUM = 14 instead of uint32_t NUM = BASE_NUM * sizeofdatatype in fast_gelu_grad/op_host/fast_gelu_grad.cpp
- Avoid multiplying tile size factors by sizeofdatatype when calculating tiling_size
- Use simple rounding (tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8) instead of data-type dependent rounding_factor
- Verify tile count reduction by checking aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size)

## 检索线索

- tiling
- constexpr
- compile-time
- tile_size
- AscendC
- vector_operator
- fast_gelu_grad
- TilingFunc
- ub_size
- BLOCK_SIZE
- data_type_dependent
- tile_count
- loop_overhead
