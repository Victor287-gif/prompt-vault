---
type: Optimization
title: "eye：内存与 Tiling优化"
description: "eye 算子的内存与 Tiling优化经验，关注Kernel 启动开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/docs/Eye.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/aclnn_eye_operation.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/aclnn_eye_operation.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/readme.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/tf_plugin/tensorflow_add_custom_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_host/eye.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_host/eye_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_kernel/eye.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/Eye_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/test_eye.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# eye：内存与 Tiling优化

## 来源信息

- experience ID：`6664797112309334`
- 源 schema version：`v2`
- 算子：`eye`
- source iteration：`round_007_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.buffer.overestimation`
- canonical family：`tiling.buffer.overestimation`
- 源记录时间：`2026-04-20T14:52:44`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Optimize DATA_NUM constant in tiling calculation to maximize UB utilization
~~~~

### Original summary

~~~~text
In Ascend C tiling calculations, the DATA_NUM constant directly impacts UB block allocation. Setting DATA_NUM too high artificially reduces the number of UB blocks available per tile (ub_block_num), forcing more tiles to be created and increasing kernel launch overhead. Apply this optimization when tuning tiling parameters for memory-bound operators to maximize UB utilization and minimize tile count.
~~~~

### Original root mechanism

~~~~text
Increasing DATA_NUM constant reduces available UB blocks per tile (ub_block_num = ub_size / BLOCK_SIZE / dataNum - 256), forcing more tiles and increasing kernel launch overhead.
~~~~

### Original causal chain

~~~~text
apply tiling.buffer.overestimation in eye/op_host/eye.cpp -> improve aiv_utilization.scalar_ratio_mean -> 5.3% latency improvement
~~~~

## 适用条件

- Operator uses Ascend C tiling with UB block calculations
- Tiling function contains DATA_NUM or similar constants affecting ub_block_num calculation
- Performance shows high scalar utilization with low vector utilization
- Tile count appears higher than necessary for the workload size

## 失效边界

- Operator is compute-bound rather than memory-bound
- UB utilization is already near 100%
- Tile count is minimal (1-2 tiles)
- DATA_NUM is determined by hardware constraints or data type requirements

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- eye/op_host/eye.cpp

### 改动符号

- optiling::TilingFunc

### 代码签名

- compile_time_specialization
- constexpr
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`121903ec8508a03cfc523ba11d6e14c025cef49822f2cd30b33f26a492723f72`

````diff
--- Before/op_host/eye.cpp
+++ After/op_host/eye.cpp
@@ -9,7 +9,7 @@
     constexpr uint32_t DATATYPE_SIZE_4 = 4;
     constexpr uint32_t TYPE_KEY_0 = 0;
     constexpr uint32_t TYPE_KEY_1 = 1;
-    constexpr uint32_t DATA_NUM = 4;
+    constexpr uint32_t DATA_NUM = 8;
     static ge::graphStatus TilingFunc(gert::TilingContext* context) {
         EyeTilingData tiling;
         uint32_t sizeofdatatype;
````

## 观测记录

- bottleneck type：`kernel_launch_overhead`
- slow duration us：`2915979.75`
- fast duration us：`2761094.5`
- improvement ratio：`0.053116023868135576`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.124802 -> 0.180798 (+44.87%)
- latency.task_duration_us: 2.76109e+06 -> 2.91598e+06 (+5.61%)
- critical_path.aiv_vector0_time_us_mean: 2.76104e+06 -> 2.91592e+06 (+5.61%)
- memory_gm.aiv_main_mem_read_bw_gbps: 4.53 -> 4.29 (-5.30%)
- memory_gm.aiv_main_mem_write_bw_gbps: 4.53 -> 4.29 (-5.30%)

### Affected metric groups

- aiv_utilization
- latency
- critical_path
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.124802,
      "after": 0.180798,
      "delta": 0.05599599999999999,
      "rel": 0.44867870707200197
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2761094.5,
      "after": 2915979.75,
      "delta": 154885.25,
      "rel": 0.056095599045958046
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2761038.75,
      "after": 2915920.75,
      "delta": 154882.0,
      "rel": 0.05609555461689916
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 4.53,
      "after": 4.29,
      "delta": -0.2400000000000002,
      "rel": -0.05298013245033117
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 4.53,
      "after": 4.29,
      "delta": -0.2400000000000002,
      "rel": -0.05298013245033117
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.121912,
  "overall_deviation": 0.048765,
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

- Examine tiling calculations in op_host/*.cpp files for DATA_NUM or similar constants
- Check ub_block_num calculation formula: ub_size / BLOCK_SIZE / dataNum - 256
- Profile operator with different DATA_NUM values to find optimal tile count
- Monitor scalar_ratio_mean metric - increases indicate inefficient tiling
- Balance DATA_NUM to maximize ub_block_num while maintaining data alignment requirements
- For eye operator, DATA_NUM=4 provided better performance than DATA_NUM=8

## 检索线索

- DATA_NUM
- ub_block_num
- tiling
- UB utilization
- kernel launch overhead
- scalar ratio
- eye operator
- Ascend C
- tiling calculation
- buffer overestimation
