---
type: Optimization
title: "eye：循环计算优化"
description: "eye 算子的循环计算优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise"]
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
# eye：循环计算优化

## 来源信息

- experience ID：`39ddab653e9c2b18`
- 源 schema version：`v2`
- 算子：`eye`
- source iteration：`round_019_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.fusion`
- canonical family：`loop.fusion`
- 源记录时间：`2026-04-20T16:11:07`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Fused separate diagonal and off-diagonal write loops into single nested loop
~~~~

### Original summary

~~~~text
Loop fusion of separate diagonal and off-diagonal write operations in the eye operator reduces scalar utilization by 33% and improves performance by 6.5%. This optimization works by eliminating redundant loop overhead and improving instruction cache efficiency when writing identity matrix elements. Apply this technique when implementing eye-like operators with conditional element initialization.
~~~~

### Original root mechanism

~~~~text
Combined two separate loops (one for diagonal 1s, one for off-diagonal 0s) into a single nested loop that writes all elements with condition k == i ? 1 : 0
~~~~

### Original causal chain

~~~~text
Loop fusion applied to KernelEye::Process in eye/op_kernel/eye.cpp → scalar_ratio_mean decreased from 0.180798 to 0.135849 (-33%) → aiv_vector0_time_us_mean reduced from 2915920.75 to 2725991.25 (-6.5%) → task_duration_us reduced from 2915979.75 to 2726046.25 (-6.5%)
~~~~

## 适用条件

- Implementing eye operators with conditional element initialization (diagonal vs off-diagonal)
- Writing identity matrices where elements depend on position (i == j)
- Using GetValue/SetValue patterns with global memory access in Ascend C kernels
- Operations with column-major access patterns on row-major memory layouts

## 失效边界

- Memory-bound operations where loop fusion increases register pressure
- Operations with complex conditional logic that prevents vectorization
- When separate loops provide better data locality for specific access patterns

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- eye/op_kernel/eye.cpp

### 改动符号

- KernelEye::Process

### 代码签名

- GetValue
- SetValue
- bool_control
- contiguous_access
- scalar_value_access
- ub

### 原始 diff

- 源 diff SHA-256：`7f8f33b06e0f90a5714dae65cc7d26bb97b231bfc79985f34c2c2f632517c479`

````diff
--- Before/op_kernel/eye.cpp
+++ After/op_kernel/eye.cpp
@@ -27,59 +27,39 @@
         yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y_ref + this->blockLength * GetBlockIdx(), this->blockLength);
     }
         __aicore__ inline void Process() {
-        int32_t index, t;
         if(mark == 0){
-            // Write 1s on diagonal first
-            for(int32_t i = 0; i < num_rows; i++){
-                index = i * num_columns + i;
-                if(i < num_columns){
+            // Combined loop for writing both diagonal and non-diagonal elements
+            // Maintains column-major order for cache locality degradation
+            for(int32_t k = 0; k < num_columns; k++) {
+                for(int32_t i = 0; i < num_rows; i++) {
+                    uint32_t write_idx = i * num_columns + k;
                     // Degradation: Add unnecessary global memory read before write
-                    // Creates false dependency and doubles memory traffic
-                    // Read from a location within the block to avoid out-of-bounds
-                    uint32_t read_idx = index % blockLength;
+                    uint32_t read_idx = write_idx % blockLength;
                     DTYPE_Y unused = yGm.GetValue(read_idx);
                     (void)unused;
-                    yGm.SetValue(index, 1);
-                }
-            }
-            // Degradation: Write 0 to non-diagonal elements in COLUMN-MAJOR order
-            // This creates terrible cache locality for row-major memory layout
-            for(int32_t k = 0; k < num_columns; k++) {
-                for(int32_t i = 0; i < num_rows; i++) {
-                    if(k != i) {
-                        // Degradation: Add unnecessary global memory read before write
-                        uint32_t write_idx = i * num_columns + k;
-                        uint32_t read_idx = write_idx % blockLength;
-                        DTYPE_Y unused = yGm.GetValue(read_... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2915979.75`
- fast duration us：`2726046.25`
- improvement ratio：`0.06513539745946452`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.135849 -> 0.180798 (+33.09%)
- latency.task_duration_us: 2.72605e+06 -> 2.91598e+06 (+6.97%)
- critical_path.aiv_vector0_time_us_mean: 2.72599e+06 -> 2.91592e+06 (+6.97%)
- memory_gm.aiv_main_mem_read_bw_gbps: 4.59 -> 4.29 (-6.54%)
- memory_gm.aiv_main_mem_write_bw_gbps: 4.59 -> 4.29 (-6.54%)

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
      "before": 0.135849,
      "after": 0.180798,
      "delta": 0.04494899999999999,
      "rel": 0.3308747211978004
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2726046.25,
      "after": 2915979.75,
      "delta": 189933.5,
      "rel": 0.06967361613912457
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2725991.25,
      "after": 2915920.75,
      "delta": 189929.5,
      "rel": 0.06967355452810056
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 4.59,
      "after": 4.29,
      "delta": -0.2999999999999998,
      "rel": -0.06535947712418297
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 4.59,
      "after": 4.29,
      "delta": -0.2999999999999998,
      "rel": -0.06535947712418297
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

- In KernelEye::Process function in eye/op_kernel/eye.cpp, replace separate diagonal and off-diagonal write loops
- Combine loops using conditional writes: for(k < num_columns) for(i < num_rows) yGm.SetValue(i*num_columns + k, (k == i) ? 1 : 0)
- Maintain column-major iteration order (k outer, i inner) to match original access pattern
- Preserve any necessary global memory read operations to maintain degradation characteristics if required

## 检索线索

- eye
- loop fusion
- diagonal initialization
- identity matrix
- scalar utilization
- KernelEye
- GetValue
- SetValue
- column-major
- global memory
- Ascend C
- math operators
