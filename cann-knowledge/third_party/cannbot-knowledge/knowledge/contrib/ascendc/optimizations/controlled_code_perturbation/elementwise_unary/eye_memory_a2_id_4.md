---
type: Optimization
title: "eye：搬运与内存优化"
description: "eye 算子的搬运与内存优化经验，关注内存带宽。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access"]
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
# eye：搬运与内存优化

## 来源信息

- experience ID：`f43fa7e5cc094387`
- 源 schema version：`v2`
- 算子：`eye`
- source iteration：`round_017_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`memory.access_pattern`
- canonical family：`untitled`
- 源记录时间：`2026-04-20T16:01:33`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant global memory reads in eye kernel
~~~~

### Original summary

~~~~text
The eye operator performance is sensitive to unnecessary global memory reads before writes. Adding redundant GetValue() calls before SetValue() operations creates false dependencies and doubles memory traffic, reducing main memory bandwidth from 4.52 Gbps to 4.29 Gbps. This optimization removes those redundant reads while maintaining the same write pattern.
~~~~

### Original root mechanism

~~~~text
Removing unnecessary global memory reads before writes eliminates false dependencies and reduces memory traffic, improving memory bandwidth utilization by 5.09%
~~~~

### Original causal chain

~~~~text
Removing redundant yGm.GetValue() calls before yGm.SetValue() operations → reduces global memory read bandwidth from 4.29 Gbps to 4.52 Gbps → decreases scalar unit utilization from 18.08% to 16.71% → reduces total execution time from 2.916 ms to 2.765 ms
~~~~

## 适用条件

- Operator performs global memory writes without needing preceding reads
- Memory-bound operators with simple write-only patterns like eye, zeros, ones, or constant initialization
- Operators using yGm.SetValue() for initialization where values don't depend on existing memory content

## 失效边界

- Operators require actual data reads before writes (e.g., element-wise operations, reductions)
- Memory reads are necessary for computation (e.g., a = b + c)
- When compiler cannot optimize away redundant reads due to complex control flow

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 原始 diff

- 源 diff SHA-256：`c4e31c99352b0b2b000b51624fe5566d2c67ee5dca006b2b7662d1b56d5c78ac`

````diff
--- original_eye/op_kernel/eye.cpp	2026-04-11 14:18:47.703077613 +0800
+++ eye/op_kernel/eye.cpp	2026-04-20 14:34:44.700118706 +0800
@@ -26,24 +26,64 @@

         yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y_ref + this->blockLength * GetBlockIdx(), this->blockLength);
     }
-    __aicore__ inline void Process() {
+        __aicore__ inline void Process() {
         int32_t index, t;
         if(mark == 0){
+            // Write 1s on diagonal first
             for(int32_t i = 0; i < num_rows; i++){
                 index = i * num_columns + i;
                 if(i < num_columns){
+                    // Degradation: Add unnecessary global memory read before write
+                    // Creates false dependency and doubles memory traffic
+                    // Read from a location within the block to avoid out-of-bounds
+                    uint32_t read_idx = index % blockLength;
+                    DTYPE_Y unused = yGm.GetValue(read_idx);
+                    (void)unused;
                     yGm.SetValue(index, 1);
                 }
             }
+            // Degradation: Write 0 to non-diagonal elements in COLUMN-MAJOR order
+            // This creates terrible cache locality for row-major memory layout
+            for(int32_t k = 0; k < num_columns; k++) {
+                for(int32_t i = 0; i < num_rows; i++) {
+                    if(k != i) {
+                        // Degradation: Add unnecessary global memory read before write
+                        uint32_t write_idx = i * num_columns + k;
+                        uint32_t read_idx = write_idx % blockLength;
+                        DTYPE_Y unused = yGm.GetValue(read_idx);
+                        (void)unused;
+                        yGm.SetValue(write_idx, 0);
+                    }
+                }
+            }
         }else{
             for(int32_t i = 0; i < batchNum; i++){
+                // Write 1s on diagonal first for this batch
                 for(int32_t j = 0; j < num_rows; j++){
                     if(j < num_columns){
                         t = j * num_columns + j;
                         index = i * batchSize + t;
+                        // Degradation: Add unnecessary global memory read before write
+                        uint32_t read_idx = index % blockLength;
+                        DTYPE_Y unused = yGm.GetValue(read_idx);
+                        (void)unused;
                         yGm.SetValue(index, 1);
                     }
                 }
+                // Degradation: Write 0 to non-diagonal elements in COLUMN-MAJOR order
+                // This creates terrible cache locality for row-major memory layout
+                for(int32_t k = 0; k < num_columns; k++) {
+                    for(int32_t j = 0; j < num_rows; j++) {
+                        if(k != j) {
+                            // Degradation: Add unnecessary global memory read before write
+                            uint32_t write_idx = i * batchSize + j * num_columns + k;
+                            uint32_t read_idx = write_idx % blockLength;
+                            DTYPE_Y unused = yGm.GetValue(read_idx);
+                            (void)unused;
+                            yGm.SetValue(write_idx, 0);
+                        }
+                    }
+                }
             }
         }
     }
````

## 观测记录

- bottleneck type：`memory_bandwidth`
- slow duration us：`2915979.75`
- fast duration us：`2765318.0`
- improvement ratio：`0.05166762560679648`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.167143 -> 0.180798 (+8.17%)
- latency.task_duration_us: 2.76532e+06 -> 2.91598e+06 (+5.45%)
- critical_path.aiv_vector0_time_us_mean: 2.76526e+06 -> 2.91592e+06 (+5.45%)
- memory_gm.aiv_main_mem_read_bw_gbps: 4.52 -> 4.29 (-5.09%)
- memory_gm.aiv_main_mem_write_bw_gbps: 4.52 -> 4.29 (-5.09%)

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
      "before": 0.167143,
      "after": 0.180798,
      "delta": 0.013654999999999973,
      "rel": 0.08169651137050293
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2765318.0,
      "after": 2915979.75,
      "delta": 150661.75,
      "rel": 0.054482612849589086
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2765262.25,
      "after": 2915920.75,
      "delta": 150658.5,
      "rel": 0.05448253596923764
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 4.52,
      "after": 4.29,
      "delta": -0.22999999999999954,
      "rel": -0.050884955752212295
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 4.52,
      "after": 4.29,
      "delta": -0.22999999999999954,
      "rel": -0.050884955752212295
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

- Inspect op_kernel/*.cpp files for GetValue() calls immediately before SetValue() to same location
- Check if read value is unused (e.g., cast to void or not used in computation)
- Remove redundant GetValue() calls when writing constant values or initialized data
- Verify memory access patterns don't create false dependencies in Ascend C kernels
- Test with performance profiling to confirm memory bandwidth improvement

## 检索线索

- eye
- global_memory
- memory_bandwidth
- redundant_read
- false_dependency
- GetValue
- SetValue
- scalar_ratio
- memory_bound
- initialization
