---
type: Optimization
title: "eye：内存优化"
description: "eye 算子的内存优化经验，关注GM 访存。"
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
# eye：内存优化

## 来源信息

- experience ID：`62756902663b6483`
- 源 schema version：`v2`
- 算子：`eye`
- source iteration：`round_004_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`kernel.memory.read.before.write`
- canonical family：`kernel.memory.read.before.write`
- 源记录时间：`2026-04-20T14:31:12`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant global memory reads before writes
~~~~

### Original summary

~~~~text
In Ascend C kernels, avoid reading from global memory before writing to the same location when the read value is unused. This optimization reduces memory bandwidth usage and eliminates false dependencies that can serialize memory operations. Apply this technique when writing to global memory locations where the previous value is not needed for computation.
~~~~

### Original root mechanism

~~~~text
Removing unnecessary global memory read operations before write operations eliminates false dependencies and reduces memory traffic by 50%.
~~~~

### Original causal chain

~~~~text
Removing redundant yGm.GetValue() calls before yGm.SetValue() operations → Reduces memory_gm.aiv_main_mem_read_bw_gbps from 0.84 to 0.5 Gbps (-40.5%) and memory_gm.aiv_main_mem_write_bw_gbps from 0.84 to 0.5 Gbps (-40.5%) → Decreases critical_path.aiv_vector0_time_us_mean from 1.568M to 0.925M us (-41.0%) → Reduces total task_duration_us from 1.568M to 0.925M us (-41.0%).
~~~~

## 适用条件

- Kernel writes to global memory using yGm.SetValue() without needing previous values
- Pattern: yGm.GetValue() followed immediately by yGm.SetValue() to same or related location
- Unused read results stored in temporary variables like DTYPE_Y unused
- Memory bandwidth metrics show high aiv_main_mem_read_bw_gbps relative to write bandwidth

## 失效边界

- Previous memory location values are actually needed for computation
- Read-modify-write operations where the read value influences the write value
- Atomic operations or memory consistency requirements
- When reading from global memory is necessary for data dependency resolution

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- eye/op_kernel/eye.cpp

### 改动符号

- KernelEye::Process

### 代码签名

- GetValue
- SetValue
- bool_control
- scalar_value_access
- ub

### 原始 diff

- 源 diff SHA-256：`b4760e2843a2a8963214cdcc6590860c0daead38483f62b6b4695067c491a6df`

````diff
--- Before/op_kernel/eye.cpp
+++ After/op_kernel/eye.cpp
@@ -32,13 +32,24 @@
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
                 // Degradation: Write 0 to non-diagonal elements in a scattered pattern
                 // This creates inefficient memory access by writing to every element
                 for(int32_t k = 0; k < num_columns; k++) {
                     if(k != i) {
-                        yGm.SetValue(i * num_columns + k, 0);
+                        // Degradation: Add unnecessary global memory read before write
+                        uint32_t write_idx = i * num_columns + k;
+                        uint32_t read_idx = write_idx % blockLength;
+                        DTYPE_Y unused = yGm.GetValue(read_idx);
+                        (void)unused;
+                        yGm.SetValue(write_idx, 0);
                     }
                 }
             }
@@ -48,13 +59,22 @@
                     if(j < num_columns){
                         t = j * num_columns + j;
                         index = i * batchSize + t;
+                        // Degradation: Add unnecessary global memory read before write
+                        uint32_t read_idx = index % blockLength;
+                        DTYPE_Y unused = yGm.GetValue(read_idx);
+                        (void)unused;
                         yGm.SetValue(index, 1);
                  ... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`memory_gm`
- slow duration us：`1568295.25`
- fast duration us：`925024.5625`
- improvement ratio：`0.41017192872324265`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 925006 -> 1.56826e+06 (+69.54%)
- latency.task_duration_us: 925025 -> 1.5683e+06 (+69.54%)
- memory_gm.aiv_main_mem_read_bw_gbps: 0.84 -> 0.5 (-40.48%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.84 -> 0.5 (-40.48%)
- aiv_utilization.icache_miss_rate_mean: 0 -> 0 (+0.00%)

### Affected metric groups

- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 925005.562,
      "after": 1568263.375,
      "delta": 643257.813,
      "rel": 0.6954096704123385
    }
  },
  {
    "latency.task_duration_us": {
      "before": 925024.562,
      "after": 1568295.25,
      "delta": 643270.688,
      "rel": 0.6954093052504263
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 0.84,
      "after": 0.5,
      "delta": -0.33999999999999997,
      "rel": -0.40476190476190477
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 0.84,
      "after": 0.5,
      "delta": -0.33999999999999997,
      "rel": -0.40476190476190477
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

- Search for patterns: yGm.GetValue() calls where result is unused or cast to void
- Check if GetValue() reads are immediately followed by SetValue() to same buffer
- Remove GetValue() calls that create false dependencies without serving computational purpose
- Verify removal doesn't break data dependencies by examining variable usage
- Profile before/after to confirm memory bandwidth reduction (aiv_main_mem_read_bw_gbps metric)

## 检索线索

- global memory read before write
- redundant memory read
- false dependency
- memory bandwidth
- yGm.GetValue
- yGm.SetValue
- memory traffic
- Ascend C optimization
- eye operator
- kernel memory optimization
