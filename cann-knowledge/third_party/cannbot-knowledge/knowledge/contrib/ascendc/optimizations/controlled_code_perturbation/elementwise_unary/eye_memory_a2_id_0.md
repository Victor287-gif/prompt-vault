---
type: Optimization
title: "eye：内存优化"
description: "eye 算子的内存优化经验，关注混合或尚未确定的瓶颈。"
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

- experience ID：`2c45b3b4bd94b7f5`
- 源 schema version：`v2`
- 算子：`eye`
- source iteration：`round_002_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`kernel.memory.scatter`
- canonical family：`kernel.memory.scatter`
- 源记录时间：`2026-04-20T14:06:05`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant scattered memory writes for diagonal matrix initialization
~~~~

### Original summary

~~~~text
For diagonal matrix initialization operators like Eye, avoid writing zeros to non-diagonal elements through scattered memory accesses. Instead, rely on the output buffer being pre-initialized to zero and only write the diagonal 1 values. This optimization reduces L2 cache misses from 12.5% to 0.02% and improves memory bandwidth utilization from 0.84 Gbps to 4.09 Gbps by eliminating unnecessary write operations.
~~~~

### Original root mechanism

~~~~text
Avoiding redundant scattered writes to zero out non-diagonal matrix elements when the output buffer is already pre-initialized to zero.
~~~~

### Original causal chain

~~~~text
Removing redundant scattered writes -> reduces L2 cache misses from 12.5% to 0.02% -> improves memory bandwidth from 0.84 to 4.09 Gbps -> reduces vector unit time from 925,006 us to 1,193 us -> improves overall task duration from 925,025 us to 1,194 us
~~~~

## 适用条件

- Implementing diagonal matrix initialization operators (Eye-like functions)
- Output buffer is guaranteed to be pre-initialized to zero before kernel execution
- Kernel writes scattered pattern to memory with SetValue operations
- Non-diagonal elements need to be zero but buffer already contains zeros

## 失效边界

- Output buffer is not guaranteed to be pre-initialized to zero
- Non-diagonal elements may contain non-zero values that need explicit clearing
- Memory access pattern is already coalesced and optimized

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- eye/op_kernel/eye.cpp

### 改动符号

- KernelEye::Process

### 代码签名

- SetValue
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`d3cb56e0f853784593371da2203631a6b9f75f0e0cc2d13d779fe6a60e4097d9`

````diff
--- Before/op_kernel/eye.cpp
+++ After/op_kernel/eye.cpp
@@ -34,6 +34,13 @@
                 if(i < num_columns){
                     yGm.SetValue(index, 1);
                 }
+                // Degradation: Write 0 to non-diagonal elements in a scattered pattern
+                // This creates inefficient memory access by writing to every element
+                for(int32_t k = 0; k < num_columns; k++) {
+                    if(k != i) {
+                        yGm.SetValue(i * num_columns + k, 0);
+                    }
+                }
             }
         }else{
             for(int32_t i = 0; i < batchNum; i++){
@@ -42,6 +49,13 @@
                         t = j * num_columns + j;
                         index = i * batchSize + t;
                         yGm.SetValue(index, 1);
+                    }
+                    // Degradation: Write 0 to non-diagonal elements in a scattered pattern
+                    // This creates inefficient memory access by writing to every element
+                    for(int32_t k = 0; k < num_columns; k++) {
+                        if(k != j) {
+                            yGm.SetValue(i * batchSize + j * num_columns + k, 0);
+                        }
                     }
                 }
             }
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`925024.5625`
- fast duration us：`1193.783936`
- improvement ratio：`0.9987094570410394`
- machine confidence：`0.9`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 0.02 -> 87.5 (+437400.00%)
- critical_path.aiv_vector0_time_us_mean: 1193.17 -> 925006 (+77424.98%)
- latency.task_duration_us: 1193.78 -> 925025 (+77386.76%)
- aiv_utilization.icache_miss_rate_mean: 0.000103 -> 0 (-100.00%)
- aiv_utilization.mte2_ratio_mean: 8.7e-05 -> 0 (-100.00%)

### Affected metric groups

- cache_l2
- critical_path
- latency
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 0.02,
      "after": 87.5,
      "delta": 87.48,
      "rel": 4374.0
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1193.171,
      "after": 925005.562,
      "delta": 923812.3910000001,
      "rel": 774.2497856551995
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1193.784,
      "after": 925024.562,
      "delta": 923830.778,
      "rel": 773.8676159171173
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 0.000103,
      "after": 0.0,
      "delta": -0.000103,
      "rel": -1.0
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 8.7e-05,
      "after": 0.0,
      "delta": -8.7e-05,
      "rel": -1.0
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

- Check if output buffer is pre-initialized to zero before kernel execution in eye/op_kernel/eye.cpp
- In KernelEye::Process function, verify that only diagonal elements need explicit setting to 1
- Remove loops that write zeros to non-diagonal elements via scattered SetValue calls
- Focus only on setting diagonal elements: yGm.SetValue(index, 1) where index corresponds to diagonal positions
- Avoid writing zeros to positions (i * num_columns + k) where k != i in the inner loops

## 检索线索

- eye
- diagonal matrix
- scattered memory writes
- redundant writes
- cache misses
- memory bandwidth
- SetValue
- kernel.memory.scatter
- pre-initialized buffer
- zero initialization
