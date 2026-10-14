---
type: Optimization
title: "tril：Tiling优化"
description: "tril 算子的Tiling优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.tiling"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/docs/Tril.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationATBPlugin/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationATBPlugin/aclnn_tril_operation.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationATBPlugin/aclnn_tril_operation.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationATBPlugin/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/script/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/script/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/script/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/readme.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/framework/tf_plugin/tensorflow_tril_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/op_host/tril.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/op_host/tril_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/op_kernel/tril.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/Tril_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/test_tril.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# tril：Tiling优化

## 来源信息

- experience ID：`cafb3b0cb9ae5834`
- 源 schema version：`v2`
- 算子：`tril`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.inefficiency`
- canonical family：`tiling.inefficiency`
- 源记录时间：`2026-04-14T21:24:45`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Optimize UB buffer sharing for better tile sizing
~~~~

### Original summary

~~~~text
Adjusting the UB_SHARING_NUM parameter from 8 to 4 in tril.cpp increases the available buffer per queue, allowing for larger tile sizes and fewer loop iterations. This optimization improves vector unit utilization from 35.4% to 46.5% and reduces MTE wait ratios while increasing GM memory bandwidth from 62.24 Gbps to 84.26 Gbps. Apply this optimization when UB buffer sharing is limiting tile size in vector operations with high memory bandwidth requirements.
~~~~

### Original root mechanism

~~~~text
Reducing the UB_SHARING_NUM divisor from 8 to 4 increases available buffer per queue, enabling larger tiles and reducing loop iterations.
~~~~

### Original causal chain

~~~~text
UB_SHARING_NUM reduced from 8 to 4 → larger tile size per queue → fewer loop iterations → reduced vector wait ratio from 74.4% to 59.8% and increased vector utilization from 35.4% to 46.5% → improved memory bandwidth utilization from 62.24 Gbps to 84.26 Gbps → reduced task duration from 2008.88μs to 1484.17μs
~~~~

## 适用条件

- Vector operations with UB buffer sharing limiting tile size
- Operators showing high MTE wait ratios (>99%) and low vector utilization (<40%)
- Memory-bound operations where GM bandwidth usage is below theoretical maximum
- Ascend C kernels using UB_SHARING_NUM parameter to distribute buffer across queues

## 失效边界

- Scalar-dominant operations with minimal vector unit usage
- Operations already achieving near-peak memory bandwidth (>90% utilization)
- When UB buffer capacity is not the limiting factor for tile sizing
- Kernels with complex dependency patterns where smaller tiles might reduce stalls

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- tril/op_host/tril.cpp

### 改动符号

- UB_SHARING_NUM

### 代码签名

- compile_time_specialization
- constexpr
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`e7f75245c46ec4ea146994fbd6bda07ec2aa8282dd2a93814080dd38648713e7`

````diff
--- Before/op_host/tril.cpp
+++ After/op_host/tril.cpp
@@ -15,7 +15,7 @@
     constexpr int VAL_ZRRO = 0;

     uint32_t typeSize = VAL_ZRRO;
-    uint64_t UB_SHARING_NUM = bufferFour;
+    uint64_t UB_SHARING_NUM = bufferFour * 2; // Degradation: artificially reduce available buffer per queue
     int64_t rowLength = VAL_ZRRO;
     int64_t columnLength = VAL_ZRRO;
     int64_t matrixNum = 1, matrixSize = 1;
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2008.880127`
- fast duration us：`1484.169678`
- improvement ratio：`0.26119549989455393`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.521871 -> 0.755986 (+44.86%)
- aiv_utilization.icache_miss_rate_mean: 0.000157 -> 9.1e-05 (-42.04%)
- critical_path.aiv_vector0_time_us_mean: 1483.54 -> 2008.31 (+35.37%)
- latency.task_duration_us: 1484.17 -> 2008.88 (+35.35%)
- memory_gm.aiv_main_mem_read_bw_gbps: 84.26 -> 62.24 (-26.13%)

### Affected metric groups

- aiv_utilization
- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.521871,
      "after": 0.755986,
      "delta": 0.23411500000000007,
      "rel": 0.44860703123952106
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 0.000157,
      "after": 9.1e-05,
      "delta": -6.599999999999999e-05,
      "rel": -0.4203821656050955
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1483.538,
      "after": 2008.306,
      "delta": 524.768,
      "rel": 0.3537273733466888
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1484.17,
      "after": 2008.88,
      "delta": 524.71,
      "rel": 0.35353766751787197
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 84.26,
      "after": 62.24,
      "delta": -22.020000000000003,
      "rel": -0.2613339662948018
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

- 1. Check the UB_SHARING_NUM parameter in the operator's host code (e.g., tril.cpp)
- 2. Reduce UB_SHARING_NUM by factors of 2 (e.g., from 8 to 4, 4 to 2) to increase available buffer per queue
- 3. Verify that vector utilization (aiv_utilization.vec_ratio_mean) increases and MTE wait ratios (aiv_mte2_wait_ratio_mean, aiv_mte3_wait_ratio_mean) decrease
- 4. Monitor GM memory bandwidth (aiv_main_mem_read_bw_gbps, aiv_main_mem_write_bw_gbps) for improvement
- 5. Ensure task duration (task_duration_us) decreases with the change

## 检索线索

- UB_SHARING_NUM
- tile size
- buffer sharing
- vector utilization
- MTE wait
- memory bandwidth
- tril
- Ascend C
- tiling optimization
