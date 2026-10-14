---
type: Optimization
title: "fast_gelu_grad：内存优化"
description: "fast_gelu_grad 算子的内存优化经验，关注存储体冲突。"
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

- experience ID：`408e408763fb15b1`
- 源 schema version：`v2`
- 算子：`fast_gelu_grad`
- source iteration：`round_011_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.ub.allocation.ratio`
- canonical family：`memory.ub.allocation.ratio`
- 源记录时间：`2026-04-20T15:23:58`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase UB memory allocation ratio for tiling to reduce bank conflicts
~~~~

### Original summary

~~~~text
For vector operators like fast_gelu_grad, allocating more UB memory for tiling (increasing from 1/4 to 1/2 of available UB) reduces bank conflicts and improves performance. This optimization works by increasing tile size, which decreases loop iteration count and reduces pipeline stalls from bank conflicts. Apply this when vector operators show high aiv_vec_bank_cflt_ratio_mean and aiv_vec_bankgroup_cflt_ratio_mean values.
~~~~

### Original root mechanism

~~~~text
Increasing UB memory allocation ratio from quarter (1/4) to half (1/2) of available memory increases tile size, reducing loop overhead and bank conflicts while improving vector unit utilization.
~~~~

### Original causal chain

~~~~text
Increased UB allocation ratio from 1/4 to 1/2 → larger tile size → reduced loop iterations → decreased bank conflicts (aiv_vec_bank_cflt_ratio_mean -54.56%, aiv_vec_bankgroup_cflt_ratio_mean -33.36%) → improved vector utilization (vec_ratio_mean +7.07%) → reduced latency (task_duration_us -31.50%)
~~~~

## 适用条件

- Vector operators with high aiv_vec_bank_cflt_ratio_mean (>0.01)
- Operators showing pipeline stalls from bank conflicts
- When tiling_size calculation uses UB memory allocation ratio
- When aiv_vec_wait_ratio_mean is high (>0.45)
- When memory_ub.ub_read_bw_vector_gbps and ub_write_bw_vector_gbps show underutilization

## 失效边界

- Scalar-dominant operators
- When UB memory is already heavily utilized (>80% bandwidth)
- When aiv_vec_total_cflt_ratio_mean is already low (<0.05)
- When increasing tile size would exceed UB capacity
- When loop overhead is not the primary bottleneck

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- fast_gelu_grad/op_host/fast_gelu_grad.cpp

### 改动符号

- TilingFunc

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`ab18ea49d4f03696ace442f19c40cff38c87aaf91398ebb42749ce2d8fe1f42e`

````diff
--- Before/op_host/fast_gelu_grad.cpp
+++ After/op_host/fast_gelu_grad.cpp
@@ -31,7 +31,7 @@
     }

     uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
-    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
+    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 4) / NUM;
     tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

     uint32_t block_size = tiling_size * ALIGN_NUM;
````

## 观测记录

- bottleneck type：`memory_bank_conflicts`
- slow duration us：`2734.734619`
- fast duration us：`1873.217407`
- improvement ratio：`0.31502772006266094`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.011546 -> 0.005246 (-54.56%)
- latency.task_duration_us: 1873.22 -> 2734.74 (+45.99%)
- critical_path.aiv_vector0_time_us_mean: 1859.53 -> 2699.01 (+45.14%)
- aiv_utilization.scalar_ratio_mean: 0.256035 -> 0.368441 (+43.90%)
- aiv_utilization.mte3_ratio_mean: 0.140202 -> 0.193703 (+38.16%)

### Affected metric groups

- pipeline_stalls
- latency
- critical_path
- aiv_utilization

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "latency.task_duration_us",
  "critical_path.aiv_vector0_time_us_mean",
  "aiv_utilization.scalar_ratio_mean",
  "aiv_utilization.mte3_ratio_mean"
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

- 1. In TilingFunc in fast_gelu_grad.cpp, change UB allocation ratio from /4 to /2 in tiling_size calculation
- 2. Update line: uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 4) / NUM; to uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
- 3. Verify tile size alignment: tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
- 4. Monitor aiv_vec_bank_cflt_ratio_mean and aiv_vec_bankgroup_cflt_ratio_mean for reduction
- 5. Check aiv_utilization.vec_ratio_mean increases and scalar_ratio_mean decreases

## 检索线索

- UB
- tiling
- memory_allocation
- bank_conflicts
- vector
- fast_gelu_grad
- pipeline_stalls
- tiling_size
- ALIGN_NUM
- BLOCK_SIZE
