---
type: Optimization
title: "fill：循环计算优化"
description: "fill 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/docs/Fill.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/framework/tf_plugin/tensorflow_sqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/op_host/fill.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/op_host/fill_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/op_kernel/fill.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/atk/aclnn_fill.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/atk/generate_fill.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/atk/op_fill.yaml
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/Fill_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/st/test_fill.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fill/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# fill：循环计算优化

## 来源信息

- experience ID：`9f75f91876ac635a`
- 源 schema version：`v2`
- 算子：`fill`
- source iteration：`round_017_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.access.pattern`
- canonical family：`loop.access.pattern`
- 源记录时间：`2026-04-20T15:23:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Sequential memory access pattern for fill operations
~~~~

### Original summary

~~~~text
Changing memory access patterns from non-sequential (i * 17) % processDataNum to sequential (i) indexing in fill kernels improves performance by 2.23x. This optimization reduces pipeline stalls and improves memory bandwidth utilization by ensuring contiguous memory access that leverages cache locality. Apply this pattern when writing scalar loops that access local tensor memory in Ascend C kernels.
~~~~

### Original root mechanism

~~~~text
Sequential memory access patterns enable better cache locality by reducing cache thrashing compared to non-sequential access patterns with prime-number strides
~~~~

### Original causal chain

~~~~text
Sequential access pattern -> reduced cache thrashing -> lower pipeline stall ratios (aiv_vec_wait_ratio_mean: -57.4%, aiv_mte3_wait_ratio_mean: -55.9%) -> improved memory bandwidth utilization (ub_write_bw_scalar_gbps: +124%, ub_to_gm_bw_usage_rate_pct: +124%) -> reduced kernel execution time (task_duration_us: -55.2%)
~~~~

## 适用条件

- Implementing fill operations with scalar loops in Ascend C kernels
- Writing to local tensor memory with SetValue API in loops
- Processing large data sets where cache locality matters
- Using TQue and AllocTensor for memory allocation in vector kernels

## 失效边界

- Using non-sequential access patterns like (i * prime) % size for memory access
- Implementing prime-number stride patterns that cause cache thrashing
- When vectorized operations (Duplicate) are available and appropriate
- When memory access patterns are already optimized for spatial locality

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- fill/op_kernel/fill.cpp

### 改动符号

- KernelFill::Compute
- KernelFill1_INT64::Compute

### 代码签名

- AllocTensor
- EnQue
- SetValue
- TQue
- contiguous_access
- queue_api
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`a6590fbf4855cc6c6e67d88400a7fbe95575813c75fd4ef66321e1a33e438051`

````diff
--- Before/op_kernel/fill.cpp
+++ After/op_kernel/fill.cpp
@@ -67,11 +67,9 @@
         } else {
             AscendC::LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
             // Degradation: Replace vectorized Duplicate with scalar loop
-            // Further degradation: Non-sequential memory access pattern causing cache inefficiency
+            // Refactored: Sequential memory access pattern for better cache locality
             for (uint32_t i = 0; i < this->processDataNum; ++i) {
-                // Access elements with stride pattern that breaks cache locality
-                uint32_t idx = (i * 17) % this->processDataNum;  // Prime stride causes cache thrashing
-                outLocal.SetValue(idx, this->value);
+                outLocal.SetValue(i, this->value);
             }
             outQueueOUT.EnQue<T>(outLocal);
         }
@@ -158,16 +156,13 @@

         // Degradation: Replace vectorized Duplicate with scalar loops
         // Process low and high parts separately in scalar fashion
-        // Further degradation: Non-sequential memory access pattern causing cache inefficiency
+        // Refactored: Sequential memory access pattern for better cache locality
         for (uint32_t i = 0; i < this->processDataNum; i += 2) {
-            // Access elements with stride pattern that breaks cache locality
-            uint32_t idx1 = (i * 17) % this->processDataNum;
-            uint32_t idx2 = ((i + 1) * 17) % this->processDataNum;
-            if (idx1 < this->processDataNum) {
-                outLocal.SetValue(idx1, low);
-            }
-            if (idx2 < this->processDataNum) {
-                outLocal.SetValue(idx2, high);
+            if (i < this->processDataNum) {
+                outLocal.SetValue(i, low);
+            }
+            if (i + 1 < this->processDataNum) {
+                outLocal.SetValue(i + 1, high);
             }
         }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`87360.84375`
- fast duration us：`39110.863281`
- improvement ratio：`0.5523067131434385`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 39076.2 -> 87333 (+123.49%)
- latency.task_duration_us: 39110.9 -> 87360.8 (+123.37%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.005955 -> 0.002536 (-57.41%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 5.9e-05 -> 2.6e-05 (-55.93%)
- aiv_utilization.mte3_ratio_mean: 0.005818 -> 0.002578 (-55.69%)

### Affected metric groups

- critical_path
- latency
- pipeline_stalls
- aiv_utilization

### Changed metrics top

~~~~json
[
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "memory_gm.aiv_main_mem_write_bw_gbps",
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "memory_ub.ub_write_bw_scalar_gbps",
  "aiv_utilization.mte2_ratio_mean",
  "aiv_utilization.scalar_ratio_mean"
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

- In fill.cpp KernelFill::Compute function, replace non-sequential indexing like (i * 17) % processDataNum with simple sequential indexing (i)
- In KernelFill1_INT64::Compute function, ensure both low and high value writes use contiguous indices (i and i+1) instead of permuted indices
- Use outLocal.SetValue(i, value) for sequential writes instead of outLocal.SetValue((i * 17) % processDataNum, value)
- Maintain the same loop structure but modify only the index calculation to ensure sequential memory access
- Apply this pattern to all scalar loops writing to local tensors allocated via AllocTensor

## 检索线索

- sequential_access
- cache_locality
- memory_access_pattern
- fill_kernel
- scalar_loop
- SetValue
- AllocTensor
- TQue
- pipeline_stall
- memory_bandwidth
