---
type: Optimization
title: "fill：内存优化"
description: "fill 算子的内存优化经验，关注流水线停顿。"
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
# fill：内存优化

## 来源信息

- experience ID：`c3bc66e281077d8d`
- 源 schema version：`v2`
- 算子：`fill`
- source iteration：`round_016_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.access.pattern.cache.sequential`
- canonical family：`memory.access.pattern.cache.inefficient`
- 源记录时间：`2026-04-20T15:13:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Maintaining sequential memory access patterns for cache efficiency
~~~~

### Original summary

~~~~text
Applying sequential memory access patterns in Ascend C fill operators improves performance by 55.2% (from 87.4ms to 39.1ms). This optimization ensures memory accesses follow contiguous address patterns, which maximizes cache line utilization and reduces pipeline stalls. Use this technique when implementing scalar loops for fill operations to maintain cache locality and avoid performance degradation from non-sequential access patterns.
~~~~

### Original root mechanism

~~~~text
Sequential memory access patterns maximize cache line utilization by ensuring consecutive memory addresses are accessed together, reducing cache thrashing and improving memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Applying sequential memory access -> reduces cache thrashing -> improves memory bandwidth utilization (ub_write_bw_scalar_gbps: 1.49 → 3.33 Gbps, +124%) -> reduces pipeline stalls (aiv_vec_wait_ratio_mean: 0.0025 → 0.0057, -55.6%) -> decreases critical path time (aiv_vector0_time_us_mean: 87333 → 39076 µs, -55.3%) -> reduces overall latency (task_duration_us: 87361 → 39102 µs, -55.2%).
~~~~

## 适用条件

- Implementing scalar loops in Ascend C operators using SetValue API
- Processing large arrays with sequential memory access patterns
- Using LocalTensor<T> with contiguous memory access in fill operations
- When memory bandwidth utilization is below expected levels
- When aiv_vec_wait_ratio_mean indicates pipeline stalls from memory access patterns

## 失效边界

- Using non-sequential access patterns like prime-stride indexing (e.g., (i * prime) % size)
- Random or scattered memory access patterns that break cache locality
- When memory access patterns cannot be made sequential due to algorithm constraints
- When using vectorized operations that already maintain sequential access

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/fill.cpp

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

- 源 diff SHA-256：`8309b4e74c76119dad3c18ba381727a54c37c60bff5b6d01db8f21fabeef7f05`

````diff
--- Before/op_kernel/fill.cpp
+++ After/op_kernel/fill.cpp
@@ -67,8 +67,11 @@
         } else {
             AscendC::LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
             // Degradation: Replace vectorized Duplicate with scalar loop
+            // Further degradation: Non-sequential memory access pattern causing cache inefficiency
             for (uint32_t i = 0; i < this->processDataNum; ++i) {
-                outLocal.SetValue(i, this->value);
+                // Access elements with stride pattern that breaks cache locality
+                uint32_t idx = (i * 17) % this->processDataNum;  // Prime stride causes cache thrashing
+                outLocal.SetValue(idx, this->value);
             }
             outQueueOUT.EnQue<T>(outLocal);
         }
@@ -155,12 +158,16 @@

         // Degradation: Replace vectorized Duplicate with scalar loops
         // Process low and high parts separately in scalar fashion
+        // Further degradation: Non-sequential memory access pattern causing cache inefficiency
         for (uint32_t i = 0; i < this->processDataNum; i += 2) {
-            if (i < this->processDataNum) {
-                outLocal.SetValue(i, low);
-            }
-            if (i + 1 < this->processDataNum) {
-                outLocal.SetValue(i + 1, high);
+            // Access elements with stride pattern that breaks cache locality
+            uint32_t idx1 = (i * 17) % this->processDataNum;
+            uint32_t idx2 = ((i + 1) * 17) % this->processDataNum;
+            if (idx1 < this->processDataNum) {
+                outLocal.SetValue(idx1, low);
+            }
+            if (idx2 < this->processDataNum) {
+                outLocal.SetValue(idx2, high);
             }
         }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`87360.84375`
- fast duration us：`39102.320312`
- improvement ratio：`0.5524045025950198`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 39076.4 -> 87333 (+123.49%)
- latency.task_duration_us: 39102.3 -> 87360.8 (+123.42%)
- aiv_utilization.mte2_ratio_mean: 1.3e-05 -> 2.1e-05 (+61.54%)
- aiv_utilization.mte3_ratio_mean: 0.005849 -> 0.002578 (-55.92%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.005714 -> 0.002536 (-55.62%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "metric": "critical_path.aiv_vector0_time_us_mean",
    "before": 39076.434,
    "after": 87332.973,
    "delta": 48256.539,
    "rel_change": 1.2349268871361188
  },
  {
    "metric": "latency.task_duration_us",
    "before": 39102.32,
    "after": 87360.844,
    "delta": 48258.524,
    "rel_change": 1.2341601214454794
  },
  {
    "metric": "aiv_utilization.mte2_ratio_mean",
    "before": 1.3e-05,
    "after": 2.1e-05,
    "delta": 8e-06,
    "rel_change": 0.6153846153846154
  },
  {
    "metric": "aiv_utilization.mte3_ratio_mean",
    "before": 0.005849,
    "after": 0.002578,
    "delta": -0.003271,
    "rel_change": -0.5592408958796375
  },
  {
    "metric": "pipeline_stalls.aiv_vec_wait_ratio_mean",
    "before": 0.005714,
    "after": 0.002536,
    "delta": -0.003178,
    "rel_change": -0.5561778088904445
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

- In fill.cpp KernelFill::Compute function, ensure memory accesses follow sequential patterns: for (uint32_t i = 0; i < processDataNum; ++i) { outLocal.SetValue(i, value); }
- In fill.cpp KernelFill1_INT64::Compute function, maintain sequential access for both low and high parts: for (uint32_t i = 0; i < processDataNum; i += 2) { outLocal.SetValue(i, low); outLocal.SetValue(i + 1, high); }
- Monitor aiv_vec_wait_ratio_mean and memory_ub.ub_write_bw_scalar_gbps metrics to detect cache inefficiency
- When performance degrades by >50%, check for non-sequential memory access patterns in scalar loops
- Use contiguous index calculations instead of modulo operations with prime numbers for memory access

## 检索线索

- memory access pattern
- cache efficiency
- sequential access
- fill operator
- Ascend C
- SetValue
- LocalTensor
- cache thrashing
- memory bandwidth
- pipeline stalls
- scalar loops
- prime stride
- contiguous access
