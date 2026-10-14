---
type: Optimization
title: "fill：计算优化"
description: "fill 算子的计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline"]
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
# fill：计算优化

## 来源信息

- experience ID：`dada0eb96f24e9d4`
- 源 schema version：`v2`
- 算子：`fill`
- source iteration：`round_017_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`arithmetic.inefficient.modulo`
- canonical family：`arithmetic.inefficient.modulo`
- 源记录时间：`2026-04-20T15:21:46`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use hardware modulo instead of subtraction loop for index calculation
~~~~

### Original summary

~~~~text
Use hardware modulo operations instead of software subtraction loops for index calculations in Ascend C kernels. The hardware modulo instruction is significantly faster than implementing modulo through repeated subtraction in a while-loop. This optimization reduces arithmetic intensity and pipeline stalls in compute-bound kernels.
~~~~

### Original root mechanism

~~~~text
Replacing hardware modulo operation with a while-loop subtraction in the inner compute loop adds O(n) overhead per element, increasing arithmetic intensity and pipeline stalls.
~~~~

### Original causal chain

~~~~text
Replace subtraction loop with hardware modulo -> reduces arithmetic operations per element -> decreases scalar ratio from 0.998684 to 0.997464 -> reduces aiv_vector0_time_us_mean from 169593us to 87333us -> improves task_duration_us by 94%
~~~~

## 适用条件

- Index calculations use modulo operations on loop counters
- Kernel uses while-loops for arithmetic operations in compute-bound sections
- Performance profile shows high scalar_ratio_mean (>0.99) indicating scalar-bound execution
- Code contains patterns like 'while (idx >= n) { idx -= n; }' for modulo calculation

## 失效边界

- Modulo divisor is not a compile-time constant
- Hardware modulo instruction is not supported for the specific data type
- Divisor is a power of two (use bitwise AND instead)
- Performance is memory-bound rather than compute-bound

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
- ub

### 原始 diff

- 源 diff SHA-256：`fc375c432fa6e4b76a62dd0d39d27c85d35609c7d8d6517077fb67451cb5343d`

````diff
--- Before/op_kernel/fill.cpp
+++ After/op_kernel/fill.cpp
@@ -68,9 +68,13 @@
             AscendC::LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
             // Degradation: Replace vectorized Duplicate with scalar loop
             // Further degradation: Non-sequential memory access pattern causing cache inefficiency
+            // Additional degradation: Inefficient modulo calculation using subtraction loop
             for (uint32_t i = 0; i < this->processDataNum; ++i) {
-                // Access elements with stride pattern that breaks cache locality
-                uint32_t idx = (i * 17) % this->processDataNum;  // Prime stride causes cache thrashing
+                // Compute idx = (i * 17) % processDataNum using slow subtraction loop
+                uint32_t idx = i * 17;
+                while (idx >= this->processDataNum) {
+                    idx -= this->processDataNum;
+                }
                 outLocal.SetValue(idx, this->value);
             }
             outQueueOUT.EnQue<T>(outLocal);
@@ -159,10 +163,18 @@
         // Degradation: Replace vectorized Duplicate with scalar loops
         // Process low and high parts separately in scalar fashion
         // Further degradation: Non-sequential memory access pattern causing cache inefficiency
+        // Additional degradation: Inefficient modulo calculation using subtraction loop
         for (uint32_t i = 0; i < this->processDataNum; i += 2) {
-            // Access elements with stride pattern that breaks cache locality
-            uint32_t idx1 = (i * 17) % this->processDataNum;
-            uint32_t idx2 = ((i + 1) * 17) % this->processDataNum;
+            // Compute idx1 = (i * 17) % processDataNum using slow subtraction loop
+            uint32_t idx1 = i * 17;
+            while (idx1 >= this->processDataNum) {
+                idx1 -= this->processDataNum;
+            }
+            // Compute idx2 = ((i + 1) * 17)... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`169622.109375`
- fast duration us：`87360.84375`
- improvement ratio：`0.48496782599924554`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 87333 -> 169593 (+94.19%)
- latency.task_duration_us: 87360.8 -> 169622 (+94.16%)
- aiv_utilization.mte2_ratio_mean: 2.1e-05 -> 1e-05 (-52.38%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 2.6e-05 -> 1.3e-05 (-50.00%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.37 -> 0.19 (-48.65%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 87332.973,
      "after": 169593.395,
      "delta": 82260.42199999999,
      "rel": 0.9419171153145101
    }
  },
  {
    "latency.task_duration_us": {
      "before": 87360.844,
      "after": 169622.109,
      "delta": 82261.265,
      "rel": 0.9416262622188036
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 2.1e-05,
      "after": 1e-05,
      "delta": -1.0999999999999998e-05,
      "rel": -0.5238095238095237
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 2.6e-05,
      "after": 1.3e-05,
      "delta": -1.3e-05,
      "rel": -0.5
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 0.37,
      "after": 0.19,
      "delta": -0.18,
      "rel": -0.48648648648648646
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

- Replace subtraction loops with hardware modulo in KernelFill::Compute and KernelFill1_INT64::Compute functions
- Change 'while (idx >= this->processDataNum) { idx -= this->processDataNum; }' to 'idx = idx % this->processDataNum;'
- Apply to all index calculations in inner loops of Ascend C kernels
- Verify performance improvement by monitoring aiv_vector0_time_us_mean and scalar_ratio_mean metrics

## 检索线索

- modulo
- subtraction loop
- arithmetic intensity
- scalar ratio
- index calculation
- fill
- AscendC
- hardware modulo
- while loop
- compute bound
