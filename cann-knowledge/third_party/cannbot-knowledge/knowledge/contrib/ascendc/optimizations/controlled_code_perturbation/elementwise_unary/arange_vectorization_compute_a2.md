---
type: Optimization
title: "arange：向量化计算优化"
description: "arange 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/docs/Arange.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/framework/tf_plugin/tensorflow_arange_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_host/arange.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_host/arange_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/op_kernel/arange.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/Arange_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/st/test_arange.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/arange/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# arange：向量化计算优化

## 来源信息

- experience ID：`af117b1fc8760181`
- 源 schema version：`v2`
- 算子：`arange`
- source iteration：`round_003_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`compute.vectorization.break.scalar.loop`
- canonical family：`compute.vectorization.break.scalar.loop`
- 源记录时间：`2026-04-14T21:24:02`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use vectorized AscendC::Add operations instead of scalar loops in Compute function
~~~~

### Original summary

~~~~text
Applying vectorized AscendC::Add operations in the Compute function improves performance by 426x compared to scalar loops. The vectorized approach processes multiple elements simultaneously using SIMD instructions, reducing instruction count and eliminating pipeline stalls. Use this optimization when performing element-wise arithmetic operations on contiguous data in AscendC kernels.
~~~~

### Original root mechanism

~~~~text
Vectorized AscendC::Add operations leverage SIMD parallelism and reduce instruction count, avoiding pipeline stalls caused by scalar loops with GetValue/SetValue per element.
~~~~

### Original causal chain

~~~~text
Replacing scalar loops with vectorized AscendC::Add operations → vector ratio increases from 0.000486 to 0.702551 → vector pipeline stall ratio increases from 8.6e-05 to 0.426273 → task duration decreases from 3.96ms to 9.26μs.
~~~~

## 适用条件

- Performing element-wise arithmetic operations on contiguous data in AscendC kernels
- Working with AscendC::LocalTensor data types in Compute functions
- Processing large arrays where vectorization can be applied
- Using AscendC APIs like Add, Sub, Mul, Div on tensor data

## 失效边界

- Processing non-contiguous or irregular memory access patterns
- When data dependencies prevent vectorization
- Working with small arrays where vectorization overhead exceeds benefits
- When using non-vectorizable operations like conditional branching per element

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- arange/op_kernel/arange.cpp

### 改动符号

- KernelArange_Cast::Compute

### 代码签名

- AllocTensor
- EnQue
- GetValue
- SetValue
- TQue
- contiguous_access
- queue_api
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`7218edad208fb6de8614ace31b44243f881975b24d14c4ea7d85103f1c7592fe`

````diff
--- Before/op_kernel/arange.cpp
+++ After/op_kernel/arange.cpp
@@ -121,8 +121,16 @@
     uint32_t calc_num = ALIGN_UP_32B_ELEMENTS(num, float);

     AscendC::LocalTensor<TYPE_OUT> outLocal = outQueue.AllocTensor<TYPE_OUT>();
-    AscendC::Add(this->calc_out, this->calc_init, this->calc_temp, calc_num);
-    AscendC::Add(this->calc_temp, this->calc_temp, this->calc_step, calc_num);
+    // Replace vectorized Add with scalar loop
+    for (int32_t idx = 0; idx < calc_num; idx++) {
+      float val = this->calc_init.GetValue(idx) + this->calc_temp.GetValue(idx);
+      this->calc_out.SetValue(idx, val);
+    }
+    // Replace vectorized Add with scalar loop
+    for (int32_t idx = 0; idx < calc_num; idx++) {
+      float val = this->calc_temp.GetValue(idx) + this->calc_step.GetValue(idx);
+      this->calc_temp.SetValue(idx, val);
+    }
     AscendC::Cast(outLocal, this->calc_out, AscendC::RoundMode::CAST_ROUND, calc_num);
     outQueue.EnQue<TYPE_OUT>(outLocal);
   }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`3955211.75`
- fast duration us：`9262.924805`
- improvement ratio：`0.9976580457911008`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 9262.16 -> 3.95513e+06 (+42602.06%)
- latency.task_duration_us: 9262.92 -> 3.95521e+06 (+42599.38%)
- memory_ub.ub_write_bw_scalar_gbps: 0.004 -> 0.7535 (+18737.50%)
- memory_ub.ub_read_bw_scalar_gbps: 0 -> 1.507 (+100.00%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.195384 -> 0 (-100.00%)

### Affected metric groups

- critical_path
- latency
- memory_ub
- aiv_utilization

### Changed metrics top

~~~~json
[
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "memory_ub.ub_write_bw_scalar_gbps",
  "memory_ub.ub_read_bw_scalar_gbps",
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean"
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

- Replace scalar loops using GetValue/SetValue with vectorized AscendC operations like AscendC::Add, AscendC::Sub, etc.
- In arange.cpp Compute function, use AscendC::Add(out, init, temp, num) instead of per-element loops
- Ensure tensor data is properly aligned for vector operations using ALIGN_UP_32B_ELEMENTS
- Use AscendC::Cast for type conversion after vector operations rather than per-element casting
- Maintain proper tensor allocation with AscendC::LocalTensor and queue operations

## 检索线索

- vectorization
- AscendC::Add
- SIMD
- scalar loop
- GetValue
- SetValue
- pipeline stall
- compute optimization
- arange
- math operator
