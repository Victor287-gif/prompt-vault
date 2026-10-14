---
type: Optimization
title: "cast：向量化计算优化"
description: "cast 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/docs/Cast.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/framework/tf_plugin/tensorflow_cast_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/op_host/cast.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/op_host/cast_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/op_kernel/cast.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/atk/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/atk/generate_cast.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/atk/op_cast.yaml
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/atk/torch_cast.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/st/Cast_case_f162f32.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/st/test_cast.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cast/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# cast：向量化计算优化

## 来源信息

- experience ID：`5ce5726384d2b1af`
- 源 schema version：`v2`
- 算子：`cast`
- source iteration：`round_015_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vectorization.break.scalar.loop`
- canonical family：`vectorization.break.scalar.loop`
- 源记录时间：`2026-04-20T15:34:48`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Maintain vectorized Cast operations instead of scalar loops
~~~~

### Original summary

~~~~text
Using a single vectorized Cast operation with the full processDataNum parameter achieves 2.4x faster performance compared to processing data in smaller CHUNK_SIZE increments. This optimization eliminates loop overhead and enables efficient vector unit utilization, reducing aiv_vec_wait_ratio_mean from 99.3% to 10.9% and improving vector utilization from 3.6% to 74.9%. Apply this pattern when converting between data types using Ascend C's Cast API.
~~~~

### Original root mechanism

~~~~text
Vectorized Cast operations process multiple elements simultaneously, avoiding loop overhead and scalar execution that cause pipeline stalls.
~~~~

### Original causal chain

~~~~text
Replacing scalar loop with vectorized Cast -> Reduces aiv_vec_wait_ratio_mean from 99.3% to 10.9% and increases vec_ratio_mean from 3.6% to 74.9% -> Decreases task_duration_us from 2734us to 1602us (41.4% improvement)
~~~~

## 适用条件

- Using Ascend C Cast operations for data type conversions
- Processing contiguous memory regions with LocalTensor<TYPE_X> and LocalTensor<TYPE_Y>
- Working with vector data types like half, float, int32_t, int16_t, int64_t, bfloat16_t
- When RoundMode parameters like CAST_TRUNC, CAST_ROUND, CAST_CEIL, CAST_RINT, or CAST_NONE are needed

## 失效边界

- Breaking vector operations into loops with small CHUNK_SIZE (e.g., 32)
- Using scalar operations for vectorizable data type conversions
- Adding unnecessary loop overhead to Cast operations

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- cast/op_kernel/cast.cpp

### 改动符号

- KernelCast0TBuf::Compute

### 代码签名

- AllocTensor
- DeQue
- EnQue
- FreeTensor
- TBuf
- TQue
- bool
- bool_control
- compile_time_specialization
- constexpr
- contiguous_access
- queue_api
- tbuf
- template

### 原始 diff

- 源 diff SHA-256：`2af816bc1cfa689047d2c3e709fea330d4a115e91cae84625312d30db5d23d54`

````diff
--- Before/op_kernel/cast.cpp
+++ After/op_kernel/cast.cpp
@@ -118,44 +118,57 @@
     {
         LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
         LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
-        if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, int32_t>) ||
-                      (std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, int16_t>) ||
-                      (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int32_t>) ||
-                      (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int16_t>) ||
-                      (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int64_t>) ||
-                      (std::is_same_v<DTYPE_X, bfloat16_t> && std::is_same_v<DTYPE_Y, int32_t>))
-        {
-            Cast(yLocal, xLocal, RoundMode::CAST_TRUNC, this->processDataNum);
-        }
-        else if constexpr ((std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, float>))
-        {
-            Cast(yLocal, xLocal, RoundMode::CAST_ROUND, this->processDataNum);
-        }
-        else if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, bool>))
-        {
-            Abs(xLocal, xLocal, this->processDataNum);
-            Mins(xLocal, xLocal, HALF_ONE, this->processDataNum);
-            Cast(yLocal, xLocal, RoundMode::CAST_CEIL, this->processDataNum);
-        }
-        else if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, float>) ||
-                           (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, half>) ||
-                           (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, int16_t>) ||
-                           (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, int64_t>) ||
-                           (std::is_same_v<DTYPE_X, int32_t> && std::is_same_v<DTYPE_Y, float>) ||
-           ... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`2734.154541`
- fast duration us：`1601.531982`
- improvement ratio：`0.4142496490288915`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.000729 -> 0.599546 (+82142.25%)
- aiv_utilization.vec_ratio_mean: 0.035616 -> 0.749432 (+2004.20%)
- aiv_utilization.scalar_ratio_mean: 0.056303 -> 0.563438 (+900.72%)
- arithmetic.aiv_vec_fops: 5.65824e+06 -> 3.35984e+07 (+493.80%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.034887 -> 0.149886 (+329.63%)

### Affected metric groups

- arithmetic
- aiv_utilization
- pipeline_stalls
- critical_path

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_misc_ratio_mean",
  "aiv_utilization.vec_ratio_mean",
  "aiv_utilization.scalar_ratio_mean",
  "arithmetic.aiv_vec_fops",
  "arithmetic.aiv_vec_fp32_ratio_mean",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "aiv_utilization.mte2_ratio_mean"
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

- Use Cast(yLocal, xLocal, RoundMode, processDataNum) with full vector length instead of loop
- Avoid breaking Cast operations into smaller chunks with CHUNK_SIZE loops
- Ensure Cast operations process the entire processDataNum at once for maximum vectorization
- Check for patterns where Cast is wrapped in for-loops with small iteration counts

## 检索线索

- Cast
- vectorization
- scalar_loop
- Ascend C
- LocalTensor
- RoundMode
- CHUNK_SIZE
- vector_ratio
- pipeline_stall
- type_conversion
