---
type: Optimization
title: "cast：循环计算优化"
description: "cast 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
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
# cast：循环计算优化

## 来源信息

- experience ID：`9f891c2b8102ffb7`
- 源 schema version：`v2`
- 算子：`cast`
- source iteration：`round_018_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.vectorization.restore`
- canonical family：`loop.vectorization.restore`
- 源记录时间：`2026-04-20T15:56:09`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Avoid chunked processing loops in vectorized Ascend C kernels
~~~~

### Original summary

~~~~text
In Ascend C cast operations, processing data in small chunks with while loops instead of using vectorized calls on the full dataset causes severe pipeline stalls. The chunked approach reduces vector unit utilization from 74.9% to 3.6% and increases vector wait ratio from 10.9% to 99.2%. Always use vectorized Cast() operations on the full processDataNum instead of breaking it into chunks.
~~~~

### Original root mechanism

~~~~text
Chunked processing loops introduce pipeline stalls by breaking vectorization and preventing Ascend C vector units from operating on full data chunks efficiently
~~~~

### Original causal chain

~~~~text
Replacing chunked while-loop processing with single vectorized Cast() calls → reduces aiv_vec_wait_ratio_mean from 99.2% to 10.9% and aiv_mte2_wait_ratio_mean from 99.7% to 0.003% → decreases latency from 2734μs to 1602μs (41.4% improvement)
~~~~

## 适用条件

- When implementing Ascend C kernels using vector operations like Cast()
- When processing contiguous data with fixed-size operations
- When using Ascend C vector APIs (Cast, Abs, Mins) with LocalTensor access patterns
- When converting between data types (half→int32, half→int16, float→int32, float→int16, float→int64, bfloat16_t→int32, int64_t→float, half→bool)

## 失效边界

- When refactoring vectorized Ascend C kernels to use chunked processing loops
- When breaking continuous data processing into smaller chunks with while/for loops
- When using CHUNK_SIZE constants to limit vector operation sizes in Ascend C
- When processing data with offset-based chunking instead of full vector operations

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

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

- 源 diff SHA-256：`c1ed0342888edf6d12d99aa41a54cd60fa48d0519b76e7da14cb2e21229c52ee`

````diff
--- Before/op_kernel/cast.cpp
+++ After/op_kernel/cast.cpp
@@ -118,57 +118,44 @@
     {
         LocalTensor<TYPE_X> xLocal = this->inQueueX.template DeQue<TYPE_X>();
         LocalTensor<TYPE_Y> yLocal = this->outQueueY.template AllocTensor<TYPE_Y>();
-        // Break vectorization by processing in smaller chunks
-        constexpr uint32_t CHUNK_SIZE = 32; // Smaller chunk size to break vectorization
-        uint32_t remaining = this->processDataNum;
-        uint32_t offset = 0;
-
-        while (remaining > 0) {
-            uint32_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
-
-            if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, int32_t>) ||
-                          (std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, int16_t>) ||
-                          (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int32_t>) ||
-                          (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int16_t>) ||
-                          (std::is_same_v<DTYPE_X, float> && std::is_same_v<DTYPE_Y, int64_t>) ||
-                          (std::is_same_v<DTYPE_X, bfloat16_t> && std::is_same_v<DTYPE_Y, int32_t>))
-            {
-                Cast(yLocal[offset], xLocal[offset], RoundMode::CAST_TRUNC, chunk);
-            }
-            else if constexpr ((std::is_same_v<DTYPE_X, int64_t> && std::is_same_v<DTYPE_Y, float>))
-            {
-                Cast(yLocal[offset], xLocal[offset], RoundMode::CAST_ROUND, chunk);
-            }
-            else if constexpr ((std::is_same_v<DTYPE_X, half> && std::is_same_v<DTYPE_Y, bool>))
-            {
-                Abs(xLocal[offset], xLocal[offset], chunk);
-                Mins(xLocal[offset], xLocal[offset], HALF_ONE, chunk);
-                Cast(yLocal[offset], xLocal[offset], RoundMode::CAST_CEIL, chunk);
-            }
-            else if constexpr ((std::is_same_v<DTYP... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`2734.154541`
- fast duration us：`1602.432007`
- improvement ratio：`0.41392047048886976`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.000728 -> 0.599546 (+82255.22%)
- aiv_utilization.vec_ratio_mean: 0.035581 -> 0.749432 (+2006.27%)
- aiv_utilization.scalar_ratio_mean: 0.055863 -> 0.563438 (+908.61%)
- arithmetic.aiv_vec_fops: 5.65824e+06 -> 3.35984e+07 (+493.80%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.034852 -> 0.149886 (+330.06%)

### Affected metric groups

- arithmetic
- aiv_utilization
- pipeline_stalls
- critical_path

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.000728,
      "after": 0.599546,
      "delta": 0.5988180000000001,
      "rel": 822.5521978021978
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 0.035581,
      "after": 0.749432,
      "delta": 0.713851,
      "rel": 20.062702003878474
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.055863,
      "after": 0.563438,
      "delta": 0.507575,
      "rel": 9.08606770134078
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 5658240.0,
      "after": 33598400.0,
      "delta": 27940160.0,
      "rel": 4.937959506843117
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.034852,
      "after": 0.149886,
      "delta": 0.115034,
      "rel": 3.300642717778033
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

- Use vectorized Cast() operations on the full processDataNum instead of chunking: Cast(yLocal[0], xLocal[0], RoundMode::CAST_TRUNC, this->processDataNum)
- Avoid while loops with CHUNK_SIZE constants that break vectorization in Ascend C kernels
- Maintain contiguous memory access patterns by processing entire datasets in single vector operations
- In KernelCast0TBuf::Compute(), call vector APIs with full data size rather than chunked offsets
- For type conversions, use the appropriate RoundMode (CAST_TRUNC for float→int, CAST_ROUND for int64_t→float, CAST_CEIL for half→bool)

## 检索线索

- Ascend C
- vectorization
- chunked processing
- pipeline stalls
- Cast operation
- vector wait ratio
- aiv utilization
- math cast
- loop refactoring
- performance degradation
- KernelCast0TBuf
- processDataNum
- LocalTensor
- RoundMode
