---
type: Optimization
title: "rsqrt：冗余消除优化"
description: "rsqrt 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/docs/Rsqrt.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/tf_plugin/tensorflow_rsqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_host/rsqrt.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_host/rsqrt_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_kernel/rsqrt.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/Rsqrt_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/test_rsqrt.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# rsqrt：冗余消除优化

## 来源信息

- experience ID：`c4255bc27fe8df04`
- 源 schema version：`v2`
- 算子：`rsqrt`
- source iteration：`round_019_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.copy`
- canonical family：`remove.redundant.copy`
- 源记录时间：`2026-04-21T01:21:54`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove redundant DataCopy and Cast operations in non-float32 path
~~~~

### Original summary

~~~~text
Eliminating redundant DataCopy and Cast operations in the non-float32 path of rsqrt kernel reduces vector floating point operations by 27.3% and improves performance by 8.0%. This optimization applies when temporary buffers are used for intermediate calculations and unnecessary data movement can be identified and removed.
~~~~

### Original root mechanism

~~~~text
Redundant DataCopy(p1, p2) and Cast(p1, xLocal) operations were performing unnecessary data movement and type conversions that increased vector operation count and memory bandwidth usage without functional benefit.
~~~~

### Original causal chain

~~~~text
Removing redundant DataCopy and Cast operations → reduces aiv_vec_fops from 58.8M to 46.2M (-21.5%) and vector bandwidth usage → decreases task duration from 1729.7μs to 1590.7μs (-8.0%)
~~~~

## 适用条件

- When using LocalTensor<float> for intermediate calculations in non-float32 data type paths
- When DataCopy operations follow Cast operations with the same source and destination buffers
- When temporary buffers are initialized with constants (e.g., Duplicate) then immediately overwritten
- When vector floating point operations (aiv_vec_fops) are higher than expected for the algorithm

## 失效边界

- When DataCopy operations are required for data dependency resolution
- When Cast operations are needed for actual type conversion between different data types
- When intermediate buffer reuse patterns are complex and require explicit data movement

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- rsqrt/op_kernel/rsqrt.cpp

### 改动符号

- KernelRsqrt::Compute

### 代码签名

- DataCopy
- datacopy
- ub

### 原始 diff

- 源 diff SHA-256：`0d06d72a4a17713a95452c4f372bfad0ce12ba5cb4c9884f8f1b47e8481e8c62`

````diff
--- Before/op_kernel/rsqrt.cpp
+++ After/op_kernel/rsqrt.cpp
@@ -78,10 +78,6 @@
             LocalTensor<float> p2 = tmp2.Get<float>();
             Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
             Duplicate<float>(p2, 1.0f, this->processDataNum);
-            // Add redundant data movement: copy p2 to p1 before sqrt
-            DataCopy(p1, p2, this->processDataNum);
-            // Restore original p1 value from xLocal
-            Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
             Sqrt(p1, p1, this->processDataNum);
             Div(p1, p2, p1, this->processDataNum);
             Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1729.69458`
- fast duration us：`1590.651855`
- improvement ratio：`0.08038570890359154`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.035396 -> 0.069971 (+97.68%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.0776 -> 0.139887 (+80.27%)
- arithmetic.aiv_vec_fops: 4.62144e+07 -> 5.88342e+07 (+27.31%)
- memory_ub.ub_read_bw_vector_gbps: 160.52 -> 197.92 (+23.30%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.126754 -> 0.154021 (+21.51%)

### Affected metric groups

- pipeline_stalls
- arithmetic
- memory_ub
- aiv_utilization

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "arithmetic.aiv_vec_fops",
  "memory_ub.ub_read_bw_vector_gbps",
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean",
  "memory_ub.ub_write_bw_vector_gbps",
  "aiv_utilization.vec_ratio_mean",
  "aiv_utilization.mte2_ratio_mean",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
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

- Inspect rsqrt.cpp:78-88 in the KernelRsqrt::Compute function for patterns of redundant DataCopy and Cast operations
- Look for sequences where Cast(p1, xLocal) is followed by DataCopy(p1, p2) then another Cast(p1, xLocal)
- Verify that p2 contains constant values (e.g., 1.0f from Duplicate operation) that don't need to be copied to p1
- Remove the redundant DataCopy and subsequent Cast operations when p1 already contains the required value from the first Cast
- Check that removing these operations doesn't break data dependencies in the computation graph

## 检索线索

- DataCopy
- Cast
- redundant
- rsqrt
- vector
- floating-point
- temporary-buffer
- non-float32
- scalar-overhead
- memory-bandwidth
- vector-operations
