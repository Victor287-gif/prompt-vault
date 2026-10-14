---
type: Optimization
title: "rsqrt：冗余消除优化"
description: "rsqrt 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.scalar"]
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

- experience ID：`32c3040c0a0a9753`
- 源 schema version：`v2`
- 算子：`rsqrt`
- source iteration：`round_015_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.copy`
- canonical family：`remove.redundant.copy`
- 源记录时间：`2026-04-21T00:22:50`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove redundant DataCopy and duplicate Cast in reciprocal square root computation
~~~~

### Original summary

~~~~text
In Ascend C reciprocal square root implementations, avoid redundant DataCopy operations between local tensors when the source value is already available. The DataCopy(p1, p2) followed by Cast(p1, xLocal) in the non-float32 path was unnecessary because p1 already contained the cast value from xLocal before the DataCopy, and the DataCopy was immediately overwritten by the second Cast with the same value. Removing these redundant operations reduces vector unit wait time by 10% and decreases memory bandwidth usage.
~~~~

### Original root mechanism

~~~~text
Redundant DataCopy and duplicate Cast operations create unnecessary memory traffic and scalar overhead without affecting the final computation result.
~~~~

### Original causal chain

~~~~text
Removing redundant DataCopy and duplicate Cast -> reduces scalar overhead and memory traffic -> decreases vector wait ratio from 0.529 to 0.429 -> improves vector unit utilization from 0.497 to 0.594 -> reduces overall latency by 5.8%
~~~~

## 适用条件

- Implementing reciprocal square root (rsqrt) or similar mathematical operations in Ascend C
- Using LocalTensor<float> for intermediate calculations with DataCopy operations
- When Cast operations are followed by redundant DataCopy between local tensors
- When scalar overhead is identified as a bottleneck (scalar_ratio_mean > 0.09)

## 失效边界

- When DataCopy is necessary for actual data movement between different memory regions
- When Cast operations produce different values than the source tensor
- When the redundant operations are required for correctness or numerical stability

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
- slow duration us：`1457.88916`
- fast duration us：`1373.347412`
- improvement ratio：`0.05798914644512473`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.013566 -> 0.032281 (+137.96%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.078527 -> 0.142088 (+80.94%)
- arithmetic.aiv_vec_fops: 4.6024e+07 -> 5.85839e+07 (+27.29%)
- memory_ub.ub_write_bw_vector_gbps: 148.29 -> 182.35 (+22.97%)
- memory_ub.ub_read_bw_vector_gbps: 175.86 -> 215.75 (+22.68%)

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
  "memory_ub.ub_write_bw_vector_gbps",
  "memory_ub.ub_read_bw_vector_gbps"
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

- In KernelRsqrt::Compute function in rsqrt.cpp, examine DataCopy operations between local tensors
- Check if Cast operations immediately follow DataCopy with the same source value
- Verify that the DataCopy source tensor contains the same value as the subsequent Cast source
- Remove redundant DataCopy(p1, p2) when p1 already contains the value from xLocal
- Remove duplicate Cast(p1, xLocal) when p1 already has the correct value
- Profile with aiv_utilization.scalar_ratio_mean metric to verify scalar overhead reduction

## 检索线索

- rsqrt
- reciprocal square root
- DataCopy
- redundant copy
- scalar overhead
- vector wait ratio
- aiv_vec_bank_cflt_ratio_mean
- LocalTensor
- Ascend C
- memory bandwidth
- pipeline stalls
