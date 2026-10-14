---
type: Optimization
title: "rsqrt：冗余消除优化"
description: "rsqrt 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
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

- experience ID：`de5d752327628fbb`
- 源 schema version：`v2`
- 算子：`rsqrt`
- source iteration：`round_018_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.copy`
- canonical family：`remove.redundant.copy`
- 源记录时间：`2026-04-21T00:51:43`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant DataCopy and Cast operations in non-float32 paths
~~~~

### Original summary

~~~~text
In Ascend C kernels, avoid redundant DataCopy and Cast operations that copy data between UB buffers only to immediately overwrite them. These operations introduce scalar overhead that reduces vector unit utilization and increases pipeline stalls. Apply this optimization when you see DataCopy followed by Cast operations on the same buffer in non-float32 computation paths.
~~~~

### Original root mechanism

~~~~text
Redundant DataCopy and Cast operations introduce unnecessary scalar overhead that reduces vector unit utilization and increases pipeline stalls
~~~~

### Original causal chain

~~~~text
Removing redundant DataCopy(p1,p2) and Cast(p1,xLocal) operations → reduces scalar operations (aiv_vec_fops decreased by 12.6M) → improves vector unit utilization (vec_ratio_mean increased from 47.9% to 57.6%) → reduces pipeline stalls (aiv_vec_wait_ratio_mean decreased from 55.3% to 45.3%) → improves overall latency by 8.2%
~~~~

## 适用条件

- Kernel uses non-float32 data types requiring Cast operations
- DataCopy operations appear between Cast operations on the same buffer
- Profile shows high scalar_ratio_mean and low vec_ratio_mean
- Code contains patterns like DataCopy(p1,p2) followed by Cast(p1,xLocal) in rsqrt.cpp

## 失效边界

- DataCopy is necessary for actual data movement between different buffers
- Cast operations are required for type conversion between different computation stages
- The copied data is actually used before being overwritten

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
- fast duration us：`1587.651733`
- improvement ratio：`0.08212018968111709`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.035423 -> 0.069971 (+97.53%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.077585 -> 0.139887 (+80.30%)
- arithmetic.aiv_vec_fops: 4.62144e+07 -> 5.88342e+07 (+27.31%)
- memory_ub.ub_read_bw_vector_gbps: 160.65 -> 197.92 (+23.20%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.126852 -> 0.154021 (+21.42%)

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
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean"
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

- Search for DataCopy calls followed by Cast operations on the same LocalTensor in rsqrt.cpp
- Verify the DataCopy destination is immediately overwritten by Cast
- Remove the redundant DataCopy operation
- Ensure the Cast operation uses the original source tensor directly
- Test that the kernel still produces correct results after removal

## 检索线索

- DataCopy
- Cast
- redundant
- scalar_overhead
- rsqrt
- non-float32
- vector_utilization
- pipeline_stalls
- ub_buffer
