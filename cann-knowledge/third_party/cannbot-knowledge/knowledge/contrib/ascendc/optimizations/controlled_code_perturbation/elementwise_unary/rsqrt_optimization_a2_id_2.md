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

- experience ID：`c8257a5f2e87304e`
- 源 schema version：`v2`
- 算子：`rsqrt`
- source iteration：`round_016_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.data.movement`
- canonical family：`remove.redundant.data.movement`
- 源记录时间：`2026-04-21T00:32:59`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant data movement in reciprocal sqrt computation
~~~~

### Original summary

~~~~text
In Ascend C reciprocal sqrt implementations, avoid redundant data movement operations that copy data between local tensors only to immediately restore the original values. These operations increase vector floating-point operations (FLOPS) by ~27% and UB vector bandwidth by ~22-23% without providing any functional benefit. Apply this optimization when you identify DataCopy operations followed by Cast operations that restore original tensor values.
~~~~

### Original root mechanism

~~~~text
Removing semantically redundant DataCopy and Cast operations reduces unnecessary vector operations and memory bandwidth consumption
~~~~

### Original causal chain

~~~~text
Removing redundant DataCopy(p1, p2) and Cast(p1, xLocal) operations → reduces aiv_vec_fops by 12.6M operations (-27.3%) → decreases UB vector read/write bandwidth by 22-23% → reduces vector wait ratio by 19% → improves task duration by 6.2%
~~~~

## 适用条件

- Implementing reciprocal sqrt (rsqrt) or similar math operations in Ascend C
- Using LocalTensor<float> with DataCopy and Cast operations in sequence
- Observing high aiv_vec_fops and UB vector bandwidth usage in profile data
- When Cast operations restore original tensor values after DataCopy operations

## 失效边界

- Data movement is necessary for algorithmic correctness
- Copy operations serve a functional purpose beyond temporary storage
- When the data movement pattern is required for data type conversion or precision requirements

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
- fast duration us：`1367.147339`
- improvement ratio：`0.062241920366566034`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.013602 -> 0.032281 (+137.33%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.078674 -> 0.142088 (+80.60%)
- arithmetic.aiv_vec_fops: 4.6024e+07 -> 5.85839e+07 (+27.29%)
- memory_ub.ub_write_bw_vector_gbps: 148.69 -> 182.35 (+22.64%)
- memory_ub.ub_read_bw_vector_gbps: 176.33 -> 215.75 (+22.36%)

### Affected metric groups

- pipeline_stalls
- arithmetic
- memory_ub
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.013602,
      "after": 0.032281,
      "delta": 0.018678999999999998,
      "rel": 1.3732539332451108
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.078674,
      "after": 0.142088,
      "delta": 0.063414,
      "rel": 0.8060350306327376
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 46024032.0,
      "after": 58583936.0,
      "delta": 12559904.0,
      "rel": 0.2728988194689244
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 148.69,
      "after": 182.35,
      "delta": 33.66,
      "rel": 0.2263770260273051
    }
  },
  {
    "memory_ub.ub_read_bw_vector_gbps": {
      "before": 176.33,
      "after": 215.75,
      "delta": 39.41999999999999,
      "rel": 0.22355810128735884
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

- Examine the Compute function in rsqrt/op_kernel/rsqrt.cpp for patterns of DataCopy followed by Cast
- Check if DataCopy operations copy between local tensors (p1, p2) without changing the final result
- Verify that Cast operations restore original values from source tensors (xLocal)
- Remove redundant DataCopy(p1, p2) and subsequent Cast(p1, xLocal) operations when they cancel each other out
- Profile the kernel to confirm reduction in aiv_vec_fops and UB vector bandwidth metrics

## 检索线索

- rsqrt
- reciprocal sqrt
- DataCopy
- Cast
- redundant data movement
- vector operations
- UB bandwidth
- aiv_vec_fops
- Ascend C
- math operator
- performance optimization
- remove redundant operations
