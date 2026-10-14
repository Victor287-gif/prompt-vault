---
type: Optimization
title: "rsqrt：数据搬运优化"
description: "rsqrt 算子的数据搬运优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
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
# rsqrt：数据搬运优化

## 来源信息

- experience ID：`d78e1f154648605f`
- 源 schema version：`v2`
- 算子：`rsqrt`
- source iteration：`round_014_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`data.movement.redundant.copy`
- canonical family：`data.movement.redundant.copy`
- 源记录时间：`2026-04-21T00:17:41`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant DataCopy between temporary buffers
~~~~

### Original summary

~~~~text
Avoid redundant DataCopy operations between temporary buffers in Ascend C kernels, especially in non-float32 computation paths. The DataCopy between p2 and p1 buffers before sqrt computation adds unnecessary memory traffic and increases pipeline stalls. Apply this optimization when implementing reciprocal square root operations using temporary buffers for intermediate calculations.
~~~~

### Original root mechanism

~~~~text
Removing unnecessary DataCopy operations between temporary buffers in the non-float32 computation path reduces memory bandwidth usage and pipeline stalls
~~~~

### Original causal chain

~~~~text
Removing DataCopy(p1, p2) → reduces UB read/write bandwidth by 22-23% → decreases aiv_vec_bank_cflt_ratio_mean by 138% → lowers pipeline stalls → reduces overall latency by 5.7%
~~~~

## 适用条件

- Implementing reciprocal square root (rsqrt) operations in Ascend C
- Using temporary LocalTensor buffers for intermediate calculations
- Working with non-float32 data types requiring type casting
- Kernel has UB bandwidth usage above 150 Gbps
- aiv_vec_bank_cflt_ratio_mean > 0.01 indicating memory conflict stalls

## 失效边界

- DataCopy is necessary for data dependency resolution
- Temporary buffers need explicit initialization from other buffers
- The copy operation serves as a memory barrier for synchronization

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- rsqrt/op_kernel/rsqrt.cpp

### 改动符号

- KernelRsqrt::Compute

### 代码签名

- DataCopy
- datacopy

### 原始 diff

- 源 diff SHA-256：`8a6f2bccc40862a7531e0cae77b82c6da7018aa0b71eea05d723960bdbfc8dcb`

````diff
--- Before/op_kernel/rsqrt.cpp
+++ After/op_kernel/rsqrt.cpp
@@ -78,6 +78,10 @@
             LocalTensor<float> p2 = tmp2.Get<float>();
             Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
             Duplicate<float>(p2, 1.0f, this->processDataNum);
+            // Add redundant data movement: copy p2 to p1 before sqrt
+            DataCopy(p1, p2, this->processDataNum);
+            // Restore original p1 value from xLocal
+            Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
             Sqrt(p1, p1, this->processDataNum);
             Div(p1, p2, p1, this->processDataNum);
             Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1457.88916`
- fast duration us：`1374.527466`
- improvement ratio：`0.05717972002754993`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.013557 -> 0.032281 (+138.11%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.078194 -> 0.142088 (+81.71%)
- arithmetic.aiv_vec_fops: 4.6024e+07 -> 5.85839e+07 (+27.29%)
- memory_ub.ub_write_bw_vector_gbps: 148.2 -> 182.35 (+23.04%)
- memory_ub.ub_read_bw_vector_gbps: 175.75 -> 215.75 (+22.76%)

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
  "magnitude_deviation": 0.212951,
  "overall_deviation": 0.08518,
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

- In rsqrt.cpp KernelRsqrt::Compute function, examine DataCopy usage between temporary buffers
- Remove DataCopy(p1, p2) calls that duplicate already available data
- Verify Cast operations provide correct data to temporary buffers before arithmetic operations
- Check that Sqrt and Div operations receive correct inputs after removing DataCopy
- Profile UB bandwidth metrics to confirm reduction in memory traffic

## 检索线索

- DataCopy
- redundant
- temporary buffers
- rsqrt
- reciprocal square root
- Ascend C
- memory bandwidth
- pipeline stalls
- vector arithmetic
- type casting
