---
type: Optimization
title: "heaviside：Tiling优化"
description: "heaviside 算子的Tiling优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/docs/Heaviside.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/tf_plugin/tensorflow_add_custom_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_host/heaviside.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_host/heaviside_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_kernel/heaviside.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# heaviside：Tiling优化

## 来源信息

- experience ID：`c99bbfdd0818df53`
- 源 schema version：`v2`
- 算子：`heaviside`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.parallelism.reduction`
- canonical family：`tiling.parallelism.reduction`
- 源记录时间：`2026-04-20T19:05:54`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase AI core parallelism in tiling calculation
~~~~

### Original summary

~~~~text
For vector operators like Heaviside, increasing the aivNum parameter in the tiling function from 20 to 40 improves performance by 9.6%. This optimization works by doubling AI core parallelism, which reduces scalar overhead per core and improves vector unit utilization from 21.4% to 11.8%. Apply this technique when aiv_utilization.vec_ratio_mean is above 20% and memory bandwidth usage is below 20%.
~~~~

### Original root mechanism

~~~~text
Increasing aivNum from 20 to 40 doubles AI core parallelism, reducing workload per core and improving overall throughput
~~~~

### Original causal chain

~~~~text
Increasing aivNum from 20 to 40 → block_count increases from 20 to 40 → aiv_vec_fops per core reduces by 50% → vector unit utilization decreases from 21.4% to 11.8% → aiv_vec_wait_ratio_mean decreases from 83.2% to 90.5% → task duration reduces from 1385.2μs to 1251.9μs
~~~~

## 适用条件

- Operator uses vector processing (op_type: vector)
- aiv_utilization.vec_ratio_mean > 20%
- memory_ub.gm_to_ub_bw_usage_rate_pct < 20%
- memory_ub.ub_to_gm_bw_usage_rate_pct < 15%
- Operator has scalar_overhead_bound bottleneck type

## 失效边界

- Memory bandwidth usage is already high (> 80%)
- Vector unit utilization is already low (< 10%)
- Operator has memory_bound bottleneck type
- Block_count is already at maximum AI core capacity

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/heaviside.cpp

### 改动符号

- optiling::TilingFunc

### 原始 diff

- 源 diff SHA-256：`5b59eb8ddeb0a4145bdfc3d65257788ec2355aa58a5a827fb014e551eecbd147`

````diff
--- Before/op_host/heaviside.cpp
+++ After/op_host/heaviside.cpp
@@ -6,7 +6,7 @@
     uint32_t size = context->GetInputTensor(0)->GetShapeSize();
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
     (void)ascendcPlatform;
-    uint32_t aivNum = 40;
+    uint32_t aivNum = 20;
     auto dt = context->GetInputTensor(0)->GetDataType();
     int DataTypeSize = 0;
     if (dt == ge::DT_FLOAT) {
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1385.207764`
- fast duration us：`1251.905029`
- improvement ratio：`0.09623302616718511`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_to_gm_kb: 13107.5 -> 26214.5 (+100.00%)
- memory_ub.gm_to_ub_kb: 26215.1 -> 52429.1 (+100.00%)
- arithmetic.aiv_vec_fops: 2.6937e+07 -> 5.38723e+07 (+99.99%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 5.68 -> 10.37 (+82.57%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 10.26 -> 18.72 (+82.46%)

### Affected metric groups

- memory_ub
- arithmetic
- pipeline_stalls
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 13107.5,
      "after": 26214.5,
      "delta": 13107.0,
      "rel": 0.9999618539004387
    }
  },
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 26215.1,
      "after": 52429.1,
      "delta": 26214.0,
      "rel": 0.9999580394505457
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 26936960.0,
      "after": 53872320.0,
      "delta": 26935360.0,
      "rel": 0.9999406020575448
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 5.68,
      "after": 10.37,
      "delta": 4.6899999999999995,
      "rel": 0.8257042253521126
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 10.26,
      "after": 18.72,
      "delta": 8.459999999999999,
      "rel": 0.8245614035087718
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
  "metrics_checked": 1,
  "metrics_wrong": 0,
  "performance_metrics_checked": 1,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- 1. In the tiling function (optiling::TilingFunc in op_host/heaviside.cpp), locate the aivNum parameter
- 2. Increase aivNum value to increase parallelism (e.g., from 20 to 40)
- 3. Verify that block_count increases proportionally to aivNum change
- 4. Monitor aiv_utilization.vec_ratio_mean decreases to 11-12% range
- 5. Check that memory bandwidth usage increases proportionally but stays below 20%
- 6. Validate that aiv_vec_wait_ratio_mean decreases and task_duration_us improves

## 检索线索

- tiling
- parallelism
- aivNum
- vector
- heaviside
- scalar_overhead_bound
- block_count
- aiv_utilization
- memory_bandwidth
