---
type: Optimization
title: "gcd：并行与 Tiling优化"
description: "gcd 算子的并行与 Tiling优化经验，关注并行度。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/Gcd.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/tensorflow_sqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd_common.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gcd：并行与 Tiling优化

## 来源信息

- experience ID：`68d8948e06960b2e`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_010_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`tiling.core_allocation`
- canonical family：`performance.degradation.removal`
- 源记录时间：`2026-04-20T20:33:07`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Maintain full core count in tiling function for vector operators
~~~~

### Original summary

~~~~text
When implementing tiling functions for vector operators, ensure the block dimension is set to the full available core count from ascendcPlatform.GetCoreNumAiv(). Avoid artificial core count reductions that limit parallelism. This optimization is critical for compute-bound vector operators where performance scales linearly with available cores.
~~~~

### Original root mechanism

~~~~text
Artificially reducing available cores in tiling functions halves block-level parallelism, directly doubling vector execution time for compute-bound vector operators
~~~~

### Original causal chain

~~~~text
Removing artificial core count reduction -> block_count increases from 24 to 48 -> aiv_vec_fops per block halves (427M to 213M) -> aiv_vector0_time_us_mean halves (10.9ms to 5.5ms) -> task_duration_us halves (11ms to 5.5ms)
~~~~

## 适用条件

- Implementing tiling functions for vector operators (op_type: vector)
- Using ascendcPlatform.GetCoreNumAiv() to determine block dimensions
- Working with compute-bound operators where vector unit utilization is high (>89%)
- When block_count equals active_block_count indicating full parallelism utilization

## 失效边界

- Artificially reducing core count with division operations (n_core = n_core / 2)
- Setting block dimensions below available hardware cores
- When operator is memory-bound rather than compute-bound
- When specific core count constraints are required for correctness

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- gcd/op_host/gcd.cpp

### 改动符号

- TilingFunc

### 代码签名

- GetCoreNum
- SetBlockDim
- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`ab97a830b8ea4cb515ef23672a6a752b4f8c1d730ea04505464d84dab4c58630`

````diff
--- Before/op_host/gcd.cpp
+++ After/op_host/gcd.cpp
@@ -49,9 +49,6 @@
     tiling.set_broadcast_mask(broadcast_mask);

     uint32_t n_core = ascendcPlatform.GetCoreNumAiv();
-    // Artificially reduce available cores to decrease parallelism and degrade performance
-    n_core = n_core / 2;
-    if (n_core < 1) n_core = 1;
     context->SetBlockDim(n_core);
     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
````

## 观测记录

- bottleneck type：`parallelism`
- slow duration us：`10999.399414`
- fast duration us：`5503.290039`
- improvement ratio：`0.499673588360158`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 344 -> 688 (+100.00%)
- arithmetic.aiv_vec_fops: 2.13504e+08 -> 4.27005e+08 (+100.00%)
- memory_ub.gm_to_ub_kb: 688.1 -> 1376.1 (+99.99%)
- critical_path.aiv_vector0_time_us_mean: 5458.11 -> 10911.3 (+99.91%)
- latency.task_duration_us: 5503.29 -> 10999.4 (+99.87%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "memory_ub.ub_to_gm_kb",
  "arithmetic.aiv_vec_fops",
  "memory_ub.gm_to_ub_kb",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us"
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

- In tiling functions, call ascendcPlatform.GetCoreNumAiv() to get available cores
- Set block dimension directly: context->SetBlockDim(n_core)
- Never apply artificial reductions like n_core = n_core / 2
- Verify block_count equals active_block_count in performance metrics
- Check that aiv_vec_fops per block remains constant when scaling cores

## 检索线索

- core_allocation
- block_dim
- parallelism
- tiling
- GetCoreNumAiv
- SetBlockDim
- vector_operator
- compute_bound
- core_count_reduction
- performance_degradation
