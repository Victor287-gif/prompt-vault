---
type: Optimization
title: "gelu_grad：并行切核优化"
description: "gelu_grad 算子的并行切核优化经验，关注并行度。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/docs/GeluGrad.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/framework/tf_plugin/tensorflow_gelu_grad_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_host/gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_host/gelu_grad_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/op_kernel/gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/GeluGrad_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/st/test_gelu_grad.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu_grad/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gelu_grad：并行切核优化

## 来源信息

- experience ID：`65ceb89478ad4049`
- 源 schema version：`v2`
- 算子：`gelu_grad`
- source iteration：`round_016_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.core_count`
- canonical family：`untitled`
- 源记录时间：`2026-04-20T21:51:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Avoid unnecessary power-of-two rounding for core parallelism
~~~~

### Original summary

~~~~text
Avoid rounding coreNum down to the nearest power-of-two when it reduces overall parallelism. The gelu_grad operator shows that using the maximum available cores (48) instead of rounding down to 32 provides 32% better performance by maintaining higher parallelism and reducing per-core workload. This optimization applies to vector operators with uniform data distribution where all cores can be fully utilized.
~~~~

### Original root mechanism

~~~~text
Forcing coreNum to power-of-two reduces parallelism by 33% (48→32 blocks) and increases per-core workload by 50%, causing longer vector execution time and higher memory transfers.
~~~~

### Original causal chain

~~~~text
Rounding coreNum from 48 to 32 reduces active blocks from 48 to 32 → increases per-core workload by 50% → increases aiv_vec_fops by 50% → increases memory transfers (gm_to_ub_kb and ub_to_gm_kb) by 50% → increases aiv_vector0_time_us by 47.95% → increases task_duration_us by 47.81%.
~~~~

## 适用条件

- Vector operators with uniform data distribution across cores
- Operators where block_count > 1 and inputLengthAlgin32 / BLOCK_SIZE >= coreNum
- When coreNum calculation uses integer division and rounding logic
- When performance profiling shows high aiv_vector0_time_us and increased memory transfers

## 失效边界

- When coreNum rounding is required for hardware alignment constraints
- When memory bandwidth is already saturated and reducing parallelism won't help
- When operator has significant data dependencies between cores
- When using mixed block dimensions (mix_block_dim > 0)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/gelu_grad.cpp
- op_kernel/gelu_grad.cpp

### 代码签名

- EnQue
- FreeTensor
- TQue
- coreNum
- core_launch
- queue_api

### 原始 diff

- 源 diff SHA-256：`5c42d8375122531cbbd6f34c4940eeefb4a1a5336143522454ccc1ed33966822`

````diff
--- Before/op_host/gelu_grad.cpp
+++ After/op_host/gelu_grad.cpp
@@ -44,6 +44,15 @@
         {
             // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
             coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
+            // DEGRADATION: Force coreNum to be power of 2 by rounding down
+            // This may reduce parallelism and cause load imbalance
+            if (coreNum > 1) {
+                uint64_t powerOfTwo = 1;
+                while (powerOfTwo * 2 <= coreNum) {
+                    powerOfTwo *= 2;
+                }
+                coreNum = powerOfTwo;
+            }
         }

         uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;

--- Before/op_kernel/gelu_grad.cpp
+++ After/op_kernel/gelu_grad.cpp
@@ -130,48 +130,48 @@
             Duplicate(p1, COEFF4, this->processDataNum);
             Div(fun, p1, fun, this->processDataNum);

-            Mul(p3, p3, fun, this->processDataNum); // z=tmp*t
-            Mul(p3, p3, tmp, this->processDataNum); // z=z*primtmp
-            Mul(p3, p3, fun, this->processDataNum); // z=z*t
-
-            Add(p3, p3, fun, this->processDataNum); // z=z+t
-            Mul(p3, p3, p2, this->processDataNum);  // z=z*dy
-
-            Cast(zLocal, p3, RoundMode::CAST_RINT, this->processDataNum);
-        }
-         else
-         {
-             // REDUNDANT: add 0.0 to dy (adds arithmetic overhead without changing value)
-             Adds(dyLocal, dyLocal, 0.0f, this->processDataNum);
-             Mul(zLocal, xLocal, xLocal, this->processDataNum);  // x*2
-             // REDUNDANT: multiply by 1.0 (adds computational overhead)
-             Muls(zLocal, zLocal, 1.0f, this->processDataNum);
-             Muls(zLocal, zLocal, COEFF0, this->processDataNum); // x^2*C... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`parallelism`
- slow duration us：`3533.790771`
- fast duration us：`2390.787842`
- improvement ratio：`0.3234495200961064`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_to_gm_kb: 11008.2 -> 16512.2 (+50.00%)
- arithmetic.aiv_vec_fops: 1.37606e+08 -> 2.06408e+08 (+50.00%)
- memory_ub.gm_to_ub_kb: 22016.6 -> 33024.6 (+50.00%)
- critical_path.aiv_vector0_time_us_mean: 2384.26 -> 3527.42 (+47.95%)
- latency.task_duration_us: 2390.79 -> 3533.79 (+47.81%)

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
  "latency.task_duration_us",
  "parallelism.active_block_count",
  "parallelism.block_count",
  "aiv_utilization.mte2_ratio_mean",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "memory_ub.gm_to_ub_bw_usage_rate_pct"
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

- In op_host/gelu_grad.cpp, avoid rounding coreNum to power-of-two when it reduces total parallelism
- Use maximum available cores: coreNum = min(coreNum, inputLengthAlgin32 / BLOCK_SIZE)
- Monitor parallelism.block_count and parallelism.active_block_count metrics
- Check if aiv_vec_fops increases proportionally with reduced block_count
- Verify that memory_ub.gm_to_ub_kb and ub_to_gm_kb don't increase with reduced parallelism

## 检索线索

- coreNum
- power-of-two
- parallelism
- block_count
- active_block_count
- gelu_grad
- vector_operator
- memory_transfer
- aiv_vec_fops
- gm_to_ub
