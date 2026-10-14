---
type: Optimization
title: "lerp：计算优化"
description: "lerp 算子的计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/docs/Lerp.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/tf_plugin/tensorflow_lerp_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_host/lerp.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_host/lerp_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_kernel/lerp.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/Lerp_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/test_lerp.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# lerp：计算优化

## 来源信息

- experience ID：`1fa44abcdb09d9f0`
- 源 schema version：`v2`
- 算子：`lerp`
- source iteration：`round_019_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.artificial.dependency`
- canonical family：`remove.artificial.dependency`
- 源记录时间：`2026-04-20T22:59:05`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removed artificial computational dependencies in tensor copy operations
~~~~

### Original summary

~~~~text
On `math/lerp`, a structural refactoring (`Removed artificial computational dependencies in tensor copy operations`) that appeared semantically equivalent unexpectedly caused 60.8% performance change (155526.5 vs 397128.5 us, 60.8% difference). This reveals that `scalar_overhead_bound` is sensitive to this code pattern. The dummy loops that read tensor values into volatile variables created artificial data dependencies between tensor copies without affecting the actual data values. Removing them preserves functional correctness while eliminating unnecessary serialization.
~~~~

### Original root mechanism

~~~~text
The dummy loops that read tensor values into volatile variables created artificial data dependencies between tensor copies without affecting the actual data values. Removing them preserves functional correctness while eliminating unnecessary serialization.
~~~~

### Original causal chain

~~~~text
structural change to remove.artificial.dependency in lerp/op_kernel/lerp.cpp -> unexpected shift in latency.task_duration_us -> 60.8% latency impact
~~~~

## 适用条件

- The target operator's profiling shows bottleneck in: latency, critical_path, aiv_utilization
- The kernel code uses related APIs/patterns: DataCopy, GetValue, datacopy, scalar_value_access
- The operator has a `scalar_overhead_bound` bottleneck profile
- You are considering a similar structural refactoring and want to verify it won't affect performance

## 失效边界

- The operator's bottleneck is unrelated to `scalar_overhead_bound`
- The `remove.artificial.dependency` pattern is not present in the target code

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- lerp/op_kernel/lerp.cpp

### 改动符号

- KernelLerp::CopyIn

### 代码签名

- DataCopy
- GetValue
- datacopy
- scalar_value_access
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`aabdbb1db6f927db91447f7ae9a0f15d51126ee4ae8f0c3d1085f6f852960a1b`

````diff
--- Before/op_kernel/lerp.cpp
+++ After/op_kernel/lerp.cpp
@@ -101,23 +101,9 @@

       while (remaining > 0) {
         uint64_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
-        // DEGRADATION: Add artificial computational dependency between tensor copies
-        // Copy start tensor first
+        // Copy all tensors without artificial dependencies
         AscendC::DataCopy(startLocal[offset], startGm[progress * this->ubPartDataNum + offset], chunk);
-        // Artificial dependency: perform dummy computation on start tensor before copying end tensor
-        // This creates a data dependency that prevents parallel copying
-        for (uint64_t i = 0; i < chunk; ++i) {
-          volatile TYPE_START dummy = startLocal.GetValue(offset + i);
-          (void)dummy; // Use the value to prevent optimization
-        }
-        // Copy end tensor second
         AscendC::DataCopy(endLocal[offset], endGm[progress * this->ubPartDataNum + offset], chunk);
-        // Artificial dependency: perform dummy computation on end tensor before copying weight tensor
-        for (uint64_t i = 0; i < chunk; ++i) {
-          volatile TYPE_END dummy = endLocal.GetValue(offset + i);
-          (void)dummy; // Use the value to prevent optimization
-        }
-        // Copy weight tensor last
         AscendC::DataCopy(weightLocal[offset], weightGm[progress * this->ubPartDataNum + offset], chunk);
         offset += chunk;
         remaining -= chunk;

````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`397128.5`
- fast duration us：`155526.453125`
- improvement ratio：`0.6083724710641518`
- machine confidence：`0.5`

### Profile evidence

- latency.task_duration_us: 155526 -> 397128 (+155.34%)
- critical_path.aiv_vector0_time_us_mean: 155268 -> 393351 (+153.34%)
- aiv_utilization.mte2_ratio_mean: 0.090795 -> 0.206351 (+127.27%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.002174 -> 0.00012 (-94.48%)
- aiv_utilization.mte3_ratio_mean: 0.046539 -> 0.018084 (-61.14%)

### Affected metric groups

- latency
- critical_path
- aiv_utilization
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "latency.task_duration_us": {
      "before": 155526.453,
      "after": 397128.5,
      "delta": 241602.047,
      "rel": 1.5534466474330253
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 155267.831,
      "after": 393351.065,
      "delta": 238083.234,
      "rel": 1.5333712879649875
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.090795,
      "after": 0.206351,
      "delta": 0.115556,
      "rel": 1.2727132551351947
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.002174,
      "after": 0.00012,
      "delta": -0.0020540000000000003,
      "rel": -0.9448022079116836
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.046539,
      "after": 0.018084,
      "delta": -0.028454999999999998,
      "rel": -0.6114226777541417
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 0.41,
      "after": 0.16,
      "delta": -0.24999999999999997,
      "rel": -0.6097560975609756
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.000151,
      "after": 5.9e-05,
      "delta": -9.200000000000001e-05,
      "rel": -0.609271523178808
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 0.004388,
      "after": 0.00172,
      "delta": -0.0026680000000000002,
      "rel": -0.6080218778486782
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.003847,
      "after": 0.001508,
      "delta": -0.002339,
      "rel": -0.6080062386275019
    }
  },
  {
    "memory_ub.ub_read_bw_vector_gbps": {
      "before": 1.68,
      "after": 0.66,
      "delta": -1.02,
      "rel": -0.6071428571428572
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

- Before applying a similar structural change, profile the `remove.artificial.dependency` code path in `lerp/op_kernel/lerp.cpp` (key symbols: KernelLerp::CopyIn)
- After the change, verify that `latency.task_duration_us` moves in the expected direction
- If metrics regress, the change is NOT performance-neutral — revert it

## 检索线索

- After
- Artificial
- AscendC
- Before
- CHUNK_SIZE
- Copy
- DEGRADATION
- DataCopy
- GetValue
- TYPE_END
- TYPE_START
- This
- aiv_utilization
- critical_path
- datacopy
- endGm
- endLocal
- latency
- pipeline_stalls
- remove.artificial.dependency
- scalar_overhead_bound
- scalar_value_access
- startGm
- startLocal
- tile
- tile_shape
- ub
- ubPartDataNum
- weightGm
- weightLocal
