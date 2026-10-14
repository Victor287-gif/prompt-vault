---
type: Optimization
title: "reshape：循环计算优化"
description: "reshape 算子的循环计算优化经验，关注数据搬运。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/docs/Reshape.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/tf_plugin/tensorflow_reshape_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_host/reshape.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_host/reshape_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_kernel/reshape.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# reshape：循环计算优化

## 来源信息

- experience ID：`62886a2d3d84a82f`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_007_degrade`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`loop.redundant.duplication`
- canonical family：`loop.redundant.duplication`
- 源记录时间：`2026-04-21T00:04:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant loop duplication in tile processing
~~~~

### Original summary

~~~~text
In Ascend C operators with tile-based processing patterns, avoid duplicating CopyIn and CopyOut operations for the same tile data. Each tile should be processed only once per iteration to prevent unnecessary memory bandwidth consumption. This optimization reduces GM-to-UB and UB-to-GM transfers by 50% for reshape-like operators that perform simple data movement.
~~~~

### Original root mechanism

~~~~text
Processing each tile twice in the execution loop - once for 'verification' and once for 'actual' work - effectively doubles the computational workload and memory transfers without changing functional behavior.
~~~~

### Original causal chain

~~~~text
Removing redundant CopyIn/CopyOut calls → GM-to-UB and UB-to-GM transfer volumes halved from 739,246KB to 369,623KB → memory bandwidth utilization reduced from 3.06 Gbps to 2.27 Gbps → task duration decreased from 230,773μs to 155,273μs (32.7% improvement)
~~~~

## 适用条件

- Operator uses tile-based processing with CopyIn/CopyOut patterns in reshape.cpp
- Memory-bound operators where GM-to-UB and UB-to-GM transfers dominate performance
- Operators with simple data movement patterns (no computation between transfers)
- When memory_ub.gm_to_ub_kb and memory_ub.ub_to_gm_kb metrics show high values relative to data size

## 失效边界

- Operators with complex computation between memory transfers
- When data validation or verification is required between transfers
- When memory transfers need to be pipelined with computation for latency hiding

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- reshape/op_kernel/reshape.cpp

### 改动符号

- MyReshape::exec

### 代码签名

- InitBuffer
- TBuf
- queue_api
- tbuf
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`4d601df20c2a942bf81fc9558e0b97c3dacdd8eb85a88171e02b2a9767377f77`

````diff
--- Before/op_kernel/reshape.cpp
+++ After/op_kernel/reshape.cpp
@@ -57,12 +57,23 @@
         p->InitBuffer(q_data, 1, this->ti.tileLength * sizeof(DATA));
     }
     __aicore__ inline void exec() {
+        // Degrade: process each tile twice - once for "verification" and once for "actual" work
+        // This adds redundant computation without changing functional behavior
         for(uint32_t i = 0; i < TI.tileNumber; i++) {
+            // First pass: "verification" copy (redundant)
+            CopyIn(i * TI.tileLength, TI.tileLength);
+            CopyOut(i * TI.tileLength, TI.tileLength);
+            // Second pass: "actual" copy (functional)
             CopyIn(i * TI.tileLength, TI.tileLength);
             CopyOut(i * TI.tileLength, TI.tileLength);
         }
-        CopyIn(TI.tileNumber * TI.tileLength, TI.reminder);
-        CopyOut(TI.tileNumber * TI.tileLength, TI.reminder);
+        // Handle reminder with same redundant pattern
+        if (TI.reminder > 0) {
+            CopyIn(TI.tileNumber * TI.tileLength, TI.reminder);
+            CopyOut(TI.tileNumber * TI.tileLength, TI.reminder);
+            CopyIn(TI.tileNumber * TI.tileLength, TI.reminder);
+            CopyOut(TI.tileNumber * TI.tileLength, TI.reminder);
+        }
     }
 };
 extern "C" __global__ __aicore__ void reshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
````

## 观测记录

- bottleneck type：`memory_transfer_bound`
- slow duration us：`230772.828125`
- fast duration us：`155273.296875`
- improvement ratio：`0.32715953547661625`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_to_gm_kb: 369623 -> 739246 (+100.00%)
- memory_ub.gm_to_ub_kb: 369623 -> 739246 (+100.00%)
- critical_path.aiv_vector0_time_us_mean: 155233 -> 230742 (+48.64%)
- latency.task_duration_us: 155273 -> 230773 (+48.62%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 1.16 -> 1.57 (+35.34%)

### Affected metric groups

- memory_ub
- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 369622.8,
      "after": 739245.5,
      "delta": 369622.7,
      "rel": 0.999999729453919
    }
  },
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 369622.9,
      "after": 739245.6,
      "delta": 369622.69999999995,
      "rel": 0.9999994589079841
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 155232.533,
      "after": 230741.802,
      "delta": 75509.269,
      "rel": 0.4864268303861279
    }
  },
  {
    "latency.task_duration_us": {
      "before": 155273.297,
      "after": 230772.828,
      "delta": 75499.53100000002,
      "rel": 0.4862364132063224
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 1.16,
      "after": 1.57,
      "delta": 0.41000000000000014,
      "rel": 0.3534482758620691
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 1.29,
      "after": 1.74,
      "delta": 0.44999999999999996,
      "rel": 0.34883720930232553
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 2.27,
      "after": 3.06,
      "delta": 0.79,
      "rel": 0.3480176211453745
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 2.27,
      "after": 3.06,
      "delta": 0.79,
      "rel": 0.3480176211453745
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.213917,
      "after": 0.28522,
      "delta": 0.07130299999999998,
      "rel": 0.33332086743924033
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 0.036038,
      "after": 0.040669,
      "delta": 0.004630999999999996,
      "rel": 0.12850324657306167
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.488172,
  "overall_deviation": 0.195269,
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

- Inspect the exec() method in reshape.cpp for duplicate CopyIn/CopyOut calls
- Ensure each tile is processed only once in the main loop
- Check that reminder handling follows the same single-pass pattern
- Verify that memory transfer volumes (gm_to_ub_kb, ub_to_gm_kb) match expected data size
- Profile with memory_ub metrics to confirm transfer reduction

## 检索线索

- reshape
- tile
- CopyIn
- CopyOut
- memory_transfer
- redundant_loop
- GM_to_UB
- UB_to_GM
- tile_based_processing
- memory_bound
