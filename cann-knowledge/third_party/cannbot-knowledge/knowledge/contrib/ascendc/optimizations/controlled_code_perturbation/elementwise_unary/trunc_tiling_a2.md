---
type: Optimization
title: "trunc：Tiling优化"
description: "trunc 算子的Tiling优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/docs/Trunc.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/op_host/trunc.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/op_host/trunc_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/op_kernel/copy_kl.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/op_kernel/sync_kl.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/op_kernel/trunc.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/op_kernel/trunc_.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/op_kernel/trunc_f32.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/trunc/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# trunc：Tiling优化

## 来源信息

- experience ID：`67bc0f4a70e165ae`
- 源 schema version：`v2`
- 算子：`trunc`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.ub.buffer.count`
- canonical family：`tiling.ub.buffer.count`
- 源记录时间：`2026-04-14T20:57:08`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Conditional UB buffer count based on data type size
~~~~

### Original summary

~~~~text
This optimization uses conditional UB buffer allocation based on data type size to adjust tile sizes. For 2-byte data types (like fp16), allocating more UB buffers (10 instead of 6) reduces tile size per iteration, which increases the total iteration count and improves vector unit utilization. This technique is effective for memory-bound operations where increasing iteration count can better hide memory latency and improve pipeline efficiency.
~~~~

### Original root mechanism

~~~~text
Applying data-type-dependent UB buffer allocation reduces tile size for 2-byte types, increasing iteration count and improving vector utilization
~~~~

### Original causal chain

~~~~text
Conditional UB buffer allocation (10 for 2-byte types, 6 otherwise) -> Reduced tile size per iteration -> Increased iteration count -> Improved vector utilization from 28.33% to 39.23% -> Reduced pipeline stalls (MTE3 wait ratio from 27.74% to 38.33%) -> Latency improvement from 2704μs to 1773μs (34.4% faster)
~~~~

## 适用条件

- When implementing vector operations with small data types (2-byte types like fp16)
- When UB buffer allocation is fixed regardless of data type size
- When operations show low vector utilization (below 30%) and high MTE3 wait ratios
- When memory bandwidth usage is suboptimal (GM-to-UB bandwidth usage below 20%)

## 失效边界

- When data types are larger than 2 bytes (4-byte types like fp32)
- When UB memory is already heavily constrained
- When increasing iteration count would cause excessive overhead from loop control
- When the operation is already compute-bound rather than memory-bound

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- trunc/op_host/trunc.cpp

### 改动符号

- TilingFunc

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`a9f7093d896bc82c5c60136470ad49cf99a7c37b758d9c53f13a16c3c0cf7ceb`

````diff
--- Before/op_host/trunc.cpp
+++ After/op_host/trunc.cpp
@@ -23,7 +23,7 @@
     uint32_t inputLength = inputBytes * inputNum; //输入长度

     // 可使用的ub空间 输入3输出1，手动考虑双缓存
-    uint32_t ubDataNumber = 6;//(inputBytes == 2) ? 10 : 6;
+    uint32_t ubDataNumber = (inputBytes == 2) ? 10 : 6;

     // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
     uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / DOUBLE_BUFFER) / ubDataNumber; //每个ub段可用的空间块数
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`2704.294189`
- fast duration us：`1773.29541`
- improvement ratio：`0.34426682673317693`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1772.74 -> 2703.66 (+52.51%)
- latency.task_duration_us: 1773.3 -> 2704.29 (+52.50%)
- aiv_utilization.icache_miss_rate_mean: 0.000207 -> 0.000113 (-45.41%)
- cache_l2.aiv_read_hit_rate_pct: 2.39 -> 3.43 (+43.51%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.069452 -> 0.03943 (-43.23%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- cache_l2

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1772.744,
      "after": 2703.661,
      "delta": 930.9170000000001,
      "rel": 0.5251277116154393
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1773.295,
      "after": 2704.294,
      "delta": 930.9989999999998,
      "rel": 0.5250107850075705
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 0.000207,
      "after": 0.000113,
      "delta": -9.4e-05,
      "rel": -0.45410628019323673
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 2.39,
      "after": 3.43,
      "delta": 1.04,
      "rel": 0.4351464435146443
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.069452,
      "after": 0.03943,
      "delta": -0.030022,
      "rel": -0.43226976904912745
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

- In the TilingFunc in trunc/op_host/trunc.cpp, implement conditional UB buffer allocation based on inputBytes
- Use pattern: uint32_t ubDataNumber = (inputBytes == 2) ? 10 : 6;
- Calculate tileBlockNum as (ubSize / BLOCK_SIZE / DOUBLE_BUFFER) / ubDataNumber
- Monitor vector utilization (aiv_utilization.vec_ratio_mean) and MTE3 wait ratio (pipeline_stalls.aiv_mte3_wait_ratio_mean) to validate improvement
- Verify that GM-to-UB bandwidth usage rate increases (from ~12% to ~18% in this case)

## 检索线索

- UB buffer allocation
- data type dependent tiling
- vector utilization improvement
- MTE3 wait reduction
- memory bandwidth optimization
- 2-byte data types
- fp16 optimization
- tile size adjustment
- pipeline stall bound
- iteration count increase
