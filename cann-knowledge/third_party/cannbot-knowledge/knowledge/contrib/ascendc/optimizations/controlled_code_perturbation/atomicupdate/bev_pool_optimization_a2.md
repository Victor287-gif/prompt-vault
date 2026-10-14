---
type: Optimization
title: "bev_pool：冗余消除优化"
description: "bev_pool 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.atomic", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/docs/BevPool.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/framework/tf_plugin/tensorflow_bev_pool_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/op_host/bev_pool.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/op_host/bev_pool_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/op_kernel/bev_pool.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/tests/st/BevPool_half_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/tests/st/test_bev_pool_half.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/bev_pool/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# bev_pool：冗余消除优化

## 来源信息

- experience ID：`0df559a56fcce54c`
- 源 schema version：`v2`
- 算子：`bev_pool`
- source iteration：`round_009_refactor`
- 源 category：`math`
- 计算分区：`atomicupdate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.initialization`
- canonical family：`remove.redundant.initialization`
- 源记录时间：`2026-04-15T01:22:46`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing redundant output buffer zero-initialization loop
~~~~

### Original summary

~~~~text
In Ascend C operators where output buffers are already initialized to zero by InitGlobalMemory() calls in the Init() function, removing explicit zero-initialization loops in Process() can improve performance by 6.8%. This optimization reduces scalar operations and memory bandwidth usage while maintaining correctness.
~~~~

### Original root mechanism

~~~~text
The Init() function already calls InitGlobalMemory() to initialize the output buffer to zero, making the loop in Process() redundant. Removing it preserves semantic equivalence while eliminating duplicate work.
~~~~

### Original causal chain

~~~~text
Removing redundant zero-initialization loop in Process() → reduces scalar_ratio_mean by 73% and vector memory bandwidth by 6.3% → decreases latency by 6.8% (539,706μs)
~~~~

## 适用条件

- Operator uses InitGlobalMemory() in Init() function to initialize output buffers
- Process() function contains explicit zero-initialization loops for already-initialized buffers
- Scalar operations show high utilization (scalar_ratio_mean > 4%)
- Memory bandwidth usage shows potential for reduction

## 失效边界

- Output buffer initialization in Init() is not guaranteed to be zero
- Buffer requires non-zero initialization values
- InitGlobalMemory() does not fully cover the buffer range used in Process()

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- op_kernel/bev_pool.cpp

### 改动符号

- KernelBevPool::Process()

### 代码签名

- InitBuffer
- SetValue
- TBuf
- queue_api
- scalar_value_access
- tbuf

### 原始 diff

- 源 diff SHA-256：`ba13f81a37c10060f21de85df53cd8c988dcbfef805defdf8d55ba83824d86b3`

````diff
--- Before/op_kernel/bev_pool.cpp
+++ After/op_kernel/bev_pool.cpp
@@ -38,10 +38,6 @@
         pipe.InitBuffer(QueueTmp3, 32 * sizeof(float));
     }
     __aicore__ inline void Process() {
-        for(int32_t i=0;i<this->B*this->C*this->D_Z*this->D_Y*this->D_X;i++)
-        {
-            outGm.SetValue(i, 0);
-        }
         int32_t b1 = this->D_Z * this->D_Y * this->D_X * this->C;
         int32_t b2 = this->D_Y * this->D_X * this->C;
         int32_t b3 = this->D_X * this->C;
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`7982455.5`
- fast duration us：`7442749.5`
- improvement ratio：`0.06761152630290265`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.041409 -> 0.071658 (+73.05%)
- latency.task_duration_us: 7.44275e+06 -> 7.98246e+06 (+7.25%)
- critical_path.aiv_vector0_time_us_mean: 7.4426e+06 -> 7.9823e+06 (+7.25%)
- memory_ub.ub_read_bw_vector_gbps: 0.79 -> 0.74 (-6.33%)
- memory_ub.ub_write_bw_vector_gbps: 0.79 -> 0.74 (-6.33%)

### Affected metric groups

- aiv_utilization
- latency
- critical_path
- memory_ub

### Changed metrics top

~~~~json
[
  "aiv_utilization.scalar_ratio_mean",
  "latency.task_duration_us",
  "critical_path.aiv_vector0_time_us_mean",
  "memory_ub.ub_read_bw_vector_gbps",
  "memory_ub.ub_write_bw_vector_gbps"
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

- Check if Init() function calls InitGlobalMemory() for output buffers
- Verify Process() contains loops calling SetValue(i, 0) on already-initialized buffers
- Remove redundant zero-initialization loops while ensuring semantic equivalence
- Profile scalar_ratio_mean and ub_read_bw_vector_gbps metrics to confirm improvements

## 检索线索

- redundant initialization
- zero initialization
- InitGlobalMemory
- SetValue
- scalar operations
- memory bandwidth
- bev_pool
- buffer initialization
