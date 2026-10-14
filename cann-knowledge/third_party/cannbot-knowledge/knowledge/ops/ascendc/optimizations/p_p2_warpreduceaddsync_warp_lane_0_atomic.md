---
type: Optimization
title: "WarpReduceAddSync + warp-lane-0 atomic"
description: "cpp // Anti-pattern: per-thread atomicAdd → 512 calls/block // Correct: warp reduce then lane-0 atomicAdd → 16 calls/block float warp_sum = Simt::WarpReduceAddSync(partial_sum); Simt::ThreadBarrier();"
tags: [task.optimization, topic.implementation.synchronization, topic.programming_model.execution_model.simt, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p2-warpreduceaddsync-warp-lane-0-atomic.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# WarpReduceAddSync + warp-lane-0 atomic

> 原 OKF v1 卡号：**P-P2**。

## 优化点

```cpp
// Anti-pattern: per-thread atomicAdd → 512 calls/block
// Correct: warp reduce then lane-0 atomicAdd → 16 calls/block
float warp_sum = Simt::WarpReduceAddSync(partial_sum);
Simt::ThreadBarrier();
if (threadIdx.x % 32 == 0 && warp_sum != 0.0f)
    atomicAdd(dst, warp_sum);
```

**Precondition**: dst must be pre-zeroed on the host side. Measured SG backward: 0.865ms → 0.121ms (7.1x).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P2，convert_patterns_to_okf.py）。confidence 未升格。 -->
