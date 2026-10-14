---
type: Optimization
title: "Cooperative-group parallel traversal + shuffle reduction"
description: "Transform single-thread sequential traversal into N-thread cooperative parallel + __shfl_xor divide-and-conquer reduction. cpp auto rank = threadIdx.x % GROUP_SIZE; for (uint32_t pos = rank; pos < arr"
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p13-cooperative-group-parallel-traversal-shuffle-red.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Cooperative-group parallel traversal + shuffle reduction

> 原 OKF v1 卡号：**P-P13**。

## 优化点

Transform single-thread sequential traversal into N-thread cooperative parallel + `__shfl_xor` divide-and-conquer reduction.

```cpp
auto rank = threadIdx.x % GROUP_SIZE;
for (uint32_t pos = rank; pos < array_size; pos += GROUP_SIZE) { ... }
for (int32_t offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
    auto other = __shfl_xor(val, offset, GROUP_SIZE);
    if (other < val) val = other;  // or +=, max, etc.
}
```

Applicable to any scenario requiring min/max/sum within a warp.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/cooperative.md（P-P13，convert_patterns_to_okf.py）。confidence 未升格。 -->
