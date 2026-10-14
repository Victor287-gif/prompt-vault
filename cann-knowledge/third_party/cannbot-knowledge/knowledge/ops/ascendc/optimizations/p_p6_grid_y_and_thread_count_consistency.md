---
type: Optimization
title: "grid_y and thread count consistency"
description: "The host-side grid_y computation must match the actual thread count used by the dispatcher. Inconsistency between the two causes incorrect work distribution."
tags: [task.optimization, topic.implementation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p6-grid-y-and-thread-count-consistency.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# grid_y and thread count consistency

> 原 OKF v1 卡号：**P-P6**。

## 优化点

The host-side grid_y computation must match the actual thread count used by the dispatcher. Inconsistency between the two causes incorrect work distribution.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P6，convert_patterns_to_okf.py）。confidence 未升格。 -->
