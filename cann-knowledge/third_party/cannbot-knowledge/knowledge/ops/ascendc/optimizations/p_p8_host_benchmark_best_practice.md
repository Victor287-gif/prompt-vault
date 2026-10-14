---
type: Optimization
title: "Host benchmark best practice"
description: "Warm up 3 times, time 10 NPU iterations, compare precision with audited CPU truth, then run boundary tests."
tags: [task.optimization, topic.precision, topic.performance.analysis]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p8-host-benchmark-best-practice.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Host benchmark best practice

> 原 OKF v1 卡号：**P-P8**。

## 优化点

Warm up 3 times, time 10 NPU iterations and take the mean, compare precision
against audited CPU truth, then run boundary tests (`edges=0`, `dim=1`, `dim=3`).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P8，convert_patterns_to_okf.py）。confidence 未升格。 -->
