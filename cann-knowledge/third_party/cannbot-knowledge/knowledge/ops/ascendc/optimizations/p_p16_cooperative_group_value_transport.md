---
type: Optimization
title: "Cooperative-group value transport"
description: "GROUP_SIZE threads share the work of a large vector copy: cpp for (uint32_t j = rank; j < dim; j += GROUP_SIZE) { dst[pos dim + j] = src[idx dim + j]; } At dim=128: 128 stores → 16 threads each do 8 s"
tags: [task.optimization, topic.programming_model.execution_unit.vector]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p16-cooperative-group-value-transport.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: []
---

# Cooperative-group value transport

> 原 OKF v1 卡号：**P-P16**。

## 优化点

GROUP_SIZE threads share the work of a large vector copy:
```cpp
for (uint32_t j = rank; j < dim; j += GROUP_SIZE) {
    dst[pos * dim + j] = src[idx * dim + j];
}
```

At dim=128: 128 stores → 16 threads each do 8 stores. Applicable to any large embedding/feature vector transport.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/cooperative.md（P-P16，convert_patterns_to_okf.py）。confidence 未升格。 -->
