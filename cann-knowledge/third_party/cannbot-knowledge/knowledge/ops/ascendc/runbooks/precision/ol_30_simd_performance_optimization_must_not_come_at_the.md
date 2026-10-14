---
type: Runbook
title: "SIMD performance optimization must not come at the cost of precision"
description: "SIMD performance optimization must not come at the cost of precision"
tags: [task.debugging, topic.precision, topic.programming_model.execution_model.simt, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-30-simd-performance-optimization-must-not-come-at-the.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3]
aliases: ["OL-30"]
---

# SIMD performance optimization must not come at the cost of precision

> 原 OKF v1 卡号：**OL-30**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - SIMD kernel optimization attempts, especially eliminating per-group/per-block loops

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process, trust_calibration
- **Loaded by**: Builder, Optimizer, QA
- **Trigger**: SIMD kernel optimization attempts, especially eliminating per-group/per-block loops
- **Lesson**: MXFP4 SIMD V4 "fast" replaced per-32-group exponent with a tile-wide shared exponent (1024 elements shared).
  Performance did improve (small tensors 1.08x faster than SIMT), but **precision no longer matches MXFP4 spec**.
  A3's hand-written SIMD has the same issue — BATCH=512 sharing one exponent is exactly the root cause of its precision bug.
  **Never publish a "faster but numerically wrong" kernel as production**.
  **Hard rules**:
  1. After any performance optimization, must compare against PyTorch spec precision (not only against the previous AscendC version)
  2. If an optimization requires changing algorithmic precision semantics (e.g. enlarging group_size), must be **explicitly labeled "approximate"** in docs
  3. "0 mismatch precision" is only meaningful when compared against the PyTorch spec. Comparing against our own CPU ref does not count (our ref may have the same bug)
  4. The A3 implementation's precision bug is precisely because per-group precision was abandoned for SIMD performance — this is not an acceptable trade-off
- **Evidence**: MXFP4 SIMD V4 (2026-04-07), A3 hand-written SIMD precision bug analysis

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-30（category=process, trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
