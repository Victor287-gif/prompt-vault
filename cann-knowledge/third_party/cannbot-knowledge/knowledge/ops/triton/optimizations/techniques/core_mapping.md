---
type: Optimization
title: Core Mapping 策略
description: Triton Ascend Core Mapping 三种策略（列优先/行优先/对角线）的选择决策、GROUP_SIZE 调参与 L2 局部性原理。
tags: [task.optimization, topic.tiling, topic.memory, topic.parallelism]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Core Mapping 策略

Core Mapping 决定每个 AI Core 处理哪个 (M,N) 输出 tile。策略选择直接影响 L2 缓存复用、负载均衡和 bank conflict 模式。选择不当可损失 20-40% 性能。

## 三种策略

| 策略 | Block ID 模式 | 最适合 | GROUP_SIZE |
|------|-------------|--------|------------|
| **列优先** | 相邻 ID 访问同 N 列 | N-dominant (N>>M) | 8/16/32 |
| **行优先** | 相邻 ID 访问同 M 行 | M-dominant (M>>N) | 8/16/32 |
| **对角线** | 相邻 ID 访问不同 (M,N) | M,N≥512（正方形） | 8/16/32 |

## 决策规则

- M≥512 且 N≥512 → 对角映射（`dl.grouped_launch_diagonal`）
- N>>M → 列优先
- M>>N → 行优先

## OPT-triton-core-mapping-001: GROUP_SIZE 配置

- GROUP_SIZE ∈ {8, 16, 32}，通过 autotune 选取
- GROUP_SIZE > 核心数 → L2 局部性变差
- 至少提供 3 个不同 GROUP_SIZE

