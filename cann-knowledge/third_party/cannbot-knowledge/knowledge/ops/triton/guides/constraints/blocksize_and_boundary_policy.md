---
type: Guide
title: BLOCK_SIZE 配置选择与边界处理策略
description: 在草图中必须添加"设计适用范围"注释：。
tags: [task.design, topic.implementation.masking, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# BLOCK_SIZE 配置选择与边界处理策略

> 源自 designer `hint-mode.md` 的后半部分；前半的 Hint DSL 语法属 skill
> 私有中间表示，按 SPEC §2.3 未迁入。

## BLOCK_SIZE 配置选择原则

### 参数范围较小（如 M in [128, 256]）
- 避免过大 BLOCK_SIZE，推荐 [32, 64, 128]
- 确保最小值至少能容纳一个 BLOCK

### 参数范围较大（如 M in range(128, 8192)）
- 提供多种选择 [64, 128, 256, 512]
- 使用 autotune 适配不同大小

## 边界处理策略

### 选项 A：使用 mask（推荐）
- 支持任意 shape
- 在草图注释中说明：`使用 mask，支持任意 shape`

### 选项 B：不使用 mask（性能更优）
- 要求 shape 整除 BLOCK_SIZE
- 在草图注释中说明：`M % 64 == 0（不使用 mask）`

## 设计适用范围注释

在草图中必须添加"设计适用范围"注释：

```python
# 设计适用范围：
# - 参数范围：M in [128, 2048], N in [128, 4096]
# - 边界处理：使用 mask / 不使用 mask（需整除）
# - BLOCK_SIZE 配置：[64, 128, 256]
# - 约束条件：如不使用 mask，M % 64 == 0
```

**重要**：
- 无论 sketch 用几维，"设计适用范围"注释中必须对原始输入的每个维度分别说明范围
- 不要只写总元素数的范围，要分别写每个维度的范围
- 如果是 2 的幂次也需要标出
