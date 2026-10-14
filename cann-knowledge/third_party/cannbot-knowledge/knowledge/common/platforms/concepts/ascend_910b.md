---
type: Concept
title: Ascend 910B2/910B3 硬件规格
description: Ascend 910B2/910B3 AI Core 架构、Grid 语义、内存层次与编译选项参考。Triton Ascend 开发必读。
tags: [task.lookup, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/cv_autotune/hardware_consts.py
  role: implementation
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Ascend 910B2/910B3 硬件规格

## 核心架构

每个 Ascend AI Core 包含两个独立的计算单元：

| 单元 | 用途 | 操作 |
|------|------|------|
| **Cube** | 矩阵乘法/GEMM | `tl.dot`、batch matmul、卷积 |
| **Vector** | 逐元素/归一化 | `tl.exp`、`tl.sum`、算术、LayerNorm |

二者可通过 `dl.async_task` 在同一 AI Core 内并发执行。

## Grid 语义：物理核绑定

Ascend Grid 与 GPU CUDA Grid 有根本区别：

| 维度 | CUDA/GPU | Ascend |
|------|----------|--------|
| Grid 含义 | 逻辑任务维度 | 物理核组 |
| 硬件映射 | 驱动自动映射 | 必须匹配物理拓扑 |
| Grid 大小上限 | 极大（百万级 block） | ≤ AI Core 总数 |
| 2D Grid | 逻辑 (rows × cols) | 须匹配物理拓扑 |
| 单 kernel 上限 | 平台相关 | ≤ 65535 |

**关键含义**：不能启动超过物理 AI Core 数的 grid block，否则需 `TRITON_ALL_BLOCKS_PARALLEL` 环境变量或多轮调度。

## 内存层次

| 层级 | 大小 | 延迟 | 用途 |
|------|------|------|------|
| **UB (L0)** | 192KB / Core | 极低 | tile 数据、寄存器溢出 |
| **L1** | ~几 MB / Core | 低 | Core 间共享 |
| **L2** | ~32MB / 芯片 | 中 | 全局缓存、多核复用 |
| **HBM (GM)** | 64GB (910B3) | 高 | 全局内存 (~1.8TB/s 带宽) |

## 编译选项

编译 Triton-Ascend kernel 时的关键选项：

- `TRITON_ALL_BLOCKS_PARALLEL=1` — 绕过 grid 核数限制（生产环境不推荐）
- `ASCEND_LAUNCH_BLOCKING=1` — 同步启动，便于调试
- `TRITON_CACHE_DIR` — autotune 缓存目录