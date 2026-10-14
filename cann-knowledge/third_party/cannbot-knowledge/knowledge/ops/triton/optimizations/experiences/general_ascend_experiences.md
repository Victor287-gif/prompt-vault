---
type: Optimization
title: Ascend 平台专项经验聚合
description: 从 16 条 Ascend 平台专项 YAML 经验中蒸馏：Grid 配置、constexpr 规则、禁用语法规避与 Ascend 特有模式。
tags: [task.optimization, topic.compilation, topic.programming_model.execution_unit.scalar, topic.implementation.masking, topic.parallelism, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# Ascend 平台专项经验聚合

从 16 条 Ascend 平台专项 YAML 经验中蒸馏。这些是 Triton-Ascend 与标准 GPU Triton 的差异点——实现任何 kernel 前必读。

## 共性模式

1. **Grid 是物理核组**：grid_size ≤ AI Core 数（默认 ≤65535），超出用 grid-stride loop
2. **禁用语法**：`tl.swizzle2d`、`return`/`break`/`continue`、链式布尔 `a and b and c`、lambda
3. **int64/int32 陷阱**：`tl.arange` 默认 int64，CMP/ADD 退化为 Scalar

## 知识要点

| 领域 | 关键约束 | 替代方案 |
|------|---------|---------|
| **Grid** | x*y*z ≤ 65535, 最多 3 维，必须 tuple | grid-stride loop 处理大 shape |
| **constexpr** | 仅 kernel 参数可用 | host 侧用普通 Python 变量 |
| **swizzle2d** | Ascend 不支持 | `grouped_launch_diagonal` |
| **索引/切片** | 无 `tensor[i]`, 无 `[start:end]` | `tl.load(ptr+offset)` |
| **条件选择** | 无 `if-else` 在 SIMT 路径 | `tl.where(cond, x, y)` |
| **负偏移** | 禁止负数指针偏移 | `tl.maximum(offset, 0)` + mask |
| **核心数** | 910b2: ~20 Cube + 48 Vector | `torch_npu.npu_core_num()` |
| **大 shape 分块** | 单 kernel 无法处理超长序列 | grid-stride, 每 program 多 block |

