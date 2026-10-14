---
type: Optimization
title: 数值稳定性与调试经验聚合
description: 从 14 条数值稳定性/调试/API YAML 经验中蒸馏：exp 溢出防护、fp32 累加器、除零防护与编译错误快速定位。
tags: [task.optimization, topic.precision, topic.compilation, topic.implementation.debugging]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 数值稳定性与调试经验聚合

从 14 条数值稳定性/调试/API 规范 YAML 经验中蒸馏。这些是保证 kernel 正确性（而不仅是性能）的基石。

## 数值稳定性

| 原则 | 做法 | 原因 |
|------|------|------|
| **exp 溢出防护** | 升 fp32 → 减 max → exp | fp16 下 exp(x) 在 x>11 时溢出 |
| **fp32 累加器** | sum/mean/var 用 fp32 | fp16 累加累积误差严重 |
| **除零防护** | `tl.maximum(denom, eps)` | 防止 Inf/NaN 传播 |
| **rsqrt 优先** | `tl.rsqrt(x+eps)` | 比 `1/tl.sqrt(x)` 更准更快 |

## 调试检查清单

| 检查项 | 说明 |
|--------|------|
| **mask 检查** | 所有 `tl.load/tl.store` 必须带 mask 或 boundary_check |
| **stride 检查** | stride 参数必须与张量实际内存布局匹配 |
| **索引越界** | `offset < n` 条件判断 |
| **shape/dtype** | host 侧确认 tensor.shape/stride/dtype |
| **编译错误** | UB 超限→减小 tile; Grid 超限→grid-stride; NaN/Inf→数值稳定性 |

