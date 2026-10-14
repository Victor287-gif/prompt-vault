---
type: Guide
title: IR 分析工作流（IR 优先的性能分析决策）
description: 以 last_pass.mlir 为主、msprof 为辅的性能分析决策流程：IR 四项检查点、msprof 指标判读与 caveat、瓶颈→优化动作映射，以及
  Phase 4 IR 多轮迭代规则。
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.scalar, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 功能说明

性能分析**优先看 IR**（由 [IR 提取](ir_extraction.md)产出的 `last_pass.mlir`），msprof 指标只作补充验证。IR 揭示真实的计算结构、tiling 策略与流水重叠；单看 msprof 比率会误判（如 `scalar_ratio` 高可能被流水重叠掩盖，并非真瓶颈）。

## Step 1：IR 分析（主要）

检查四类模式：

- **标量繁重模式**：过多 `arith` 操作（int64 比较、类型转换）且未被流水重叠吸收。
- **访存模式**：非连续 load/store、冗余拷贝、tiling 未对齐。
- **流水结构**：Vector 与 Cube 操作是否重叠；scalar 操作在关键路径上还是被数据传输掩盖。
- **同步/屏障密度**：相互独立的操作之间是否存在多余 barrier。

## Step 2：msprof 指标（补充，验证 IR 发现）

| 指标 | 理想值 | Caveat |
|------|--------|--------|
| `aiv_vec_ratio` | > 80% | 偏低可能是 Vector 流水 stalled，也可能是 Cube/传输占主导——与 IR 交叉确认 |
| `aiv_mte2_ratio` | < 50% | 偏高证实 IR 中看到的访存瓶颈 |
| `aiv_scalar_ratio` | < 20% | **偏高不必然 scalar-bound**：scalar 被流水重叠掩盖（MTE2 等待覆盖 scalar 执行）时不是瓶颈；仅当 IR 确认 scalar 在关键路径才按标量降级处理 |
| `aic_cube_ratio` | > 80% | 偏低可能 Cube 流水 stalled——与 IR tiling 交叉确认 |
| `aic_mte1_ratio` | 适中 | 偏高证实 IR 中看到的 L1→L0A/L0B 传输瓶颈 |

## 瓶颈 → 优化动作

| 瓶颈 | 动作 |
|------|------|
| 计算 bound（vec/cube ratio 高） | 增大 tiling、CV 融合 |
| 访存 bound（mte2 ratio 高） | MultiBuffer、`care_padding=False`、连续访存 |
| 标量降级（IR 确认在关键路径） | 类型转换（int64→int32、int 比较→fp32 比较）、`where`→`get_element`+`insert_slice` |
| 流水掩盖的 scalar（ratio 高但 IR 显示被重叠） | 不处理——scalar 被传输延迟掩盖 |
| 同步开销 | sync_solver、减少 barrier 数 |

## IR 多轮迭代模式（优化点 31）

triton-op-generator Phase 4 开启 IR 多轮迭代后，IR（优化点 31）允许跨轮重复命中，其他优化点单轮即过。每轮规则：

- **每轮重新提取 IR**：必须重新执行 `run_and_extract.sh` 提取基于"上一轮优化后代码"的最新 `last_pass.mlir`，禁止复用历史 IR 快照。
- **聚焦增量分析**：(a) 上一轮 IR 优化是否带来预期效果（profiling / speedup 对比）；(b) 是否还有**新的**优化建议。
- **返回字段**：`ir_has_more_suggestions: bool`——本轮还能给出新建议为 `true`；结论与上轮相同、无新增空间为 `false`，Phase 4 据此退出 IR 迭代。
- **退出条件**：达到 `ir_max_iterations`（默认 20）或 `ir_has_more_suggestions == false` 任一成立。
- **建议命中历史优化点时**：已执行过 → 重新执行并诊断为何之前未生效；未命中过 → 执行并分析未命中原因；无对应优化点 → 直接应用 IR 分析建议。

> ⚠️ **流程差异**：本卡的 Step 1/Step 2 出自迁移向导流（msprof 作为 IR 的补充验证）；而 triton-op-generator Phase 4 流程规定**优化耗尽前严禁使用 msprof**（simulator 采集是 latency-optimizer 普通点 + IR 点均耗尽且未达 target 后的分支）。两者上下文不同，按所属流程执行，不要混用顺序。