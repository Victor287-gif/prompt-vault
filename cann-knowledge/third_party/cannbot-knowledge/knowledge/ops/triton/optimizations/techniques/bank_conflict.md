---
type: Optimization
title: Bank Conflict 消除
description: Triton Ascend Bank Conflict 诊断与消除：对角线遍历、Padding 对齐与 L2 Cache 局部性优化。
tags: [task.optimization, topic.memory, topic.data_movement, topic.tiling, topic.alignment]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Bank Conflict 消除

Bank Conflict 发生在多个 AI Core 同时访问同一 bank 的 L2/UB 时，导致访存串行化。大矩阵（M,N≥512）场景下性能损失可达 25-35%。

## 症状

- `aiv_mte2_time` 远高于理论值（>2×）
- M,N 越大性能退化越严重
- Profiling 显示大量 MTE2 停顿

## OPT-triton-bank-001: 对角遍历

行/列优先 PID 映射导致相邻 core 同时访问同一行/列 → bank conflict。对角线遍历将相邻 core 映射到不同 (M,N) tile：

```python
# 对角线 PID 映射
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
group_id = pid // (GROUP_SIZE * num_pid_n)
pid_m = group_id * GROUP_SIZE + (pid % GROUP_SIZE)
pid_n = (pid % (GROUP_SIZE * num_pid_n)) // GROUP_SIZE
```
触发条件：M≥512, N≥512

## OPT-triton-bank-002: Padding 对齐

对维度做 padding 使其对齐到 bank 边界（512B），消除跨 bank 访问冲突。

## OPT-triton-bank-003: 512B 行宽约束

matmul 中 BLOCK_N × dtype_bytes 必须为 512B 的整数倍，不满足时自动产生跨 bank 访问。

