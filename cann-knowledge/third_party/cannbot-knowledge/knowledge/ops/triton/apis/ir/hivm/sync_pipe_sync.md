---
type: API
title: HIVM 管道同步操作（set_flag / wait_flag / pipe_barrier）
description: hivm.hir.set_flag / wait_flag / pipe_barrier 操作语义：同一 Block 内跨 Pipeline
  的 Event Flag 同步，(set_pipe, wait_pipe, event_id) 三元组配对，静态/动态 Event ID，典型 Pipe 组合与完整
  Load→Compute→Store 同步模式。
tags: [task.lookup, topic.compilation, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 管道同步操作（set_flag / wait_flag / pipe_barrier）

管道同步是 HIVM 最细粒度的同步，用于同一 Block 内不同 Pipeline（MTE1/MTE2/MTE3/M/V/FIX）间的数据依赖。冗余的 wait_flag/barrier 是"同步开销"瓶颈的直接来源，分析手段见 [../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md) 同步开销条目。

## 三个操作

| 操作 | 参数 | 说明 |
|------|------|------|
| `hir.set_flag[set_pipe, wait_pipe, event_id]` | 2×HIVM_PipeAttr + Event | 在 set_pipe 上发信号，通知 wait_pipe 数据就绪 |
| `hir.wait_flag[set_pipe, wait_pipe, event_id]` | 同上 | 在 wait_pipe 上阻塞等待该信号 |
| `hir.pipe_barrier[pipe]` | HIVM_PipeAttr | 等待该 Pipe 上所有先前操作完成（重量级屏障） |

Event ID 二选一：`static_event_id`（`#hivm.event<EVENT_IDx>`，编译时确定）或 `dynamic_event_id`（i64 运行时值），不能同时存在。

## Event ID 规则

- 每个 Pipe 有 **8 个** Event ID（EVENT_ID0~EVENT_ID7）。
- set_flag/wait_flag 必须使用**相同**的 (set_pipe, wait_pipe, event_id) 三元组配对。
- 同一 (set_pipe, wait_pipe) 对中，使用中的 Event ID 不得重复（前一对完成 wait 后可复用）。
- 复杂场景下 Event ID 的分配与复用由 InjectSync / GraphSyncSolver 自动管理（见 [sync_sync_injection.md](sync_sync_injection.md)）。

## 典型 Pipe 组合

| set_pipe | wait_pipe | 场景 |
|----------|-----------|------|
| PIPE_MTE1 | PIPE_M | L1 数据加载完成 → 通知 Cube 计算 |
| PIPE_MTE2 | PIPE_M | GM 数据加载完成 → 通知 Cube 计算 |
| PIPE_M | PIPE_MTE3 | Cube 计算完成 → 通知 GM 写回 |
| PIPE_M | PIPE_FIX | Cube 计算完成 → 通知 Fixpipe（[dma_fixpipe.md](dma_fixpipe.md)） |
| PIPE_M | PIPE_V | Cube 计算完成 → 通知 Vector 后处理 |

## IR 示例

```mlir
// 静态 Event ID
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]

// 动态 Event ID（循环中按迭代复用）
%eventId = arith.constant 1 : i64
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE1>, #hivm.pipe<PIPE_M>, %eventId]

// Pipe 屏障
hivm.hir.pipe_barrier [#hivm.pipe<PIPE_M>]
```

### 完整 Load → Compute → Store 模式

```mlir
hivm.hir.load ins(%src_gm : ...) outs(%dst_l1 : ...)
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.mmadL1 ins(...) outs(...)
hivm.hir.set_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE3>, #hivm.event<EVENT_ID1>]
hivm.hir.wait_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE3>, #hivm.event<EVENT_ID1>]
hivm.hir.store ins(...) outs(...)
```

## 约束与选型

1. set_flag 必须先于对应 wait_flag 执行，否则 wait 永久阻塞。
2. pipe_barrier 等整个 Pipe 排空，远比 set/wait 重；只同步特定依赖时优先 set/wait。
3. 静态 Event ID 用于编译时确定的单发同步点；循环复用场景用动态 Event ID（按循环变量计算）。
4. lowering 到硬件指令时，静态/动态 Event ID 分别对应 `SET.FLAG.IMM` / `SET.FLAG.REG` 内建指令（见 [intrin_overview.md](intrin_overview.md)）。

相关：[sync_overview.md](sync_overview.md)、[sync_block_sync.md](sync_block_sync.md)、[sync_sync_injection.md](sync_sync_injection.md)、[attr_enumerations.md](attr_enumerations.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMSynchronizationOps.td#L34-L49](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L34-L49)
- [HIVMSynchronizationOps.td#L51-L66](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L51-L66)
- [HIVMSynchronizationOps.td#L68-L72](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L68-L72)
- [HIVMSynchronizationOps.td#L34-L72](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L34-L72)
- [sync-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/sync-ops.mlir)
