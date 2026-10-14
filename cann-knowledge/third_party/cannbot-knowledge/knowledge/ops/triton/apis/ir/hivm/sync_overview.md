---
type: API
title: HIVM 同步体系总览（Pipe Sync / Block Sync / UnitFlag）
description: HIVM 三层同步体系：管道同步（set_flag/wait_flag/pipe_barrier，Event ID 机制）、跨核同步（sync_block
  系，FFTS 机制）、UnitFlag 条件同步，以及 InjectSync / GraphSyncSolver 自动同步注入。
tags: [task.lookup, topic.compilation, topic.implementation.synchronization, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]

---

# HIVM 同步体系总览

NPU 采用多 Pipeline 并行架构（MTE1/MTE2/MTE3/M/V/FIX，见 [pipeline_execution_model](../architecture/pipeline_execution_model.md)），不同 Pipe 上的操作存在数据依赖时必须显式同步。HIVM 同步体系分三层，冗余或缺失同步分别对应 IR 性能分析中的"同步开销"瓶颈与正确性问题（见 [../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md) 同步开销条目）。

```
┌ 跨核同步 (Block Sync) ── sync_block / sync_block_set / sync_block_wait
│                          create_sync_block_lock / sync_block_lock / unlock
├ 管道同步 (Pipe Sync) ─── set_flag / wait_flag / pipe_barrier
└ UnitFlag 同步 ───────── unit_flag_cond / unit_flag_mode（嵌入宏操作）
```

## 管道同步（最细粒度）

同一 Block 内不同 Pipeline 间的协调，基于 **Event ID** 机制：每个 Pipe 有 8 个 Event ID（EVENT_ID0~7），`set_flag`/`wait_flag` 以 (set_pipe, wait_pipe, event_id) 三元组配对。Event ID 可静态（`#hivm.event<EVENT_IDx>`）或动态（i64 运行时值）。详见 [sync_pipe_sync.md](sync_pipe_sync.md)。

| 操作 | 助记符 | 说明 |
|------|--------|------|
| SetFlagOp | `hir.set_flag` | 在 set_pipe 上设置 Event Flag |
| WaitFlagOp | `hir.wait_flag` | 在 wait_pipe 上等待 Event Flag |
| PipeBarrierOp | `hir.pipe_barrier` | 等待该 Pipe 所有先前操作完成（重量级） |

## 跨核同步（FFTS）

不同 Block 之间的协调，基于 **FFTS（Fast Flag Transfer System）** 硬件机制：每个 Block 有 FFTS 基地址，FFTS 收集特定 flag_id 后回设给 Block 组完成同步。Ascend910B 及以上平台必须用 `hir.set_ffts_base_addr` 设置基地址。详见 [sync_block_sync.md](sync_block_sync.md)。

| 操作 | 助记符 | 说明 |
|------|--------|------|
| SyncBlockOp | `hir.sync_block` | 高层跨核同步，按 SyncBlockMode 封装 |
| SyncBlockSetOp / SyncBlockWaitOp | `hir.sync_block_set` / `hir.sync_block_wait` | 低层 set/wait 接口 |
| CreateSyncBlockLockOp / SyncBlockLockOp / SyncBlockUnlockOp | `hir.create_sync_block_lock` / `hir.sync_block_lock` / `hir.sync_block_unlock` | 跨核锁原语 |

### SyncBlockMode

| 模式 | 值 | 说明 |
|------|---|------|
| ALL_CUBE | 0 | 所有 Cube Core 同步 |
| ALL_VECTOR | 1 | 所有 Vector Core 同步 |
| ALL_SUB_VECTOR | 2 | 所有 Sub-Vector 同步 |
| BARRIER_CUBE | 3 | Cube-Cube 屏障 |
| BARRIER_VECTOR | 4 | Vector-Vector 屏障 |
| ALL | 5 | 所有 AIC/AIV 同步 |

## UnitFlag 同步

嵌入宏操作（mmadL1 等）内的条件同步，处理循环"至少执行一次"依赖：循环可能零次执行时仍保持 set/wait 配对。四模式 DISABLED / RESERVED / ENABLED_WITHOUT_UPDATE / ENABLED_WITH_UPDATE。详见 [sync_unit_flag.md](sync_unit_flag.md)。

## 自动同步注入

用户 IR 通常不含同步操作，由编译器 Pass 自动插入（见 [sync_sync_injection.md](sync_sync_injection.md)）：

1. **InjectSync Pass**：启发式规则，按操作顺序分析依赖插入 set_flag/wait_flag（BARRIERALL 模式可全插 pipe_barrier 用于调试）。
2. **GraphSyncSolver Pass**：构建操作依赖图后求解最优同步，Event ID 利用更优、冗余同步更少，但编译更慢。

宏操作（[macro_overview.md](macro_overview.md)）因跨 Pipe 特性是同步注入的重点对象：InjectSync 在其前后自动插入跨 Pipe set/wait，GraphSyncSolver 将其建模为多节点依赖图。

相关：[sync_pipe_sync.md](sync_pipe_sync.md)、[sync_block_sync.md](sync_block_sync.md)、[sync_unit_flag.md](sync_unit_flag.md)、[sync_sync_injection.md](sync_sync_injection.md)、[intrin_overview.md](intrin_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)
- [sync-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/sync-ops.mlir)
