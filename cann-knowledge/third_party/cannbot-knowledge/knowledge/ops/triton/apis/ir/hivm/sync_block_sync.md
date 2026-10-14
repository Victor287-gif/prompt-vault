---
type: API
title: HIVM 跨核同步操作（sync_block / sync_block_set / sync_block_wait / 锁原语）
description: hivm.hir.sync_block 系跨核同步语义：基于 FFTS 硬件机制，SyncBlockMode 六种模式及必需参数、sync_block_set/wait
  低层配对、SyncBlockInstrMode 三级作用域、create_sync_block_lock/lock/unlock 互斥锁原语。
tags: [task.lookup, topic.compilation, topic.implementation.synchronization, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]

---

# HIVM 跨核同步操作（sync_block 系）

跨核同步协调**不同 Block** 之间的执行（大矩阵分块、AllReduce 等多 Block 协作场景），基于 **FFTS（Fast Flag Transfer System）** 硬件机制：FFTS 收集特定 flag_id 后回设给 Block 组中的 Block 完成同步。Ascend910B 及以上平台 `ffts_base_addr` 必须设置（通常函数入口 `hir.set_ffts_base_addr`）。Block 内 Pipe 级同步见 [sync_pipe_sync.md](sync_pipe_sync.md)。

## 操作与参数

### SyncBlockOp（高层封装）

```mlir
hivm.hir.sync_block[#hivm.sync_block_mode<ALL>, 1 : i16]
          ffts_base_addr = %ffts_base_addr
          tcube_pipe = #hivm.pipe<PIPE_FIX>
          tvector_pipe = #hivm.pipe<PIPE_MTE3>
```

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `sync_block_mode` | HIVM_SyncBlockModeAttr | 是 | 同步模式（下表） |
| `flag_id` | IntegerAttr | 否 | 静态 Flag ID |
| `ffts_base_addr` | I64 | 否 | FFTS 基地址（910B+ 必须） |
| `tcube_pipe` / `tvector_pipe` | HIVM_PipeAttr | 否 | Cube/Vector Core 等待的 Pipe |

### SyncBlockSetOp / SyncBlockWaitOp（低层接口）

| 参数 | 说明 |
|------|------|
| `tcore_type` | Core 类型（CUBE/VECTOR，见 [attr_enumerations.md](attr_enumerations.md) TCoreType） |
| `tpipe` / `pipe` | 目标 Core 的 Pipe / 当前操作的 Pipe |
| `static_flag_id` / `dynamic_flag_id` | 静态 / 动态 Flag ID（二选一） |
| `ffts_base_addr` | FFTS 基地址（仅 SetOp） |
| `tsync_instr_mode` | 同步指令模式，默认 INTRA_BLOCK_SYNCHRONIZATION |

set/wait 必须以相同 (tcore_type, tpipe, pipe, flag_id) 组合配对。

### SyncBlockMode 与必需参数

| 模式 | 值 | 说明 | 必需参数 |
|------|---|------|---------|
| ALL_CUBE | 0 | 所有 Cube Core 同步到同一点 | `tcube_pipe` |
| ALL_VECTOR | 1 | 所有 Vector Core 同步到同一点 | `tvector_pipe` |
| ALL_SUB_VECTOR | 2 | 所有 Sub-Vector 同步 | — |
| BARRIER_CUBE | 3 | Cube-Cube 屏障，lowering 为 barrier.pipe_all，仅复制到 AIC kernel | — |
| BARRIER_VECTOR | 4 | Vector-Vector 屏障，lowering 为 barrier.pipe_all，仅复制到 AIV kernel | — |
| ALL | 5 | 所有 AIC/AIV 同步到同一点 | `tvector_pipe` |

### SyncBlockInstrMode

| 模式 | 值 | 说明 |
|------|---|------|
| INTER_BLOCK_SYNCHRONIZATION | 0 | 跨 Block |
| INTER_SUBBLOCK_SYNCHRONIZATION | 1 | 跨 Sub-Block |
| INTRA_BLOCK_SYNCHRONIZATION | 2 | Block 内（默认） |

## 锁原语

```mlir
%lock = hivm.hir.create_sync_block_lock() : memref<1xi64>
hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
// ... 临界区 ...
hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
```

- `create_sync_block_lock`：分配锁内存（也可 `from %arg : from memref<?xi8> to memref<1xi64>` 从外部内存创建）。
- `sync_block_lock`：阻塞直到 `lock_var == block_idx`（按 Block 序号轮转获取）。
- `sync_block_unlock`：`lock_var` 递增并释放。
- 用途：多 Block 互斥访问同一 GM 区域等共享资源场景。

## IR 层约束

1. `sync_block` 只能在数据已搬运到 GM 后使用。
2. 910B 及以上平台跨 Block 同步必须设 `ffts_base_addr`。
3. 高层 `sync_block` 自动处理 set/wait 配对，优先使用；低层 set/wait 需手动配对。
4. lowering 对应 `SET.CROSS.CORE` / `WAIT.FLAG.DEV.REG`（跨核）与 `SET.INTRA.BLOCKI.mode` / `WAIT.INTRA.BLOCKI.mode`（核内立即数模式）内建指令（见 [intrin_overview.md](intrin_overview.md)）。

相关：[sync_overview.md](sync_overview.md)、[sync_pipe_sync.md](sync_pipe_sync.md)、[sync_sync_injection.md](sync_sync_injection.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMSynchronizationOps.td#L78-L118](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L78-L118)
- [HIVMSynchronizationOps.td#L120-L148](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L120-L148)
- [HIVMSynchronizationOps.td#L150-L175](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L150-L175)
- [HIVMSynchronizationOps.td#L78-L244](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td#L78-L244)
- [sync-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/sync-ops.mlir)
