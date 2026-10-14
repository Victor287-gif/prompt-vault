---
type: API
title: HIVM 同步注入 Pass（InjectSync / GraphSyncSolver / InjectBlockSync）
description: HIVM 自动同步注入机制：hivm-inject-sync（启发式，sync-mode NORMAL/BARRIERALL，enable-unit-flag，assume-alive-loops）与
  hivm-graph-sync-solver（图求解，Event ID 分配更优）的选项、变换效果、对比与 Pipeline 选择逻辑。
tags: [task.lookup, topic.compilation, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 同步注入 Pass（InjectSync / GraphSyncSolver）

用户编写的 HIVM IR 通常不含同步操作，编译器根据操作间数据依赖自动插入 `set_flag`/`wait_flag`（操作语义见 [sync_pipe_sync.md](sync_pipe_sync.md)）。注入质量直接决定"同步开销"瓶颈（多余 barrier、串行化），分析优化入口见 [../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md) 同步开销条目。

## InjectSync Pass（`hivm-inject-sync`，func 级）

启发式规则：按操作顺序分析数据依赖并插入同步。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `sync-mode` | SyncMode | NORMAL | NORMAL=按依赖插 set/wait；BARRIERALL=每个操作前后插 pipe_barrier（**仅调试**，性能极差但可排除同步问题） |
| `enable-unit-flag` | bool | false | 启用 [UnitFlag](sync_unit_flag.md)，处理循环可能零次执行的同步场景 |
| `assume-alive-loops` | bool | false | 假设所有循环至少执行一次，简化分析、省掉 UnitFlag；假设不成立会同步错误 |

### 变换效果（NORMAL 模式）

```mlir
// 变换前：load → mmadL1 → fixpipe（无同步）
// 变换后：
hivm.hir.load ins(%src : ...) outs(%dst_l1 : ...)
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.mmadL1 ins(...) outs(...)
hivm.hir.set_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_FIX>, #hivm.event<EVENT_ID1>]
hivm.hir.wait_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_FIX>, #hivm.event<EVENT_ID1>]
hivm.hir.fixpipe ins(...) outs(...)
```

BARRIERALL 模式则在每步后插 `hivm.hir.pipe_barrier [PIPE_MTE2] / [PIPE_M] / [PIPE_FIX]` 等，用于二分定位同步 bug。

## GraphSyncSolver Pass（`hivm-graph-sync-solver`，func 级）

基于图的同步求解：

1. **IR 转换**：MLIR → 内部同步图表示（SyncSolverIR）。
2. **图构建**：节点为操作、边为数据依赖。
3. **求解**：图上求解最优同步方案（Event ID 分配 + 同步点选择）。
4. **代码生成**：结果转回 MLIR，插入 set/wait。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enable-unit-flag` | bool | false | 启用 UnitFlag |
| `enable-tester-mode` | bool | false | 启用同步测试模式 |
| `sync-tester-options` | list\<i64\> | [] | num_runs, init_seed, num_ops, num_ptrs, enable-multibuffer |

## InjectSync vs GraphSyncSolver

| 特性 | InjectSync | GraphSyncSolver |
|------|-----------|-----------------|
| 算法 | 启发式规则 | 图求解 |
| Event ID 使用 | 可能较多 | 更优化（8 个/Pipe 的预算内复用更好） |
| 同步粒度 | 操作级 | 可更细 |
| 编译速度 | 较快 | 较慢 |
| 同步质量 | 基本正确 | 冗余同步更少（同步开销更低） |
| 调试支持 | BARRIERALL 模式 | Tester 模式 |

**Pipeline 选择逻辑**（HIVMPipelines）：`enableHIVMGraphSyncSolver=true` 且 `enableHIVMInjectBarrierAllSync=false` → GraphSyncSolver；否则 InjectSync。默认 InjectSync。

## InjectBlockSync Pass（`hivm-inject-block-sync`）

跨 Block 同步注入（操作语义见 [sync_block_sync.md](sync_block_sync.md)）：

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `block-all-sync` | bool | false | 注入全部 Block 同步 |
| `assume-alive-loops` | bool | false | 假设循环至少执行一次 |

## 实践要点

- 怀疑同步 bug 时先用 InjectSync BARRIERALL 验证正确性，再切回 NORMAL/GraphSyncSolver 定位多余或缺失的同步点。
- 性能侧发现同步开销瓶颈时，启用 GraphSyncSolver 是编译器侧的第一手段；其余优化（减少依赖、调整分块）见 [../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md)。

相关：[sync_overview.md](sync_overview.md)、[sync_pipe_sync.md](sync_pipe_sync.md)、[sync_unit_flag.md](sync_unit_flag.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td#L163-L183](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L163-L183)
- [Passes.td#L195-L213](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L195-L213)
- [HIVMPipelines.cpp#L81-L98](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp#L81-L98)
- [Passes.td#L163-L213](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L163-L213)
- [InjectSync.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/Transforms/InjectSync/InjectSync.cpp)
- [GraphSyncSolver.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/GraphSyncSolver.cpp)
- [HIVMPipelines.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp)
