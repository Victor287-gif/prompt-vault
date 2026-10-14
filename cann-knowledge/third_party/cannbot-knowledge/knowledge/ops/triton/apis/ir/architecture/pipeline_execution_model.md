---
type: API
title: Pipeline 执行模型（PIPE 枚举、Pipe Trait、set_flag/wait_flag/pipe_barrier 同步）
description: AI Core 多 Pipeline 并行执行模型：PIPE 枚举全表、操作→Pipe 的 Trait 绑定（OpPipeTrait/MacroOpPipeTrait）、跨
  Pipeline 同步原语（set_flag/wait_flag/pipe_barrier）与 Event ID/SyncBlock 机制。
tags: [task.lookup, topic.architecture, topic.compilation, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# Pipeline 执行模型

AI Core 内部多 Pipeline 并行：每个 Pipeline 对应一个硬件执行单元，不同 Pipeline 可并行工作，同一 Pipeline 内操作严格按程序顺序执行。HIVM IR 通过 `PIPE` 枚举和 TableGen Trait 把每个操作绑定到 Pipeline；`InjectSync` Pass 据此做同步分析，在跨 Pipeline 数据依赖处插入 `set_flag`/`wait_flag`/`pipe_barrier`。

## PIPE 枚举

| 值 | 硬件单元 | 数据流 | 典型操作 |
|----|----------|--------|----------|
| PIPE_S(0) | Scalar | 标量计算/控制流 | 循环控制 |
| PIPE_V(1) | Vector (AIV) | UB→UB | `hivm.vadd`/`vmul`/`vcast` 等全部向量 op |
| PIPE_M(2) | Cube (AIC) | L0A/L0B→L0C | `hivm.mmadL1` 的矩阵乘法部分 |
| PIPE_MTE1(3) | MTE1 DMA | L1→L0A/L0B/BT | `hivm.l12ub`、mmadL1 加载部分 |
| PIPE_MTE2(4) | MTE2 DMA | GM↔L1/UB（双向） | `hivm.load`、`hivm.nd2nz` |
| PIPE_MTE3(5) | MTE3 DMA | UB→GM/L1（单向） | `hivm.store`、`hivm.nz2nd` |
| PIPE_ALL(6) | 全部 | 同步用统称 | barrier |
| PIPE_MTE4(7) / PIPE_MTE5(8) | DMA | 预留扩展 | 当前无 op 使用 |
| PIPE_V2(9) | 第二 Vector | UB→UB | 第二向量单元 |
| PIPE_FIX(10) | FixPipe | L0C→GM/L1/UB | `hivm.fixpipe` |
| VIRTUAL_PIPE_MTE2_L1A/L1B(11/12) | 虚拟 | 区分 MTE2 到 L1 的 A/B 矩阵两路数据流 | 编译器内部细粒度同步 |
| PIPE_NUM(13) / PIPE_UNASSIGNED(99) | — | 计数 / 未分配 | — |

## 操作 → Pipe 的 Trait 绑定

三种 Trait（定义于 HIVMTraits.td）：

- `SinglePipeOpTrait`：标记操作只有单一 Pipeline。
- `OpPipeTrait<"PIPE::PIPE_X">`：把操作静态绑定到单一 Pipeline（继承 SinglePipeOpTrait）。
- `MacroOpPipeTrait<"PIPE::IN, PIPE::OUT">`：标注宏操作的输入/输出 Pipeline（继承 MacroOpTrait）；InPipe=首个子操作所在 Pipe，OutPipe=末个子操作所在 Pipe，同步分析用它们确定宏操作与外部的依赖。

统一的 `OpPipeInterface` 提供 `isSinglePipeOp()` / `isMacroOp()` / `getPipe()` / `getInPipe()` / `getOutPipe()`。

典型绑定：

| 操作 | Pipe | 方式 |
|------|------|------|
| `hivm.load` / `hivm.nd2nz` | MTE2 | OpPipeTrait |
| `hivm.store` / `hivm.nz2nd` | MTE3 | OpPipeTrait |
| `hivm.fixpipe` | FIX | OpPipeTrait（核心类型 CUBE） |
| `hivm.l12ub` | MTE1 | OpPipeTrait |
| `hivm.copy` | 按源/目标地址空间动态推断（getPipe）：UB→UB=V、L0C→GM=FIX、GM→L1=MTE2、UB→L1=MTE3 | 无静态 Trait |
| 全部向量 op（`HIVM_VectorOp` 基类） | PIPE_V + VectorCoreType + SinglePipeOpTrait | 基类 Trait |
| `hivm.mmadL1` / `batchMmadL1` | InPipe=MTE1, OutPipe=M | MacroOpPipeTrait |
| `hivm.matmul` / `mix_matmul` / `mix_group_matmul` | InPipe=MTE2, OutPipe=MTE3 | MacroOpPipeTrait |

## 同步原语

```mlir
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>, EVENT_ID0]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_V>, EVENT_ID0]
hivm.hir.pipe_barrier [#hivm.pipe<PIPE_V>]
```

- `set_flag[set_pipe, wait_pipe, event]`：set_pipe（数据生产方）完成后通知 wait_pipe（消费方）。
- `wait_flag`：消费方执行前等待生产方事件；参数含义与 set_flag 对称。
- `pipe_barrier[pipe]`：同一 Pipeline 内的屏障，确保该 Pipe 之前所有操作完成。

典型模式：`MTE2 加载 → set_flag[MTE2, V, e0] → V 计算前 wait_flag[MTE2, V, e0]`。

常见跨 Pipeline 依赖对：MTE2→V（GM 加载后向量算）、V→MTE3（算完写回）、MTE1→M（L0 加载后乘）、M→FIX（乘完输出）、FIX→V（950 L0C 直通 UB 后再算）。

## Event ID / Unit Flag / SyncBlock

- **Event ID**：硬件提供 8 个（EVENT_ID0~7），由 `SyncEventIdAllocation` Pass 分配复用。
- **UNIT_FLAG**：条件同步，用于循环中依赖首次迭代才能确定同步的场景；取值 DISABLED(0) / RESERVED(1) / ENABLED_WITHOUT_UPDATE(2) / ENABLED_WITH_UPDATE(3)。
- **SyncBlock**：跨 AI Core 同步，模式有 ALL_CUBE / ALL_VECTOR / ALL_SUB_VECTOR / BARRIER_CUBE / BARRIER_VECTOR / ALL；BARRIER_* 降低为 `barrier.pipe_all`。

## 紧耦合缓冲区的 Pipeline 组合

| 路径 | 数据流 | Pipeline 组合 |
|------|--------|----------------|
| InsertCVTightCoupledBuffer（950） | L0C→UB | FIX + V |
| InsertCVDataMovement（950 布局优化） | L0C→UB | FIX + V |
| InsertLoadStoreForMixCV（非 950） | L0C→GM→UB | FIX + MTE2 + V |

执行约束小结：跨 Pipeline 并行、Pipeline 内顺序、跨 Pipe 数据依赖必须 set_flag/wait_flag 配对。地址空间与 Pipeline 的对应见 [memory_hierarchy.md](memory_hierarchy.md)；架构差异（Reg-based 跳过 PIPE_V barrier 等）见 [npu_hardware_overview.md](npu_hardware_overview.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
- [HIVMAttrs.td#L203-L236](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L236)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMDMAOps.cpp#L616-L622](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L616-L622)
- [HIVMVectorOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td)
- [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)
- [HIVMTraits.td#L157-L173](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td#L157-L173)
- [OpPipeInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.td)
- [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)
- [HIVMAttrs.td#L479-L498](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L479-L498)
- [HIVMAttrs.td#L512-L534](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L512-L534)
- [HIVMAttrs.td#L540-L563](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L540-L563)
