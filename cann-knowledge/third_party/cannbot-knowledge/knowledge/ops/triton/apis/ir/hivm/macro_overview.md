---
type: API
title: HIVM 宏操作总览（MacroOpTrait / MacroOpPipeTrait）
description: HIVM 宏操作体系：MacroOpTrait 与 MacroOpPipeTrait 标记跨 Pipeline 复合计算，本地 MMAD（mmadL1/batchMmadL1，MTE1→M）与全局
  MMAD（matmul/mix_matmul/mix_group_matmul，MTE2→MTE3）两大类，UnitFlag 同步与 InjectSync 处理。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 宏操作总览

宏操作（Macro Operations）是 HIVM 中表达**跨 Pipeline 复合计算**的抽象：一次操作涉及多个硬件 Pipeline 之间的数据流动与同步，对应 Cube Core 与其他单元（MTE/Vector）的协作。Triton 的 `tl.dot` / `torch.matmul` 在编译时通常被映射为宏操作。Cube 流水的执行背景见 [pipeline_execution_model](../architecture/pipeline_execution_model.md)。

## 核心 Trait

- **MacroOpTrait**：标识跨 Pipeline 宏操作。同步注入（InjectSync）、Pipe 分配、GraphSyncSolver 等 Pass 据此特殊处理。
- **MacroOpPipeTrait\<InOutPipes\>**：参数化 Trait，声明输入/输出 Pipe 组合，格式 `MacroOpPipeTrait<"PIPE::PIPE_IN, PIPE::PIPE_OUT">`。区别于单 Pipe 操作的 `SinglePipeOpTrait + OpPipeTrait`。

| 操作 | MacroOpPipeTrait 参数 | 含义 |
|------|----------------------|------|
| mmadL1 / batchMmadL1 | `PIPE_MTE1, PIPE_M` | L1 加载（MTE1）→ Cube 计算（M） |
| matmul | `PIPE_MTE2, PIPE_MTE3` | GM 加载（MTE2）→ 结果写回 GM（MTE3） |
| mix_matmul | `PIPE_MTE2, PIPE_MTE3` | 同 matmul，附加 Vector 后处理 |
| mix_group_matmul | `PIPE_MTE2, PIPE_MTE3` | 同 matmul，附加分组（MoE）与 Vector 后处理 |

## 类层次

```
HIVM_MacroOp (基类)
├── HIVM_LocalMmadOp（本地 MMAD：数据在 L1→L0C 片上层次流动）
│   ├── MmadL1Op         -- hir.mmadL1
│   └── BatchMmadL1Op    -- hir.batchMmadL1
└── HIVM_GlobalMmadOp（全局 MMAD：数据直接来自 GM）
    ├── MatmulOp          -- hir.matmul
    ├── MixMatmulOp       -- hir.mix_matmul
    └── MixGroupMatmulOp  -- hir.mix_group_matmul
```

## UnitFlag 同步

宏操作支持 UnitFlag 同步模式，处理循环"至少执行一次"的依赖场景（如 Split-K 循环次数可能为 0 时保持 set/wait 配对）。四种模式：DISABLED(0) / RESERVED(1) / ENABLED_WITHOUT_UPDATE(2) / ENABLED_WITH_UPDATE(3)。本地 MMAD 通过 `unit_flag_cond`（i1 条件）与 `unit_flag_mode`（UnitFlagArrayAttr）启用。详见 [sync_unit_flag.md](sync_unit_flag.md)。

## 宏操作 vs 单 Pipe 操作

| 特性 | 单 Pipe 操作 | 宏操作 |
|------|------------|--------|
| Pipe 数量 | 单个 | 多个 |
| 标记 Trait | SinglePipeOpTrait + OpPipeTrait | MacroOpTrait + MacroOpPipeTrait |
| 同步需求 | Pipe 内同步 | 跨 Pipe 同步（InjectSync 自动注入 set_flag/wait_flag） |
| 典型操作 | load, store, vadd | mmadL1, matmul |
| DestinationStyleOpInterface | 支持 | 支持 |

## 操作速查

| 操作 | 助记符 | 数据通路 | 详细卡 |
|------|--------|---------|--------|
| MmadL1Op | `hir.mmadL1` | L1→L0C 本地乘加 | [macro_mmad_l1.md](macro_mmad_l1.md) |
| BatchMmadL1Op | `hir.batchMmadL1` | 批量 L1→L0C | [macro_batch_mmad_l1.md](macro_batch_mmad_l1.md) |
| MatmulOp | `hir.matmul` | GM→GM 全局矩阵乘 | [macro_matmul.md](macro_matmul.md) |
| MixMatmulOp | `hir.mix_matmul` | GM→GM + Vector 后处理 | [macro_mix_matmul.md](macro_mix_matmul.md) |
| MixGroupMatmulOp | `hir.mix_group_matmul` | MoE 分组矩阵乘 | [macro_mix_group_matmul.md](macro_mix_group_matmul.md) |

相关：[sync_overview.md](sync_overview.md)、[attr_enumerations.md](attr_enumerations.md)、[attr_traits.md](attr_traits.md)、[overview.md](overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)
