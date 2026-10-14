---
type: API
title: HIVM Trait 定义（Rank / Elementwise / Pipe / CoreType 四类）
description: HIVM Trait 体系：Rank 系（SameRank/StaticMaxRank/InferMaxRank/NoMaxRank 及适用操作）、ElementwiseNaryOpTrait、BroadcastableOTF/TransposableOTF
  内联语义与约束、SinglePipeOpTrait/OpPipeTrait 与 MacroOpTrait/MacroOpPipeTrait、CoreTypeTrait
  三变体及依赖关系图。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM Trait 定义

Trait 是附加在操作上的行为约束标记，用于 Pass 中的操作分类与验证。

## Rank 相关 Trait

- **HIVMOpSameOperandsAndResultRank**：所有操作数与结果 rank 相同（临时缓冲区除外）；依赖 HIVMStructuredOpInterface；适用大多数 Elementwise 操作。
- **OpLibraryMaxRankTrait\<MaxRank\>**：库函数最大 rank。MaxRank>0 静态已知；=0 静态未知（需推断）；=-1 无限制。
- **StaticMaxRankTrait\<MaxRank\>**（继承 OpLibraryMaxRankTrait）：debug=4D、embedding_gather=3D、indirect_load/indirect_store/gatherT/scatterT/index_put=5D。
- **InferMaxRankTrait** = OpLibraryMaxRankTrait\<0\>。
- **NoMaxRankTrait** = OpLibraryMaxRankTrait\<-1\>：mmadL1、matmul、mix_matmul、mix_group_matmul、fixpipe 等。

## Elementwise Trait

**ElementwiseNaryOpTrait\<N\>**：N 输入逐元操作产单结果。约束：实现 DPS 接口；输入数 = N；shaped 操作数与结果 rank 相同；buffer 语义时最终维 stride 相等。依赖 HIVMStructuredOpInterface + HIVMOpSameOperandsAndResultRank。

## 广播与转置 Trait（内联 on-the-fly）

### BroadcastableOTF

```
for i            // <- broadcast dim
  for j
    dst[i, j] = some_op(src1[0, j], ..., srcN[0, j])
```

约束：DPS 接口；必须有 `DenseI64ArrayAttr` 名 `"broadcast"`；broadcast 维唯一且在 `[0, rank(dst))`；broadcast 维上 `dim(src_i,d)=1 或 =dim(dst,d)`；非 broadcast 维上 `dim(src_i,d)=dim(dst,d)`。

### TransposableOTF

```
for i            // transpose = [1, 0]
  for j
    dst[i, j] = some_op(src1[j, i], ..., srcN[j, i])
```

约束：DPS 接口；`DenseI64ArrayAttr` 名 `"transpose"`；transpose 是 `range(rank(dst))` 的排列；`transpose[rank(dst)-1] = rank(dst)-1`（末维不动）；`dim(dst,d) = dim(src_i, transpose[d])`。

## Pipe 相关 Trait

- **SinglePipeOpTrait**：单 Pipe 操作标记。
- **OpPipeTrait\<Pipe\>**（依赖 SinglePipeOpTrait）：声明单个执行 Pipe。embedding_gather / indirect_load / indirect_store / gatherT / scatterT / index_put = PIPE_V；custom 由属性指定。
- **MacroOpTrait**：跨 Pipeline 宏操作标记。
- **MacroOpPipeTrait\<InOutPipes\>**（依赖 MacroOpTrait）：mmadL1/batchMmadL1 = `PIPE_MTE1, PIPE_M`；matmul/mix_matmul/mix_group_matmul = `PIPE_MTE2, PIPE_MTE3`（详见 [macro_overview.md](macro_overview.md)）。

## Core Type Trait（CoreTypeTrait\<CoreType\> 三变体）

| Trait | CoreType | 适用操作 |
|-------|----------|---------|
| VectorCoreTypeTrait | VECTOR | 大多数 Vector 操作 |
| CubeCoreTypeTrait | CUBE | mmadL1、batchMmadL1 |
| CubeVectorCoreTypeTrait | CUBE_OR_VECTOR | get_block_idx、pointer_cast、set_ffts_base_addr |

## 其他 Trait

- **NoLibraryFunctionTrait**：无预定义库函数实现（batchMmadL1，见 [macro_batch_mmad_l1.md](macro_batch_mmad_l1.md)）。
- **CommutativeOpTrait**：输入可交换。
- **VectorOnlyTrait\<idx\> / ScalarOnlyTrait\<idx\>**：操作数仅 Vector / 标量。
- **OperElemTypeConstraints\<indices, types\>**：元素类型约束。
- **UniformReassociationFlattenTrait**：可统一展平维度（依赖 HIVMStructuredOpInterface）。
- **CollapsibleConsecutiveTargetDimsTrait**：展平时保留目标维独立性与 rank（依赖上述两者）。

## Trait 依赖关系

```
HIVMStructuredOpInterface
├── HIVMOpSameOperandsAndResultRank
│   ├── ElementwiseNaryOpTrait<N>
│   ├── BroadcastableOTF
│   └── TransposableOTF
├── UniformReassociationFlattenTrait
│   └── CollapsibleConsecutiveTargetDimsTrait
└── OpLibraryMaxRankTrait<MaxRank>
    ├── StaticMaxRankTrait<MaxRank>
    ├── InferMaxRankTrait
    └── NoMaxRankTrait

SinglePipeOpTrait
└── OpPipeTrait<Pipe>

MacroOpTrait
└── MacroOpPipeTrait<InOutPipes>
```

相关：[attr_overview.md](attr_overview.md)、[attr_interfaces.md](attr_interfaces.md)、[attr_type_system.md](attr_type_system.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
