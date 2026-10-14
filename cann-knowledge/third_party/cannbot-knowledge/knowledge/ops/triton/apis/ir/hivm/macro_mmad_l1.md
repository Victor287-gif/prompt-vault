---
type: API
title: HIVM 宏操作 MMAD-L1（hivm.hir.mmadL1）
description: hivm.hir.mmadL1 操作语义：Cube Core 上 L1→L0C 的本地矩阵乘加 C = C + A×B + (可选) channel_bias，支持
  a/b_transpose、HF32 加速、per-channel bias、Split-K 条件初始化（init_condition）与 UnitFlag 同步。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 宏操作 MMAD-L1（hivm.hir.mmadL1）

`hir.mmadL1` 是 Cube Core 上的本地矩阵乘加，对应硬件 mma_tile 指令：从 L1 读 A/B，在 L0C 中执行 `C = C + A × B + (可选) channel_bias`，结果留在 L0C。涉及 MTE1 与 M 两个 Pipeline（`MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">`），属 HIVM 中最基础的矩阵计算单元；Triton `tl.dot` 在 Split-K 场景常映射为它。数据上游是 [dma_nd2nz.md](dma_nd2nz.md)（GM→L1），下游是 [dma_fixpipe.md](dma_fixpipe.md)（L0C→GM/UB/L1）。Cube 流水背景见 [pipeline_execution_model](../architecture/pipeline_execution_model.md)。

## 输入操作数（ins）

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$a` | TensorOrMemref | 是 | 矩阵 A `[M, K]`，rank=2 |
| `$b` | TensorOrMemref | 是 | 矩阵 B `[K, N]`，rank=2 |
| `$init_condition` | I1 | 是 | L0C 清零条件：true 时先清零再累加 |
| `$real_m` / `$real_k` / `$real_n` | Index | 是 | M/K/N 维实际数据大小 |
| `$c` | TensorOrMemref | 是 | 矩阵 C（输出/累加）`[M, N]` |
| `$per_channel_bias` | TensorOrMemref | 否 | Per-channel bias，形状 `[N]` |
| `$sync_related_args` | Variadic\<I64\> | 否 | 同步参数，InjectSync Pass 自动管理 |
| `$unit_flag_cond` | Variadic\<I1\> | 否 | UnitFlag 启用条件（循环依赖场景） |

输出 `outs($result_tensors)` 为 Variadic\<AnyRankedTensor\>。

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `a_transpose` / `b_transpose` | UnitAttr | 加载前转置 A/B（逻辑形状不变，读数方式转置） |
| `enable_HF32` | UnitAttr | FP32 舍入为 HF32（10-bit 尾数）再算，性能翻倍、精度降低，仅对 FP32 有效 |
| `unit_flag_mode` | UnitFlagArrayAttr | 每个输出 Tensor 的 UnitFlag 模式，见 [sync_unit_flag.md](sync_unit_flag.md) |

## IR 示例

### 基本用法（memref 语义）

```mlir
%init = arith.constant 1 : i1
hivm.hir.mmadL1 ins(%ma, %mb, %init, %c256, %c128, %c256 :
                      memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                outs(%mc : memref<256x256xf32>)
```

### Split-K 循环中的条件初始化

```mlir
scf.for %arg0 = %start to %end step %step {
  %init_condition = arith.cmpi eq, %arg0, %start : index
  hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, ... : ...)
                  outs(%mc : memref<256x256xf32>)
}
```

首次迭代 `init_condition=true` 清零 L0C，后续迭代为 false 累加部分和——这是 Split-K 累加语义的关键。

### Tensor 语义

```mlir
%res = hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                             tensor<256x128xf16>, tensor<128x256xf16>, i1, index, index, index)
                       outs(%mc_iter : tensor<256x256xf32>) -> tensor<256x256xf32>
```

## IR 层约束

1. A/B/C rank 必须为 2（batchMmadL1 为 3）。
2. `CubeCoreTypeTrait`：只能在 Cube Core 执行；`NoMaxRankTrait`：无 rank 上限。
3. `per_channel_bias` 形状必须为 `[N]`，与 B 的列维匹配。
4. 实现 `OpLayoutInterface.getOperandsTargetFractalLayout`，并提供 `getOperandALayout/BLayout/CLayout/BiasLayout()`（均返回 `FailureOr<DataLayoutAttr>`），Fractal 布局参数见 [attr_parameterized_attrs.md](attr_parameterized_attrs.md)。
5. `getOpName()` 返回 `"mma_tile"`；`getMatmulBiasMode()` / `shouldDecomposeBiasByElementAdd()` 用于 bias 处理决策。

## 选型要点

- mmadL1 需要用户（或前端 lowering）手动管理 L1 数据搬运与同步，适合 Split-K 等精细控制场景；端到端大矩阵乘优先用 [macro_matmul.md](macro_matmul.md)（编译器自动管理 L1/L0 与同步）。
- HF32 适用于对精度不敏感但要 FP32 级吞吐的场景。

相关：[macro_overview.md](macro_overview.md)、[macro_batch_mmad_l1.md](macro_batch_mmad_l1.md)、[dma_fixpipe.md](dma_fixpipe.md)、[sync_unit_flag.md](sync_unit_flag.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMMacroOps.td#L152-L172](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L152-L172)
- [HIVMMacroOps.td#L49-L150](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L49-L150)
- [HIVMMacroOps.td#L49-L172](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L49-L172)
- [ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/ops.mlir)
