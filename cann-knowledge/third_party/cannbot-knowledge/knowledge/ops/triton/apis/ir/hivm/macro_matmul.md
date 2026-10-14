---
type: API
title: HIVM 宏操作 Matmul（hivm.hir.matmul）
description: hivm.hir.matmul 操作语义：GM→GM 全局矩阵乘 C = descale × (A×B + bias)，编译器自动管理 L1/L0
  搬运与同步，支持反量化 descale（PerChannel/PerTensor）、bias、转置、Swizzle 调度与 L1/L0 分块参数。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 宏操作 Matmul（hivm.hir.matmul）

`hir.matmul` 是全局矩阵乘：直接从 GM 读 A/B、结果写回 GM，计算语义 `C = A × B` 或 `C = descale × (A × B + bias)`（有 bias/descale 时）。涉及 MTE2（GM 加载）与 MTE3（GM 写回）两个 Pipeline，**L1/L0 层次的数据搬运与同步由编译器自动管理**——用户只给 GM 地址和 Tiling 参数。Triton `tl.dot` 在非 Split-K 场景通常映射为它；需要手动控制数据流时用 [mmadL1](macro_mmad_l1.md)。

## 参数

### 输入/输出操作数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$a` / `$b` | AnyShaped | 是 | 矩阵 A `m×k`、B `k×n` |
| `$c` | AnyShaped | 是 | 输出矩阵 C `m×n` |
| `$tilingParams` | AnyShaped | 否 | Tiling 参数 |
| `$bias` | AnyShaped | 否 | Bias 向量，形状 `[n]` |
| `$descale` | AnyShaped | 否 | 反量化缩放因子，形状随 descaleMode |
| `$result`（outs） | Variadic\<AnyRankedTensor\> | — | 结果 Tensor |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `aTranspose` / `bTranspose` | UnitAttr | A/B 转置加载 |
| `descaleMode` | HIVM_DescaleModeAttr | 反量化模式（见下表） |

### I64 操作数（Tiling / 调度参数）

| 参数 | 类型 | 说明 |
|------|------|------|
| `blockSizes` | Variadic\<I64\> | M/N/K 在 L1 层次的分块大小（通常 3 个值） |
| `processSizes` | Variadic\<I64\> | M/N/K 在 L0 层次的分块大小（通常 3 个值） |
| `swizzleOffset` | I64 | Swizzle 调度的连续块起始编号 |
| `swizzleDirection` | I64 | Swizzle 遍历方向 |
| `epiloguePTiles` | I64 | Epilogue 一次处理的 P tile 数量 |

### DescaleMode

| 模式 | 值 | descale 形状 | 说明 |
|------|---|-------------|------|
| DescaleNull | 0 | 无 | 不反量化 |
| DescalePerChannel | 1 | `[n]` | 按 Channel 反量化 |
| DescalePerTensor | 2 | `[1]` | 按 Tensor 反量化 |

枚举详情见 [attr_enumerations.md](attr_enumerations.md)。

## IR 示例

```mlir
// 基本矩阵乘
hivm.hir.matmul
   ins(%A_gm, %B_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                      memref<16x16xf16, #hivm.address_space<gm>>)
   outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
   descale_mode = #hivm.descale_mode<DescaleNull>

// Per-Channel Descale + Bias
hivm.hir.matmul
   ins(%A_gm, %B_gm : ...)
   outs(%res_gm : ...)
   bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
   descale = %descale_gm : memref<16xf16, #hivm.address_space<gm>>
   descale_mode = #hivm.descale_mode<DescalePerChannel>
```

## IR 层约束

1. Core Type 由 `HIVMInferCoreTypeInterface` 推断（Cube Core）；Pipeline 为 MTE2 + MTE3。
2. `NoMaxRankTrait`：不限制 rank；`getOpName()` 返回 `"matmul"`。
3. 操作数通常带 `#hivm.address_space<gm>` 标记。
4. descale 操作数存在时 descaleMode 必须与其形状一致；bias 形状必须 `[n]`。

## 要点

- **Swizzle**：改变 GM 数据块读取顺序以规避 bank conflict；offset 定起始块、direction 定方向。
- **descale 公式**：`C = descale × (A × B + bias)`，服务量化推理场景。
- 需要在矩阵乘后融合 Vector 后处理（激活/量化/通信）时，改用 [mix_matmul](macro_mix_matmul.md)。

相关：[macro_overview.md](macro_overview.md)、[macro_mix_matmul.md](macro_mix_matmul.md)、[macro_mmad_l1.md](macro_mmad_l1.md)、[pipeline_execution_model](../architecture/pipeline_execution_model.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMMacroOps.td#L236-L318](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L236-L318)
- [ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/ops.mlir)
