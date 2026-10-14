---
type: API
title: HFusion 矩阵乘操作（matmul_mx / group_matmul）
description: hfusion.matmul_mx 微缩放（MX）格式 FP8 矩阵乘（A·scale_a dot B·scale_b）与 hfusion.group_matmul
  MoE 分组矩阵乘的签名、indexing maps、iterator types 与数据格式约束。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 矩阵乘操作

两个矩阵乘操作均为 NPU Cube 单元的核心负载，lowering 后落到 HIVM 宏操作族（[macro_overview](../hivm/macro_overview.md)）。

## matmul_mx（微缩放格式矩阵乘）

计算 `C = (A * scale_a) dot (B * scale_b)`，输入为 FP8 等量化类型，缩放因子按 OCP Microscaling Formats（MX）规范共享（典型为每 32 个元素共享 1 个 UI8 scale），在保精度同时降低存储与计算开销。

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `inputA` / `inputB` | `ShapedTypeOf<[F8E4M3FN, F8E5M2]>` | 左/右矩阵 |
| `scaleA` / `scaleB` | `ShapedTypeOf<[UI8, I8]>` | 左/右缩放因子 |
| `acc` | `ShapedTypeOf<[AnyFloat]>` | 累加器（DPS init） |
| `result` | `ShapedTypeOf<[AnyFloat]>` | 输出 |

Traits：`Pure`、`DestinationStyleOpInterface`、`BiShengIRAggregatedOpInterface`（支持 decomposeOperation 分解）。

```mlir
%r = hfusion.matmul_mx
  ins(%a, %b, %sa, %sb : tensor<16x32xf8E4M3FN>, tensor<32x64xf8E4M3FN>, tensor<2x2xui8>, tensor<2x4xui8>)
  outs(%acc : tensor<16x64xf32>) -> tensor<16x64xf32>
```

FP8 数据格式：`F8E4M3FN`（4 位指数 3 位尾数、无无穷，前向用）、`F8E5M2`（5 位指数 2 位尾数、支持无穷，反向用）。

## group_matmul（分组矩阵乘，MoE 场景）

每个 Expert 的权重矩阵与其分配的 Token 做矩阵乘，用于 Mixture of Experts。

| 操作数 | 角色 | shape_map |
|--------|------|-----------|
| `w1` (T1) | Expert 权重矩阵 | `(d0, d1, d2) -> (d2, d1, d0)` |
| `tokens` (T2) | Token 嵌入 | `(d0, d1, d2) -> (d1, d2)` |
| `tokens_per_expert` (T3) | 每 Expert 的 Token 数 | `(d0, d1, d2) -> (1)` |
| `output` (T4) | 输出 | `(d0, d1, d2) -> (d0, d0)` |

Iterator types：`["parallel", "parallel", "reduction"]`。

```mlir
%r = hfusion.group_matmul
  ins(%w1, %tokens, %tokens_per_expert : tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<1xi64>)
  outs(%output : tensor<?x?xf32>)
```

## 辅助枚举 MmMapMode

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `CoreOp` | `core_op` | 核心操作模式 |
| `MacroInstr` | `macro_instr` | 宏指令模式 |

## 与 HIVM 的对应关系

- 普通矩阵乘 lowering → `hir.matmul`，见 [macro_matmul](../hivm/macro_matmul.md)
- MoE 分组矩阵乘 lowering → `hir.mix_group_matmul`，见 [macro_mix_group_matmul](../hivm/macro_mix_group_matmul.md)
- Mix 核心协同变体见 [macro_mix_matmul](../hivm/macro_mix_matmul.md)

相关：[overview.md](overview.md)、[attributes_enums.md](attributes_enums.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HFusionOps.td#L964-L1021](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L964-L1021)
- [HFusionNamedStructuredOps.yaml#L386-L424](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yaml#L386-L424)
