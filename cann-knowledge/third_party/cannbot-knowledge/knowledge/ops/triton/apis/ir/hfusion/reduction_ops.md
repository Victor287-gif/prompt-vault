---
type: API
title: HFusion 归约操作（reduce_with_index / cumsum / cumprod）
description: hfusion.reduce_with_index 带索引 max/min 归约（tie_break_left、dimensions、unsigned_src）与
  hfusion.cumsum/cumprod 累积操作的签名、类型约束与语义。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 归约操作

三个归约类操作：带索引归约 `reduce_with_index`（结构化操作）与累积操作 `cumsum` / `cumprod`。普通值归约由 Linalg 命名操作承载，本方言补充"值+索引"与累积语义。

## reduce_with_index

对 AnyShaped 执行 max/min 归约，同时产出归约值与对应索引。

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `inputs` | `Variadic<AnyShaped>` | 输入张量（两种模式：输入+索引，或仅输入） |
| `inits` | `Variadic<AnyShaped>` | 初始值（DPS init） |
| `reduce_kind` | `HFusion_ReduceWithIndexOpAttr` | `min` / `max` |
| `unsigned_src` | `BoolAttr` | 源是否为无符号整数 |
| `tie_break_left` | `OptionalAttr<BoolAttr>` | 多元素同值时取最左索引 |
| `dimensions` | `DenseI64ArrayAttr` | 归约维度 |
| `result` | `Variadic<AnyTensor>` | 归约值 + 索引 |

Traits：`AttrSizedOperandSegments`、`ResultOnlyIfTensor`、`DestinationStyleOpInterface`、`LinalgStructuredInterface`、`ReifyRankedShapedTypeOpInterface`。

语义（max 为例）：`result_value[i] = max(input[i][:])`，`result_index[i] = argmax(input[i][:])`；`tie_break_left = true` 时同值取最左索引。

```mlir
%val, %idx = hfusion.reduce_with_index
  ins(%input, %val_init, %idx_init : tensor<128x256xf32>, tensor<128xf32>, tensor<128xi64>)
  outs(%val_out, %idx_out : tensor<128xf32>, tensor<128xi64>)
  reduce_with_index_kind = <max> unsigned_src = false dimensions = [1]
```

## cumsum / cumprod

沿指定维度做累积和/累积积。属性 `cum_dims`（DenseI64ArrayAttr，严格递增，当前仅支持单维度）、`reverse`（BoolAttr，是否反向累积）。

| 操作 | 输入/输出元素类型 |
|------|-------------------|
| `cumsum` | BF16, F16, F32, I8, I16, I32, I64, F8E4M3FN, F8E5M2 |
| `cumprod` | BF16, F16, F32, I8, I16, I32, I64（不含 FP8） |

语义：正向 `output[i] = sum/product(input[0..i])`；反向 `output[i] = sum/product(input[i..N-1])`。

```mlir
%0 = hfusion.cumsum %input cum_dims = [1] reverse = false : tensor<128x256xf32> -> tensor<128x256xf32>
%1 = hfusion.cumprod %input cum_dims = [0] reverse = true : tensor<128x256xf32> -> tensor<128x256xf32>
```

辅助枚举 `CumOpType`：`undefined` / `cumsum` / `cumprod`，供 Pass 区分累积类型。

## 与 HIVM 的对应关系

- 值归约 lowering 到 `hir.vreduce` 等，见 [vector_reduction_ops](../hivm/vector_reduction_ops.md)
- 累积与排序类操作在 HIVM 侧的承载见 [vector_cumulative_sort](../hivm/vector_cumulative_sort.md)

相关：[overview.md](overview.md)、[attributes_enums.md](attributes_enums.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HFusionStructuredOps.td#L60-L142](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td#L60-L142)
- [HFusionOps.td#L275-L323](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L275-L323)
