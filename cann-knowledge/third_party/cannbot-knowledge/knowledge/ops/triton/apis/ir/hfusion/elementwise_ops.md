---
type: API
title: HFusion 逐元操作（elemwise_unary / elemwise_binary / compare / select / cast / bitcast）
description: HFusion 六个逐元结构化操作及张量语义 load/store 的语义、操作数与函数选择属性（unary_fn/binary_fn/compare_fn/round_mode
  等），经 OpDSL 声明式定义，lowering 到 HIVM 向量指令。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 逐元操作

六个逐元操作均为 Linalg 结构化操作，经 OpDSL（HFusionNamedStructuredOps.yaml）声明式定义，Region 内描述标量逻辑。函数选择枚举的完整整数值见 [attributes_enums.md](attributes_enums.md)；lowering 后的 HIVM 对应物见各互链卡。

## 操作速查

| 操作 | 语义 | 关键属性 |
|------|------|----------|
| `hfusion.load` | `O = I`，逐元加载不做类型转换 | — |
| `hfusion.store` | `O = atomic_kind(I)`，支持原子存储语义 | `atomic_kind`（默认 NONE） |
| `hfusion.elemwise_unary` | `O = fun(cast(I))` | `unary_fn`（默认 sqrt）、`cast`（默认 cast_signed） |
| `hfusion.elemwise_binary` | `O = fun(cast(lhs), cast(rhs))` | `binary_fn`（默认 vand）、`cast` |
| `hfusion.compare` | `O = compare_fn(lhs, rhs)`，两输入同类型 | `compare_fn`（默认 veq） |
| `hfusion.select` | `O = select(cond, lhs, rhs)` | — |
| `hfusion.cast` | `O = round(cast(I), round_mode, unsigned_mode)` | `round_mode`（默认 RINT）、`enable_overflow`（默认 true）、`enable_saturate`（默认 false）、`unsigned_mode`（默认 SI2SI）、`cast` |
| `hfusion.bitcast` | 位模式不变的逐元位转换 | — |

## 函数选择取值

### unary_fn（18 个）

relu, sqrt, rsqrt, rec, vnot, tanh, sin, cos, atan, tan, absi, erf, log2, log10, log1p, exp2, expm1, ilogb

### binary_fn（19 个）

vor, vand, vxor, minf, maxf, powf, mod, shli, shrsi, shrui, ldexp, ceildivsi, ceildivui, floordivsi, powi, minnumf, maxnumf, modui, divfhp（高精度除法）

### compare_fn（10 个）

veq, vne, vle, vlt, vge, vgt（有符号），vule, vult, vuge, vugt（无符号）

## MLIR 示例

```mlir
%0 = hfusion.elemwise_unary ins(%in : tensor<128x256xf16>)
  outs(%init : tensor<128x256xf32>) unary_fn = <sqrt> cast = <cast_signed>

%1 = hfusion.elemwise_binary ins(%lhs, %rhs : tensor<128x256xf16>, tensor<128x256xf16>)
  outs(%init : tensor<128x256xf16>) binary_fn = <vadd> cast = <cast_signed>

%2 = hfusion.compare ins(%lhs, %rhs : tensor<128x256xf32>, tensor<128x256xf32>)
  outs(%init : tensor<128x256xi1>) compare_fn = <vgt>

%3 = hfusion.cast ins(%in : tensor<128x256xf32>)
  outs(%init : tensor<128x256xi8>) round_mode = <FLOOR> cast = <cast_signed>
```

## 与 HIVM 的对应关系

lowering 到 HIVM 后，逐元二元/一元/比较/转换分别落到 `hir.vadd` 等 `hir.v*` 向量指令族：

- 二元算术/位运算 → [vector_binary_ops](../hivm/vector_binary_ops.md)
- 一元函数 → [vector_unary_ops](../hivm/vector_unary_ops.md)
- 比较（vcmp）→ [vector_compare_ops](../hivm/vector_compare_ops.md)
- 类型转换（vcast，含 round_mode 映射）→ [vector_cast_ops](../hivm/vector_cast_ops.md)

相关：[overview.md](overview.md)、[attributes_enums.md](attributes_enums.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HFusionNamedStructuredOps.yaml](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yaml)
