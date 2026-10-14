---
type: API
title: HFusion 属性与枚举速查（UnaryFn / BinaryFn / CompareFn / RoundMode / AtomicKind / FusionKind
  等）
description: HFusion 方言全部控制枚举与属性：逐元函数选择（unary/binary/compare/ternary/type_fn）、舍入与无符号模式、原子操作类型、融合类型
  FusionKind、FlattenMode/OutputMode/CastMode 及 Padding/Cache/Eviction 访存修饰。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector, topic.implementation.atomic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 属性与枚举速查

控制逐元操作函数选择、类型转换、舍入、原子类型与融合策略的全部枚举。定义于 HFusionEnums.td / HFusionBase.td / HFusionAttrs.td。使用这些枚举的操作见 [elementwise_ops.md](elementwise_ops.md)、[memory_ops.md](memory_ops.md)、[transforms.md](transforms.md)。

## UnaryFn（unary_fn = <sqrt>，18 个）

| 值 | 整数 | 值 | 整数 | 值 | 整数 |
|----|------|----|------|----|------|
| relu | 0 | sin | 6 | log2 | 12 |
| sqrt | 1 | cos | 7 | log10 | 13 |
| rsqrt | 2 | atan | 8 | log1p | 14 |
| rec | 3 | tan | 9 | exp2 | 15 |
| vnot | 4 | absi | 10 | expm1 | 16 |
| tanh | 5 | erf | 11 | ilogb | 17 |

## BinaryFn（binary_fn = <vand>，19 个）

| 值 | 整数 | 说明 | 值 | 整数 | 说明 |
|----|------|------|----|------|------|
| vor | 0 | 按位或 | ldexp | 10 | ldexp |
| vand | 1 | 按位与 | ceildivsi | 11 | 有符号向上取整除 |
| vxor | 2 | 按位异或 | ceildivui | 12 | 无符号向上取整除 |
| minf | 3 | 浮点最小 | floordivsi | 13 | 有符号向下取整除 |
| maxf | 4 | 浮点最大 | powi | 14 | 整数幂 |
| powf | 5 | 浮点幂 | minnumf | 15 | IEEE 浮点最小 |
| mod | 6 | 取模 | maxnumf | 16 | IEEE 浮点最大 |
| shli | 7 | 左移 | modui | 17 | 无符号取模 |
| shrsi | 8 | 算术右移 | divfhp | 18 | 高精度除法 |
| shrui | 9 | 逻辑右移 | | | |

## CompareFn（compare_fn = <veq>，10 个）

veq(0), vne(1), vle(2), vlt(3), vge(4), vgt(5)（有符号）；vule(6), vuge(7), vugt(8), vult(9)（无符号）。

## TernaryFn / TypeFn

- `ternary_fn`：select(0)，条件选择
- `type_fn`：cast_signed(0) 有符号转换 / cast_unsigned(1) 无符号转换 / bitcast(2) 位转换

## RoundMode（round_mode = <RINT>）

| 值 | 整数 | 说明 |
|----|------|------|
| RINT | 0 | 四舍五入到偶数（C rint） |
| ROUND | 1 | 四舍五入远离零（C round） |
| FLOOR | 2 | 向负无穷取整 |
| CEIL | 3 | 向正无穷取整 |
| TRUNC | 4 | 向零取整 |
| ODD | 5 | 向奇数取整（Von Neumann 舍入） |
| TRUNCWITHOVERFLOW | 6 | 截断并检测溢出 |

## UnsignedMode（unsigned_mode = <SI2SI>）

SI2SI(0) / SI2UI(1) / UI2SI(2) / UI2UI(3)，整数有/无符号互转方向。

## AtomicKind（atomic_kind = <add>）

| 值 | 整数 | 值 | 整数 |
|----|------|----|------|
| NONE | 0 | XOR | 6 |
| ADD | 1 | CAS | 7 |
| MAX | 2 | XCHG | 8 |
| MIN | 3 | UMAX | 9 |
| AND | 4 | UMIN | 10 |
| OR | 5 | | |

MAX/MIN 为有符号，UMAX/UMIN 为无符号。

## FusionKind（融合类型，hfusion-fuse-ops 的 fusion-mode 选项）

| 值 | 整数 | 助记符 | 说明 |
|----|------|--------|------|
| PureElemwise | 1 | PURE_ELEMWISE | 纯逐元融合 |
| AnyPB | 2 | ANY_PB | 任意 Producer-Consumer 融合 |
| LastAxisPBR | 3 | LAST_AXIS_PBR | 末轴 PBR 融合 |
| AnyPBR | 4 | ANY_PBR | 任意 PBR 融合 |
| SingleCube | 5 | SINGLE_CUBE | 单 Cube 融合 |
| ShallowCV | 6 | SHALLOW_CV | 浅层 Cube-Vector 融合 |
| ShallowVV | 7 | SHALLOW_VV | 浅层 Vector-Vector 融合 |
| MixCV | 8 | MIX_CV | 混合 Cube-Vector 融合 |
| MixC2 | 9 | MIX_C2 | 双 Cube 混合融合 |
| Unknown | 10 | UNKNOWN | 未知 |

## 其他枚举

| 枚举 | 取值 |
|------|------|
| FlattenMode | Greedy(1) 贪心展平 / Tidy(2) 带分析的整洁展平 |
| OutputMode | Multiple(1) / Single(2) / SingleAggressive(3) |
| ReduceWithIndexKind | min / max |
| CastMode | F32TOI8, F32TOI16, F16TOI8, I64TOI32, I64TOI16, I64TOI8, I32TOI16, I32TOI8, I16TOI8 |
| TaylerMode | sin / atan（泰勒展开类型） |
| MmMapMode | core_op / macro_instr（矩阵乘映射模式，见 [matmul_ops.md](matmul_ops.md)） |

## HFusion 属性列表

| 属性 | 助记符 | 说明 |
|------|--------|------|
| FusionKindAttr | `fusion_kind` | 融合类型 |
| StrideAlignDimsAttr | `stride_align_dims` | 步长对齐维度 |
| StrideAlignValueInByteAttr | `stride_align_value_in_byte` | 步长对齐字节数 |
| MultiBufferAttr | `multi_buffer` | 多缓冲（语义见 [multi_buffer](../memory_management/multi_buffer.md)） |
| BindSubBlockAttr | `bind_sub_block` | 绑定子 Block |
| ReduceComposeAttr | `reduce_composed` | 归约组合 |
| ReduceWithIndexOpAttr | `reduce_with_index_kind` | 带索引归约类型 |
| ReturnOperandNumAttr | `return_operand_num` | 返回操作数编号 |
| PaddingOptionAttr | — | 填充选项 |
| CacheModifierAttr | — | 缓存修饰符 |
| EvictionPolicyAttr | — | 驱逐策略 |

### PaddingOption / CacheModifier / EvictionPolicy

| 枚举 | 取值 |
|------|------|
| PaddingOption | PAD_ZERO(1, `zero`) / PAD_NAN(2, `nan`) |
| CacheModifier | NONE(1) / CA(2) / CG(3) / WB(4) / CS(5) / WT(6) / CV(7) |
| EvictionPolicy | NORMAL(1, `evict_normal`) / EVICT_FIRST(2) / EVICT_LAST(3) |

HIVM 侧的属性枚举体系见 [attr_overview](../hivm/attr_overview.md) 与 [attr_enumerations](../hivm/attr_enumerations.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HFusionEnums.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td)
- [HFusionBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionBase.td)
- [HFusionAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td)
