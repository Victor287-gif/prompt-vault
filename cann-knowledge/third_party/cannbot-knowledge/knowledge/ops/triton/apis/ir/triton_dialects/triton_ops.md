---
type: API
title: Triton 方言操作详解（tt.*）
description: Triton 方言（tt 前缀）全部操作的签名与语义：类型转换、特殊算术、指针算术、load/store、原子操作、形状变换、SPMD、tt.dot/dot_scaled、reduce/scan/map_elementwise、张量指针与
  TMA 描述符操作、函数与外部调用。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.implementation.atomic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Triton 方言操作详解（tt.*）

Triton 方言是设备无关的张量计算 IR 层，绝大多数操作带 `Pure` + `Elementwise` traits。类型约束（`TT_PtrLike`、`TT_Tensor` 等）与枚举属性（`CacheModifier`、`EvictionPolicy` 等）见 [triton_types_attrs.md](triton_types_attrs.md)。

## 类型转换

| 操作 | 语义 |
|------|------|
| `tt.int_to_ptr` / `tt.ptr_to_int` | i64 ↔ 指针互转（`TT_I64Like` ↔ `TT_PtrLike`） |
| `tt.bitcast` | 同位宽位转换（arith.bitcast 不支持指针，故由 Triton 提供） |
| `tt.fp_to_fp` | 浮点互转，支持 F8 与自定义舍入（`rounding = #tt<rounding rtne>`），`hasFolder` |

## 特殊算术与指针算术

- `tt.clampf %x, %min, %max, propagateNan = #tt<propagate_nan all>`：浮点限幅。
- `tt.precise_sqrt` / `tt.precise_divf`：高精度平方根 / 除法；`tt.mulhiui`：无符号乘积高 N 位。
- `tt.addptr %ptr, %offset : !tt.ptr<f32>, i32`：指针按元素偏移前进（`hasFolder`）；`tt.advance %ptr, [%off0, %off1]`：推进张量指针。

## 内存访问

### tt.load / tt.store

| 项 | tt.load | tt.store |
|----|---------|----------|
| 输入 | `ptr`: `TT_PtrLike \| TT_TensorPtr`；可选 `mask`、`other` | `ptr`（MemWrite）、`value`、可选 `mask` |
| 属性 | `boundaryCheck`（DenseI32Array，默认 `[]`）、`padding`（可选）、`cache`（默认 NONE）、`evict`（默认 NORMAL）、`isVolatile`（默认 false） | `boundaryCheck` / `cache` / `evict` 同左 |

```mlir
%result = tt.load %ptr, %mask, %other cacheModifier = #tt<cache ca> evictionPolicy = #tt<evict evict_normal> : !tt.ptr<tensor<128xf32>>
tt.store %ptr, %value, %mask cacheModifier = #tt<cache cg> : !tt.ptr<tensor<128xf32>>
```

## 原子操作

- `tt.atomic_rmw`：读-改-写，返回旧值。`atomic_rmw_op` 枚举 AND/OR/XOR/ADD/FADD/MAX/MIN/UMAX/UMIN/XCHG；属性 `sem`（MemSemantic）、`scope`（MemSyncScope）。
- `tt.atomic_cas`：比较-交换，相等则写入 `$val`，返回旧值。

```mlir
%old = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) -> tensor<128xf32>
```

## 形状操作

| 操作 | 语义 |
|------|------|
| `tt.splat` / `tt.unsplat` | 标量广播为张量 / 单元素张量转标量 |
| `tt.expand_dims` | 指定轴插入大小为 1 的维度 |
| `tt.reshape` | 重解释形状；`allow_reorder` 允许改元素序，`efficient_layout` 允许选高效布局 |
| `tt.broadcast` | 大小为 1 的维度广播 |
| `tt.cat` | 沿最内维连接（非 Pure，可能重排元素） |
| `tt.join` / `tt.split` | 沿新最内维合并（两 `4x8` → `4x8x2`）/ 沿最内维拆分（最内维须为 2） |
| `tt.trans` | 维度重排，实现 `tl.trans`/`tl.permute`；属性 `order: DenseI32ArrayAttr` |

## SPMD

`tt.get_program_id x` / `tt.get_num_programs x`：获取程序 ID / 程序总数（axis 为 X/Y/Z），返回 i32。

## 矩阵乘法

### tt.dot

`d = matmul(a, b) + c`。属性 `inputPrecision`（默认 IEEE；f32 输入时可选 `tf32` / `tf32x3` / `ieee`）、`maxNumImpreciseAcc`（默认 0）。输出类型同 `$c`。

```mlir
%d = tt.dot %a, %b, %c, inputPrecision = #tt<input_precision tf32> : tensor<16x32xf16> * tensor<32x64xf16> -> tensor<16x64xf32>
```

### tt.dot_scaled

`d = matmul(scale(a, a_scale), scale(b, b_scale)) + c`，输入可为 Float/I8 张量加可选 scale 张量；属性 `a_elem_type` / `b_elem_type`（ScaleDotElemType：e4m3/e5m2/e2m3/e3m2/e2m1/bf16/fp16）、`fastMath`、`lhs_k_pack` / `rhs_k_pack`（默认 true）。

## 归约 / 扫描 / 映射

- `tt.reduce`：通用组合归约（属性 `axis`，区域 `combineOp` 内含 `tt.reduce.return`）；`tt.scan`：关联前缀扫描（属性 `axis`、`reverse`）。
- `tt.map_elementwise`：把标量子区域映射到每个元素，属性 `pack`。

```mlir
%result = tt.reduce %src {axis = 0 : i32} {
^bb0(%arg0: f32, %arg1: f32):
  %0 = arith.addf %arg0, %arg1 : f32
  tt.reduce.return %0 : f32
} : tensor<128xf32> -> f32
```

## 范围 / 统计 / 调试

- `tt.make_range {start, end}`：生成 1D i32 张量（步长 1，不含 end）；`tt.histogram`：直方图（bin 宽 1，从 0 起）。
- `tt.gather %src[%indices] {axis, efficient_layout}`：按索引收集。
- `tt.print` / `tt.assert`：设备端打印 / 断言（均 `MemWrite<GlobalMemory>`）。

## 张量指针与 TMA 描述符

| 操作 | 语义 |
|------|------|
| `tt.make_tensor_ptr` | 由 base/shape/strides/offsets + `order` 构造 `TT_TensorPtr` |
| `tt.make_tensor_descriptor` | 构造 `!tt.tensordesc`（TMA 描述符，属性 `padding` 默认 PAD_ZERO） |
| `tt.descriptor_load` / `tt.descriptor_store` | 基于描述符的 TMA 读写（按 `indices` 定位） |
| `tt.descriptor_reduce` | 描述符归约存储，kind = ADD/MIN/MAX/INC/DEC/AND/OR/XOR |
| `tt.descriptor_gather` / `tt.descriptor_scatter` | 描述符多行收集 / 散射（`x_offsets` + `y_offset`） |

## 函数与外部调用

- `tt.func` / `tt.call` / `tt.return`：函数定义（`FunctionOpInterface`、`IsolatedFromAbove`）、调用与返回。
- `tt.extern_elementwise`：调用外部函数（属性 `libname`/`libpath`/`symbol`/`pure`）。
- `tt.elementwise_inline_asm`：对打包元素组执行内联汇编（属性 `asm_string`/`constraints`/`packed_element`）。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [TritonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Triton/IR/TritonOps.td)
