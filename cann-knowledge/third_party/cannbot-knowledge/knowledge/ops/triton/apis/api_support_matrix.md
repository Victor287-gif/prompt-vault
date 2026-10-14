---
type: API
title: Triton API 在 Ascend NPU 上的支持矩阵
description: 全部 Triton Python API 在 Ascend NPU 上的支持状态与 dtype 矩阵（内存/数学/归约/线性代数/原子/形状/索引/扫描排序/随机数），含通用约束。
tags: [task.lookup, topic.quality.compatibility.version_migration, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.cube, topic.implementation.atomic, topic.compute.reduction]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---
# API 支持矩阵（API Support Matrix）

状态标记：✓ 完全支持；✓* 部分支持（有 dtype 或使用限制）；× 不支持；A Ascend 扩展。

> dtype 列：i8=int8, i16=int16, i32=int32, i64=int64, f16=fp16, f32=fp32, bf16=bfloat16。uint8/16/32/64 与 fp64 在 Ascend 上普遍不支持（硬件限制），不单独列出。

## 内存/指针操作

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.load` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.store` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | cache_modifier/eviction_policy 对 NPU 无效 |
| `tl.make_block_ptr` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 不允许算术运算；复杂循环+分支可能编译失败 |
| `tl.make_tensor_descriptor` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 需与 load/store_tensor_descriptor 配套使用 |
| `tl.load_tensor_descriptor` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 同上 |
| `tl.store_tensor_descriptor` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 同上 |
| `tl.advance` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 不允许通过调整 stride 顺序实现转置 |

## 数学操作

### 算术运算

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `+ (add)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `- (sub)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `* (mul)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `/ (div)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `// (floordiv)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓* | - |
| `% (mod)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | int64 仅支持 -2^24 ~ 2^24 |
| `- (neg)` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | - |

### 位运算

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `& (and)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `\| (or)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `^ (xor)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `~ (invert)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `! (not)` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | - |
| `<< (lshift)` | ✓ | ✓ | ✓ | ✓ | × | × | × | × | - |
| `>> (rshift)` | ✓ | ✓ | ✓ | ✓ | × | × | × | × | - |

### 比较运算

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|
| `>` `>=` `<` `<=` `==` `!=` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |

### 浮点数学函数

| API | f16 | f32 | bf16 | 限制说明 |
|-----|:---:|:---:|:----:|---------|
| `tl.abs` | ✓ | ✓ | ✓ | 整数类型也支持 |
| `tl.ceil` | ✓ | ✓ | ✓ | - |
| `tl.clamp` | ✓ | ✓ | ✓ | - |
| `tl.cos` | ✓ | ✓ | ✓ | - |
| `tl.div_rn` | ✓ | ✓ | ✓ | - |
| `tl.erf` | ✓ | ✓ | ✓ | - |
| `tl.exp` | ✓ | ✓ | ✓ | 近似计算 |
| `tl.exp2` | ✓ | ✓ | ✓ | 近似计算 |
| `tl.fdiv` | ✓ | ✓ | ✓ | - |
| `tl.floor` | ✓ | ✓ | ✓ | - |
| `tl.fma` | ✓ | ✓ | ✓ | - |
| `tl.log` | ✓ | ✓ | ✓ | - |
| `tl.log2` | ✓ | ✓ | ✓ | - |
| `tl.maximum` | ✓ | ✓ | ✓ | 整数类型也支持 |
| `tl.minimum` | ✓ | ✓ | ✓ | 整数类型也支持 |
| `tl.rsqrt` | ✓ | ✓ | ✓ | - |
| `tl.sigmoid` | ✓ | ✓ | ✓ | - |
| `tl.sin` | ✓ | ✓ | ✓ | - |
| `tl.softmax` | ✓ | ✓ | ✓ | - |
| `tl.sqrt` | ✓ | ✓ | ✓ | - |
| `tl.sqrt_rn` | ✓ | ✓ | ✓ | - |
| `tl.round` | × | ✓ | × | 仅 fp32 |
| `tl.umulhi` | × | × | × | 仅 i32，不支持负数输入 |

### 逻辑运算

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.where` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓* | - |
| `logical_and` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `logical_or` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |

## 归约操作

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.sum` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `tl.max` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `tl.min` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `tl.argmax` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | - |
| `tl.argmin` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | - |
| `tl.reduce` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | - |
| `tl.xor_sum` | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓* | - |

## 线性代数操作

| API | i8 | i16 | i32 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.dot` | ✓ | × | × | ✓ | ✓ | ✓ | × | acc 不支持 fp16；max_num_imprecise_acc 暂不支持；out_dtype 缺乏 int8/fp16 |
| `tl.dot_scaled` | × | × | × | × | × | × | × | fp4/fp8 在 A2/A3 不支持；910_95 支持 fp8 的 dot_scaled；scale 为 int8（GPU 为 uint8） |

## 原子操作

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|---------|
| `tl.atomic_add` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | 不支持多核 add+保存中间结果 |
| `tl.atomic_and` | ✓ | ✓ | ✓ | ✓ | × | × | × | 不支持在 loop 中使用 |
| `tl.atomic_cas` | × | ✓ | ✓ | ✓ | ✓ | ✓ | × | 不支持在 loop 中使用 |
| `tl.atomic_max` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | - |
| `tl.atomic_min` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | - |
| `tl.atomic_or` | ✓ | ✓ | ✓ | ✓ | × | × | × | 不支持在 loop 中使用 |
| `tl.atomic_xchg` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | 不支持在 loop 中使用 |
| `tl.atomic_xor` | ✓ | ✓ | ✓ | ✓ | × | × | × | 不支持在 loop 中使用 |

> 所有 atomic op：`sem` 只支持默认值 `"acq_rel"`，`scope` 只支持默认值 `"gpu"`。

## 形状操作

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.broadcast` / `tl.broadcast_to` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.expand_dims` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.interleave` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.join` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.permute` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | 不支持不相邻轴转置 |
| `tl.ravel` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.reshape` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.split` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.trans` | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | 不支持不相邻轴转置 |
| `tl.view` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 已弃用，用 reshape |

## 创建操作

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.arange` | × | × | ✓ | × | × | × | × | × | 仅 i32 |
| `tl.cat` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 仅 1D 且 can_reorder=True |
| `tl.full` / `tl.zeros` / `tl.zeros_like` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.cast` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Ascend 扩展 overflow_mode 参数 |

## 索引操作

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.flip` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.swizzle2d` | ✓ | ✓ | ✓ | ✓ | × | × | × | × | - |
| `tl.gather` | × | × | × | × | ✓ | ✓ | ✓ | × | 仅支持 axis=n-1 |

## 扫描/排序操作

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.associative_scan` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | ✓ | - |
| `tl.cumprod` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.cumsum` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| `tl.histogram` | × | × | ✓ | ✓ | × | × | × | × | - |
| `tl.sort` | × | × | × | × | × | × | × | × | 毕昇编译器限制 |

## 随机数生成

| API | i8 | i16 | i32 | i64 | f16 | f32 | bf16 | bool | 限制说明 |
|-----|:--:|:---:|:---:|:---:|:---:|:---:|:----:|:----:|---------|
| `tl.randint4x` | ✓ | ✓ | ✓ | × | × | × | × | ✓ | 输出类型 |
| `tl.randint` | ✓ | ✓ | ✓ | × | × | × | × | ✓ | 输出类型 |
| `tl.rand` | × | × | × | × | ✓ | ✓ | ✓ | ✓ | 输出类型 |
| `tl.randn` | × | × | × | × | ✓ | ✓ | ✓ | ✓ | 输出类型 |

## 通用约束

- **UB 空间**：A2/A3 系列所有 tensor 总和不能超过 96KB（开启 double buffer）/ 192KB（关闭）；910_95 系列不能超过 128KB（开启）/ 256KB（关闭）
- **Shape 限制**：tensor 任何维度 size 不允许小于 1；默认最高支持 8 维 tensor
- **int8 特殊处理**：int8 占用更大片上空间，编译易造成 UB overflow，通常调整 tiling 即可解决
- **标量 tensor**：不支持使用 shape 为 `[[]]` 的标量 tensor 进行计算

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [outline.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/sources/python-api/outline.md)
