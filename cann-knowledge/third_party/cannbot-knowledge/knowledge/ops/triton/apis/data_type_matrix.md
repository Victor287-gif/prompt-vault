---
type: API
title: 数据类型支持矩阵（Data Type Support Matrix）
description: Triton-Ascend 各数据类型在 load/store、dot、数学、归约、cast、原子操作上的支持矩阵；512B 对齐建议、dtype
  选择推荐与 Ascend 扩展 cast 参数。
tags: [task.lookup, topic.precision, topic.quality.compatibility.version_migration, topic.alignment, topic.compute.quantization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Creation_Ops/cast.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---
# 数据类型支持矩阵

标记：✓ 完全支持；✓* 支持但 Triton 内部将 bool 转为 int8 运算；× 不支持（硬件限制）；部分 = 部分操作支持。910_95 对 FP8 有额外支持，表中以备注说明。

## 数据类型总览

| 数据类型 | Triton 名称 | 字节数 | Ascend 支持 | 说明 |
|---------|------------|:------:|:----------:|------|
| int8 | `tl.int8` | 1 | ✓ | 占用更大片上空间 |
| int16 | `tl.int16` | 2 | ✓ | - |
| int32 | `tl.int32` | 4 | ✓ | 最常用整数类型 |
| int64 | `tl.int64` | 8 | ✓ | mod 仅支持 -2^24 ~ 2^24 |
| uint8 | `tl.uint8` | 1 | × | 硬件不支持 |
| uint16 | `tl.uint16` | 2 | × | 硬件不支持 |
| uint32 | `tl.uint32` | 4 | × | 硬件不支持 |
| uint64 | `tl.uint64` | 8 | × | 硬件不支持 |
| fp16 | `tl.float16` | 2 | ✓ | 近似计算 |
| fp32 | `tl.float32` | 4 | ✓ | 推荐计算精度 |
| fp64 | `tl.float64` | 8 | × | 硬件不支持 |
| bf16 | `tl.bfloat16` | 2 | ✓ | NPU 推荐推理精度 |
| fp8e4 | `tl.float8e4nv` | 1 | 部分 | 910_95 支持类型转换和 dot_scaled；A2/A3 不支持 |
| fp8e5 | `tl.float8e5` | 1 | 部分 | 910_95 支持类型转换和 dot_scaled；A2/A3 不支持 |
| bool | `tl.int1` | 0.125 | ✓* | 内部转为 int8 |

## Load/Store 支持

| 数据类型 | tl.load | tl.store | 说明 |
|---------|:-------:|:--------:|------|
| int8 / int16 / int32 / int64 | ✓ | ✓ | - |
| uint8 / uint16 / uint32 / uint64 | × | × | 硬件限制 |
| fp16 / fp32 / bf16 | ✓ | ✓ | - |
| fp64 | × | × | 硬件限制 |
| bool | ✓ | ✓ | - |

## Dot（矩阵乘）支持

| 输入类型 | tl.dot | 累加器类型 | out_dtype | 说明 |
|---------|:------:|:---------:|:---------:|------|
| int8 × int8 | ✓ | fp32 | int32 | out_dtype 缺乏 int8 支持 |
| int16 × int16 | × | - | - | - |
| fp16 × fp16 | ✓ | fp32 | fp32 | 硬件默认 fp32 累加 |
| fp32 × fp32 | ✓ | fp32 | fp32 | - |
| bf16 × bf16 | ✓ | fp32 | fp32 | NPU 推荐 |

> `max_num_imprecise_acc` 暂不支持。

## 数学运算支持

| 数据类型 | 算术(+,-,*,/) | 位运算(&,\|,^,~,<<,>>) | 三角函数 | 指数/对数 | 比较(>,<,==) |
|---------|:------------:|:--------------------:|:-------:|:--------:|:-----------:|
| int8 | ✓ | ✓ | × | × | ✓ |
| int16 | ✓ | ✓ | × | × | ✓ |
| int32 | ✓ | ✓ | × | × | ✓ |
| int64 | ✓ | ✓ | × | × | ✓ |
| fp16 | ✓ | × | ✓ | ✓ | ✓ |
| fp32 | ✓ | × | ✓ | ✓ | ✓ |
| bf16 | ✓ | × | ✓ | ✓ | ✓ |
| bool | ✓* | ✓ | × | × | ✓* |

## 归约操作支持

| 数据类型 | tl.sum | tl.max | tl.min | tl.argmax | tl.argmin | tl.reduce |
|---------|:------:|:------:|:------:|:---------:|:---------:|:---------:|
| int8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bool | ✓* | ✓* | ✓* | × | × | ✓* |

## 类型转换（Cast）支持

所有支持类型间可互转（int8/16/32/64、fp16/32、bf16、bool 全矩阵 ✓）。Ascend 扩展 cast 参数：

| 参数 | 说明 | Ascend 扩展 |
|------|------|:----------:|
| `fp_downcast_rounding` | 浮点降精度舍入：`rtne`（默认）、`rtz` | - |
| `bitcast` | 位级别重解释 | - |
| `overflow_mode` | 整数溢出处理：`trunc`（默认）、`saturate` | ✓ Ascend 扩展 |

```python
@triton.jit
def cast_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    x = tl.load(x_ptr + i, mask=mask)
    y = tl.cast(x, tl.float32)
    y = tl.cast(y, tl.bfloat16, fp_downcast_rounding="rtz")
    tl.store(y_ptr + i, y, mask=mask)

# 饱和转换
y = tl.cast(x, tl.int8, overflow_mode="saturate")
```

## 对齐要求汇总

NPU 芯片亲和 512B 对齐场景：

| 数据类型 | 元素大小 | 512B 对齐元素数 | 推荐 BLOCK_SIZE |
|---------|:-------:|:--------------:|:--------------:|
| int8 | 1B | 512 | 512 的倍数 |
| int16 / fp16 / bf16 | 2B | 256 | 256 的倍数 |
| int32 / fp32 | 4B | 128 | 128 的倍数 |
| int64 | 8B | 64 | 64 的倍数 |

## 原子操作 dtype 支持

| 数据类型 | atomic_add | atomic_max | atomic_min | atomic_and | atomic_or | atomic_xor | atomic_xchg | atomic_cas |
|---------|:---------:|:----------:|:----------:|:----------:|:---------:|:----------:|:-----------:|:----------:|
| int8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| int16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int64 | × | × | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp16 | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ |
| fp32 | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ |
| bf16 | ✓ | ✓ | ✓ | × | × | × | × | × |

> 所有 atomic op：`sem` 只支持 `"acq_rel"`，`scope` 只支持 `"gpu"`；`atomic_or/xor/and/xchg/cas` 暂不支持在 loop 中使用。

## dot_scaled dtype 支持

| 缩放张量类型 | fp4 | fp8 | bf16 | fp16 | 说明 |
|------------|:---:|:---:|:----:|:----:|------|
| Ascend A2/A3 | × | × | ✓ | ✓ | 缩放张量值为 int8（GPU 为 uint8） |
| Ascend 910_95 | × | ✓ | ✓ | ✓ | 缩放张量值为 int8（GPU 为 uint8） |

## 推荐 dtype 选择

| 场景 | 输入 | 计算 | 输出 | 说明 |
|------|:----:|:----:|:----:|------|
| 矩阵乘推理 | bf16 | fp32 | bf16 | NPU 上 bf16 性能最优 |
| 矩阵乘训练 | fp16 | fp32 | fp16 | fp16 动态范围更大 |
| 归约操作 | fp16/bf16 | fp32 | fp16/bf16 | 归约内部 fp32 避免精度损失 |
| 向量操作 | fp32 | fp32 | fp32 | 简单操作直接 fp32 |
| 索引操作 | int32 | int32 | int32 | 索引统一 int32 |
| 量化推理 | int8 | fp32 | int8/bf16 | int8 省带宽但占更大片上空间 |

精度建议：
1. **累加器始终 fp32**：`tl.dot` 累加器、归约中间结果都用 fp32
2. **bf16 优于 fp16**：动态范围更大，计算性能相当
3. **避免 fp64**：硬件不支持，用 fp32 替代
4. **避免 uint 类型**：硬件不支持，host 端先转 int32（如 uint8→int32），计算后转回
5. **bool 建议显式用 int8 替代**：内部转 int8 运算（✓*），结果可能与预期不同
6. **int8 易 UB overflow**：遇报错调整 tiling（增大 BLOCK_SIZE 或减少同时存活的 tensor 数）

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [outline.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/sources/python-api/outline.md)
