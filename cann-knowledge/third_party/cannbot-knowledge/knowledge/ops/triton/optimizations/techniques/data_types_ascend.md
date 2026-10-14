---
type: Optimization
title: NPU 数据类型支持矩阵与约束
description: Triton-Ascend 数据类型支持矩阵、tl.dot/tl.load 类型约束、32B/512B 对齐要求、隐式类型提升规则与 FP8
  支持现状。
tags: [task.optimization, topic.precision, topic.programming_model.execution_unit.scalar, topic.compute.quantization, topic.alignment, topic.compute.composite]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Creation_Ops/cast.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# NPU 数据类型支持矩阵与约束

## 综合支持矩阵

| 数据类型 | tl.load/store | tl.dot | 数学运算 | cast | NPU 支持状态 |
|----------|:---|:---:|:---:|:---:|---|
| int1 (bool) | 支持 | 支持(输入) | 部分 | 支持 | 部分（内部转 int8，占 1 字节） |
| int8 / int16 / int32 | 支持 | 支持 | 支持 | 支持 | 完全支持 |
| int64 | 支持 | 不支持 | 部分 | 支持 | 部分（Vector ADD/CMP/MUL/DIV 退化） |
| uint8 | 不支持 | 不支持 | 部分 | 支持 | 不支持，host 端转 int8 |
| uint16/32/64 | 不支持 | 不支持 | 不支持 | 不支持（uint8 可 cast） | 不支持 |
| fp16 / bf16 / fp32 | 支持 | 支持 | 支持 | 支持 | 完全支持 |
| fp64 | 不支持 | 不支持 | 部分 | 不支持 | 不支持，用 fp32 替代 |
| fp8e4nv/e4b15/e5 | 部分 | A2/A3 不支持；910_95 支持 dot_scaled | 不支持 | 支持（910_95） | 部分 |

Ascend 后端支持 FP8 集合：`("fp8e5", "fp8e4b15", "fp8e4nv", "fp8e4b8", "fp8e5b16")`。

## tl.dot 特殊限制

- acc 不支持 FP16，硬件默认 FP32 累加
- `max_num_imprecise_acc` 暂不支持
- `out_dtype` 相比 GPU 缺 int8 和 FP16
- `input_precision` 默认 `"ieee"`，可选 `"hf32"`（仅 fp32 输入时生效）
- FP8 作 dot 输入：A2/A3 不支持；910_95 仅支持 `tl.dot_scaled` 的 FP8 输入

## 对齐要求（尾轴）

| 场景 | 对齐 | 说明 |
|------|------|------|
| VV 类算子（纯 Vector） | 尾轴 32B | fp16/bf16：元素数为 16 倍数；fp32：8 倍数；int8：32 倍数 |
| CV 类算子（含 Cube） | 尾轴 512B | fp16：256 倍数；fp32：128 倍数 |
| L0A/L0B/L0C | 512B | 硬件要求 |

尾轴不满足时硬件自动补齐：额外内存占用 + 无效搬运计算。短轴场景（如 `(2048, 3)` bf16）用借轴转置规避：按 1D load 后 `reshape.trans.reshape` 重组出对齐轴。

## 隐式类型提升规则

integer_promote（整数）：同符号取宽类型；有符号 vs 无符号时，有符号宽度 >= 无符号宽度取有符号类型，否则取无符号类型。

computation_type（混合运算）：

| 操作数 A | 操作数 B | 运算 | 提升结果 |
|----------|----------|------|----------|
| fp16 | fp16 | +,-,* | fp16 |
| fp16 | fp16 | /,% | fp32（除法/取模强制升精度） |
| bf16 | bf16 | +,-,* | bf16 |
| bf16 | bf16 | /,% | fp32 |
| bf16 | fp16 | 任意 | fp32（混 16-bit 类型升 fp32，有额外开销） |
| fp32 | 任意低精度 | 任意 | fp32 |
| fp8 x | fp8 y (x≠y) | 任意 | fp16 |
| 标量 | tensor | 任意 | tensor 类型（标量级别更低时不提升；fp16/bf16 除法升 fp32） |

## bf16 vs fp16 选择

| 特性 | fp16 | bf16 |
|------|------|------|
| 尾数/指数位 | 10/5 | 7/8 |
| 表示范围 | ~5.96e-8 ~ 65504 | ~1.18e-38 ~ 3.39e+38 |
| 十进制精度 | ~3.3 位 | ~2.1 位 |
| 与 fp32 转换 | 调指数+尾数 | 仅截断/扩展尾数（开销更低） |

推荐：matmul 输入用 bf16（范围同 fp32 不易溢出）；高精度累加用 fp32；同一 kernel 统一一种 16-bit 类型避免隐式升 fp32。

## int64 大偏移处理

数据量超 2^31 时用 int64 偏移，借助扩展 API：

```python
from triton.backends.ascend.language.cann.extension.core import int64 as al_int64

large_offset = al_int64(pid) * al_int64(BLOCK_SIZE)
full_offsets = large_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
```

## UB 容量上限（dtype 相关硬约束）

所有 tensor（含 multiBuffer 副本）总和：A2/A3 开 double buffer 时不超过 96KB；910_95 开 double buffer 时不超过 128KB。

## NPU 适配要点

1. uint 系列与 fp64 多数操作不支持，host 端转 int8/16/32 或 fp32
2. matmul 推荐 bf16 输入 + fp32 累加器
3. FP8 仅用于存储传输，计算前转 fp16/bf16/fp32
4. bool 内部转 int8 运算，大量 bool tensor 易致 UB 溢出
5. int64 的 ADD/CMP/MUL/DIV 退化为 scalar，见标量降级规避卡


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L288-L293](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L288-L293)
- [compiler.py#L835](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L835)
- [semantic.py#L46-L58](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L46-L58)
- [semantic.py#L61-L100](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L61-L100)
- [core.py#L288-L683](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L288-L683)
- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py)
- [dot.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md)
- [tl.load.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md)
- [cast.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Creation_Ops/cast.md)
