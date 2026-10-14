---
type: API
title: 向量操作扩展（insert_slice / extract_slice / get_element / sort / flip / cast）
description: Triton-Ascend 向量操作扩展：张量切片插入/提取、单元素访问、硬件排序与翻转、增强 cast（overflow_mode 与
  fp_downcast_rounding）。
tags: [task.lookup, topic.programming_model.execution_unit.vector, topic.compute.ordering, topic.compute.quantization, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/insert_slice.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T08:55:00Z'
platforms: [a2, a3]
---
# 向量操作扩展（insert_slice / extract_slice / get_element / sort / flip / cast）

标准 Triton API 不具备或功能不同的张量操作扩展，在昇腾 NPU 上有专门硬件支持，通过 SIMD 指令高效执行。

## 与标准 Triton 的对照

| 操作 | 类别 | 标准 Triton 对应 | 主要区别 |
|------|------|-----------------|----------|
| `insert_slice` | 切片 | 无 | 将子张量插入指定偏移位置 |
| `extract_slice` | 切片 | 无 | 从指定偏移提取子张量 |
| `get_element` | 元素访问 | 无 | 按索引取单个元素 |
| `sort` | 排序 | 无 | 硬件加速，**仅支持最后一维** |
| `flip` | 翻转 | 无 | 硬件加速，SIMD/SIMT 双实现 |
| `cast` | 类型转换 | `tl.cast` | 支持 `overflow_mode` 和 `fp_downcast_rounding` |

## API 参考

### insert_slice / extract_slice

```python
@_tensor_member_fn
@builtin
def insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor
```

- `offsets`/`sizes`/`strides`（`tuple[int]`）：各维偏移/大小/步长，长度必须等于张量维度数。
- 约束：`ful` 与 `sub` 维度数相同；所有 `sizes >= 1`；所有 `strides >= 0`。
- `extract_slice` 返回形状为 `sizes` 的子张量。
- `offsets` 支持编译期常量（`constexpr`）与运行时张量（动态偏移场景）。
- 与 Python 切片（`tensor[0:4]` 语法糖）的区别：显式指定 offsets/sizes/strides，直接映射硬件指令。

### get_element

```python
def get_element(src, indice, _builder=None, _generator=None)
```

`indice` 为索引元组，长度必须等于张量维度数；返回标量张量（元素类型同源张量）。

### sort

```python
@builtin
def sort(ptr, dim=-1, descending=False, _builder=None)
```

- **仅支持沿最后一维排序**（硬件排序单元限制）；其他维度需先转置、排序、再转置回。
- 支持 dtype：int8/int16/int32/int64/fp16/bf16/fp32/fp8e4nv/fp8e5。
- int8/int16 排序结果自动附加 `overflow_mode="saturate"` 提示（排序中间值可能超原始范围）。

### flip

```python
@builtin
def flip(ptr, dim=-1, _builder=None, _generator=None)
```

- **SIMD 模式**：直接调硬件 flip 指令，支持任意维度和形状。
- **SIMT 模式**：经 reshape + xor_swap 实现，**要求翻转维度大小为 2 的幂**。

### cast（增强版类型转换）

```python
@_tensor_member_fn
@builtin
def cast(input, dtype, fp_downcast_rounding=None, bitcast=False, overflow_mode=None, _builder=None)
```

| 参数 | 取值 | 说明 |
|------|------|------|
| `fp_downcast_rounding` | `"rtne"`（默认）/ `"rtz"` | 浮点下舍入：最近偶数 / 向零 |
| `overflow_mode` | `"trunc"`（默认）/ `"saturate"` | 整数转换溢出：截断 / 饱和到最大最小值 |
| `bitcast` | `False`/`True` | 是否按位转换 |

**与标准 tl.cast 的区别**：支持 `overflow_mode` 与 `fp_downcast_rounding`；内部针对昇腾优化（bf16 中转、fp8 支持等）。

**平台差异（saturate 模式）**：910_95 上直接用硬件 `int_cast` 指令，并经 `compile_hint` 附加 `saturate_src_unsigned`/`saturate_dst_unsigned` 告知符号性；非 910_95 平台经 fp32 中转实现，精度可能略差。

## 代码示例

### insert_slice / extract_slice

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def slice_ops_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
    sub = al.extract_slice(data, offsets=[0, 0], sizes=[BLOCK_M // 2, BLOCK_N], strides=[1, 1])
    full = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    result = al.insert_slice(full, sub, offsets=[0, 0], sizes=[BLOCK_M // 2, BLOCK_N], strides=[1, 1])
    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], result)
```

### get_element / sort / flip

```python
elem = al.get_element(data, indice=[0, 0])
sorted_desc = al.sort(data, dim=-1, descending=True)
flipped = al.flip(data, dim=-1)
```

### cast 带 overflow_mode 与 fp_downcast_rounding

```python
result_sat  = al.cast(data, tl.int8, overflow_mode="saturate")   # 溢出钳位
result_rtz  = al.cast(data, tl.float16, fp_downcast_rounding="rtz")
```

## NPU 适配要点

1. **bf16 中转**：bf16 到非 fp32 类型的转换需先转 fp32 再转目标类型（硬件不直接支持）。
2. **fp8/fp64 限制**：非 910_95 平台不支持 fp8 和 fp64。
3. **saturate 使用场景**：量化中从大范围整数（int32）转小范围（int8）时避免溢出异常值。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [vec_ops.py#L47-L92](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L47-L92)
- [vec_ops.py#L95-L137](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L95-L137)
- [vec_ops.py#L139-L177](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L139-L177)
- [vec_ops.py#L316-L397](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L316-L397)
- [vec_ops.py#L179-L277](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L179-L277)
- [vec_ops.py#L524-L562](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L524-L562)
- [vec_ops.py#L400-L522](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L400-L522)
