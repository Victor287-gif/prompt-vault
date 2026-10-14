---
type: Guide
title: Triton Ascend API 参考
description: def kernel_function(...):。
tags: [task.implementation, topic.implementation.masking, topic.tiling, topic.compute.elementwise.math_function, topic.data_movement.memory_access, topic.implementation.atomic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/README.md
  role: supporting
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Triton Ascend API 参考

## 2. API 参考

### 2.1 装饰器与启动

| API | 说明 |
|-----|------|
| `@triton.jit` | 将 Python 函数编译为 Triton 内核 |
| `tl.program_id(axis)` | 当前程序在 grid 指定维度上的 ID |
| `tl.num_programs(axis)` | grid 在指定维度上的程序总数 |
| `triton.cdiv(x, y)` | 向上取整除法：`(x + y - 1) // y` |

**@triton.jit 示例**：
```python
@triton.jit
def kernel_function(...):
    pass
```
- **作用**: 将 Python 函数编译为硬件内核
- **约束**: 函数内部不能使用 `return`、`break`、`continue` 语句

**tl.program_id(axis) 示例**：
```python
pid = tl.program_id(axis)  # axis: 0, 1, or 2
```
- **参数**: `axis` - 维度轴 (0, 1, 2)
- **返回**: 当前程序在该轴上的 ID
- **用途**: 确定当前程序块处理的数据范围

**tl.num_programs(axis) 示例**：
```python
num_pids = tl.num_programs(axis)  # axis: 0, 1, or 2
```
- **参数**: `axis` - 维度轴 (0, 1, 2)
- **返回**: 该轴上的总程序数
- **用途**: 计算网格大小和边界条件

**triton.cdiv(a, b) 示例**：
```python
grid_size = triton.cdiv(total_elements, block_size)
```
- **参数**: `a`, `b` - 被除数和除数
- **返回**: 向上取整的除法结果
- **用途**: host侧使用，计算启动网格大小

### 2.2 内存访问

| API | 说明 |
|-----|------|
| `tl.load(ptr, mask=None, other=0.0, boundary_check=None)` | 从全局内存加载数据 |
| `tl.store(ptr, value, mask=None, boundary_check=None)` | 将数据写入全局内存 |
| `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` | 创建块指针，用于 2D/高维块访问 |
| `tl.advance(block_ptr, dim, delta)` | 沿指定维度推进块指针 |

**tl.load 示例:**
```python
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```
- **参数**:
  - `pointer`: 内存指针
  - `mask`: 布尔掩码，True 表示有效位置
  - `other`: 掩码为 False 时的默认值
  - `boundary_check`: 边界检查维度 (0, 1) 或 None
- **返回**: 加载的张量数据
- **用途**: 从全局内存加载数据

**tl.store 示例:**
```python
tl.store(ptr + offsets, result, mask=mask)
```
- **参数**:
  - `pointer`: 内存指针
  - `value`: 要存储的值
  - `mask`: 布尔掩码，True 表示有效位置
  - `boundary_check`: 边界检查维度 (0, 1) 或 None
- **用途**: 将数据存储到全局内存

**tl.make_block_ptr 示例:**
```python
p_q = tl.make_block_ptr(
    base=q + (bos * HQ + i_hq) * K,
    shape=(T, K),
    strides=(HQ*K, 1),
    offsets=(i_t * BT, 0),
    block_shape=(BT, BK),
    order=(1, 0)
)
val = tl.load(p_q, boundary_check=(0, 1), padding="zero")
```
- **参数**:
  - `base`: 基础内存指针
  - `shape`: 完整张量的形状
  - `strides`: 每个维度的步长
  - `offsets`: 当前块的起始偏移
  - `block_shape`: 当前块的大小
  - `order`: 内存布局顺序 (1, 0) 表示行主序
- **返回**: 块指针对象
- **用途**: 高效访问 2D 数据块

**tl.advance 示例:**
```python
block_ptr = tl.advance(block_ptr, (BLOCK_M, 0))
```
- **参数**:
  - `ptr`: 块指针
  - `offsets`: 各维度的偏移量
- **返回**: 移动后的块指针
- **用途**: 移动块指针到下一个位置

### 2.3 张量构造与索引

| API | 说明 |
|-----|------|
| `tl.arange(start, end)` | 创建一维索引张量 |
| `tl.zeros(shape, dtype)` | 创建全零张量 |
| `tl.full(shape, value, dtype)` | 创建填充指定值的张量 |
| `tl.cast(tensor, dtype)` | 类型转换 |

**tl.arange 示例:**
```python
offsets = tl.arange(0, BLOCK_SIZE)
```
- **参数**: `start`, `end` - 起始和结束值
- **返回**: 连续整数序列
- **用途**: 创建索引序列

**tl.zeros 示例:**
```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```
- **参数**:
  - `shape`: 张量形状
  - `dtype`: 数据类型
- **返回**: 全零张量

**tl.full 示例:**
```python
ones = tl.full((M, N), 1.0, dtype=tl.float32)
```
- **参数**:
  - `shape`: 张量形状
  - `value`: 填充值
  - `dtype`: 数据类型
- **返回**: 填充指定值的张量

**tl.cast 示例:**
```python
float_data = tl.cast(int_data, tl.float32)
```
- **参数**:
  - `input`: 输入张量
  - `dtype`: 目标数据类型
- **返回**: 类型转换后的张量

### 2.4 数学与归约

| API | 说明 |
|-----|------|
| `tl.cdiv(a, b)` | 向上取整除法 |
| `tl.dot(a, b)` | 矩阵乘法 / 点积 |
| `tl.sum(tensor, axis)` | 沿轴求和 |
| `tl.max(tensor, axis)` | 沿轴求最大值 |
| `tl.where(condition, x, y)` | 条件选择 |

**tl.cdiv 示例:**
```python
result = tl.cdiv(offset, BLOCK_SIZE)
```
- **参数**: `a`, `b` - 被除数和除数
- **返回**: 向上取整的除法结果 ⌈a/b⌉
- **用途**: kernel内部使用，计算向上整除结果，等价于 `(a + b - 1) // b`

**tl.dot 示例:**
```python
result = tl.dot(a, b, acc=accumulator)
```
- **参数**:
  - `a`, `b`: 输入矩阵
  - `acc`: 累加器 (可选)
  - `allow_tf32`: 是否允许 TF32 精度
- **返回**: 矩阵乘法结果
- **用途**: 核心矩阵乘法操作

**tl.sum 示例:**
```python
block_sum = tl.sum(data, axis=0)
```
- **参数**:
  - `tensor`: 输入张量
  - `axis`: 归约轴
- **返回**: 归约结果

**tl.max 示例:**
```python
max_val = tl.max(data, axis=0)
```
- **参数**:
  - `tensor`: 输入张量
  - `axis`: 归约轴
- **返回**: 最大值

**tl.where 示例:**
```python
result = tl.where(mask, data, 0.0)
```
- **参数**:
  - `condition`: 条件张量
  - `x`, `y`: 选择值
- **返回**: 根据条件选择的值
- **用途**: SIMD 友好的条件选择

### 2.5 扫描与原子操作

| API | 说明 |
|-----|------|
| `tl.cumsum(tensor, axis)` | 沿轴累积和 |
| `tl.cumprod(tensor, axis)` | 沿轴累积积 |
| `tl.atomic_add(ptr, value)` | 原子加 |
| `tl.atomic_max(ptr, value)` | 原子最大值 |

**tl.cumsum 示例:**
```python
cumulative_sum = tl.cumsum(data, axis=0)
reverse_cumsum = tl.cumsum(data, axis=1, reverse=True)
```
- **参数**:
  - `tensor`: 输入张量
  - `axis`: 累积求和的轴 (默认为 0)
  - `reverse`: 是否反向累积 (默认为 False)
- **返回**: 累积求和结果张量
- **用途**: 计算沿指定轴的累积和，常用于前缀和计算

**tl.cumprod 示例:**
```python
cumulative_prod = tl.cumprod(data, axis=0)
reverse_cumprod = tl.cumprod(data, axis=1, reverse=True)
```
- **参数**:
  - `tensor`: 输入张量
  - `axis`: 累积乘积的轴 (默认为 0)
  - `reverse`: 是否反向累积 (默认为 False)
- **返回**: 累积乘积结果张量
- **用途**: 计算沿指定轴的累积乘积，常用于概率计算和序列处理

**tl.atomic_add 示例:**
```python
tl.atomic_add(output_ptr, block_sum)
```
- **参数**:
  - `ptr`: 目标内存指针
  - `value`: 要添加的值
- **用途**: 线程安全的加法操作

**tl.atomic_max 示例:**
```python
tl.atomic_max(max_ptr, local_max)
```
- **参数**:
  - `ptr`: 目标内存指针
  - `value`: 要比较的值
- **用途**: 线程安全的最大值更新

### 2.6 编译时常量

| API | 说明 |
|-----|------|
| `tl.constexpr` | 标记参数为编译时常量，用于 kernel 参数 |

```python
def kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
BLOCK_SIZE: tl.constexpr = 1024
```
- **用途**: 标记编译时常量参数
- **约束**: 必须在函数签名中声明

### 2.7 布局与分块

| API | 说明 |
|-----|------|
| `tl.swizzle2d(i, j, size_i, size_j, size_g)` | 2D 分块时的 swizzle 布局优化 |

**tl.swizzle2d 示例:**
```python
task_i, task_j = tl.swizzle2d(block_i, block_j, NUM_BLOCKS_I, NUM_BLOCKS_J, GROUP_SIZE)
```
- **参数**:
  - `i`, `j`: 原始块索引
  - `size_i`, `size_j`: 总块数
  - `size_g`: 分组大小(通常为2/4/8)
- **返回**: 重排后的块索引 (task_i, task_j)
- **用途**: 2D块重排,提升缓存局部性
- **适用场景**: 矩阵乘法等多维块计算,改善数据复用
- **注意**: 仅支持行优先(i方向)分组,列优先需手动实现

### 2.8 libdevice接口

Triton-Ascend libdevice 提供了一系列已优化的基础数学函数库。**需要此类函数时，优先检查 libdevice 是否已有实现。一旦命中下面的接口，一定要使用libdevice接口**

libdevice 已覆盖的函数包括：

- **数学运算**：`round`、`trunc`、`nearbyint`、`pow`、`reciprocal`
- **三角函数**：`tan`、`atan`、`atan2`、`acos`、`asin`
- **双曲函数**：`sinh`、`cosh`、`tanh`、`asinh`、`acosh`、`atanh`
- **指数对数**：`expm1`、`log1p`、`log10`
- **激活函数**：`relu`
- **特殊函数**：`erfinv`、`gamma`、`lgamma`、`cyl_bessel_i0`
- **浮点判断与操作**：`isnan`、`isinf`、`signbit`、`nextafter`、`ldexp`、`ilogb`、`copysign`、`hypot`

如果命中上述接口，请务必加载各函数的详细路径示例、支持类型与使用说明见：`latency-optimizer/references/libdevice-usage.md`，并使用对应接口
---
