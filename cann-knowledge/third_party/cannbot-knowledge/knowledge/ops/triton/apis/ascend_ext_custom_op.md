---
type: API
title: 自定义算子注册与调用（register_custom_op / custom）
description: 通过 @register_custom_op 注册 CANN bitcode 自定义算子并以 al.custom() 调用；含 4 个内置索引算子（index_select/index_put/gather_load/scatter_store）的参数约束。
tags: [task.lookup, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.cube, topic.compute.indexing, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/custom_op.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# 自定义算子注册与调用（register_custom_op / custom）

Triton-Ascend 自定义算子机制：通过 `@register_custom_op` 注册算子类，在 kernel 中以 `al.custom(name, *args, **kwargs)` 调用，用于将 CANN 自定义算子（bitcode 形式）集成进 Triton kernel。另内置 4 个 `__builtin_*` 索引算子，无需注册直接用。

## 注册流程与算子类属性

```
1. 定义算子类（设置 name/core/pipe/mode 等属性）
2. @register_custom_op 装饰器注册
3. kernel 中 al.custom(name, *args, out=..., **kwargs) 调用
```

| 属性 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | `str` | 是 | 算子名，全局唯一；未设置时用类名 |
| `core` | `CORE` | 是 | 执行核心（VECTOR/CUBE 等） |
| `pipe` | `PIPE` | 是 | 执行流水线（PIPE_V/PIPE_M 等） |
| `mode` | `MODE` | 是 | 执行模式（SIMD/SIMT/MIX） |
| `symbol` | `str` | 非内置必需 | CANN bitcode 中的符号名 |
| `bitcode` | `str` | 非内置必需 | bitcode 文件路径（转绝对路径，须存在） |
| `__init__` | 方法 | 推荐 | 参数校验和初始化 |
| `iterator_types` | `List[IteratorType]` | 可选 | 迭代器类型（配 indexing_map） |
| `indexing_map` | `AffineMap` | 可选 | 循环嵌套与张量维度的仿射映射 |
| `extra_buffers` | `tuple` | 可选 | 额外缓冲区规格（类型, 大小），如 `(tl.float32, 1024)` |
| `align_dim` | `dict` | 可选 | 参数对齐维度约束，如 `{'input': 16}` |
| `extra_attr` | `str` | 可选 | 额外编译器属性字符串 |

**注册校验**：必须是类；算子名不可重复；`core/pipe/mode` 必须设置；非内置算子（名不以 `__builtin_` 开头）必须有 `symbol` 和 `bitcode`。

## custom 调用与参数类型转换

```python
@builtin
def custom(name: str, *args, _builder=None, **kwargs) -> tl.tensor | tuple[tl.tensor, ...] | None
```

- 输出张量通过 `kwargs` 的 `out` 传入；多输出传 list，返回对应 tuple。
- 参数类型转换：`tl.tensor` 直接用 handle；Python `int` 默认转 **int32**，需 int64 时用 `al.int64(x)` 包装；`float` 默认转 fp32；`tl.constexpr` 自动解包。
- `self.arg_type` 字典可覆盖参数类型推断，如 `self.arg_type['src_stride'] = index.dtype`。
- 注册后 `tl.dtype` 获得 `cname` 属性（C 类型名）：fp16→`half`、bf16→`bfloat16_t`、fp32→`float`、fp64→`double`、fp8e4nv→`float8_e4m3_t`、int8→`int8_t` 等，便于生成 bitcode 声明。

## 内置自定义算子（SIMT，无需注册）

| 算子名 | 方向 | 源/值 rank | 说明 |
|--------|------|-----------|------|
| `__builtin_index_select` | GM → UB | src 2D-5D | 按索引选择 |
| `__builtin_index_put` | UB → GM | value 2D-5D | 按索引写入 |
| `__builtin_gather_load` | GM → UB | index 1D-5D | 索引收集加载 |
| `__builtin_scatter_store` | UB → GM | index 2D-5D | 索引散列存储 |

均为 `core=VECTOR / pipe=PIPE_V / mode=SIMT`。

**与 SIMD 版本的区别**：SIMD 版本（如 `al.index_select_simd`）为数据级并行、自动返回结果张量、参数更高层（`src_shape/src_offset/read_shape`）；内置 SIMT 版本需 `out` 预分配、用 `bound/src_stride/dst_stride` 底层参数，适合控制流密集、动态索引场景。

### __builtin_index_select 调用形式

```python
al.custom('__builtin_index_select',
    src,           # GM 源张量指针
    index,         # UB 索引张量（整数，1D 或 2D）
    dim,           # 选择维度，0 <= dim < src_rank
    bound,         # 索引上界（al.int64）
    end_offset,    # 索引各维结束偏移（tuple）
    start_offset,  # 源各维起始偏移（tuple）
    src_stride,    # 源各维步长（tuple）
    other=None,    # 越界默认值（可选）
    out=output)    # 输出（必需，UB），dtype 与 src 元素类型一致
```

约束：`len(start_offset)==len(src_stride)`；`len(end_offset)==index_rank+len(start_offset)-1`。

### __builtin_index_put 调用形式

```python
al.custom('__builtin_index_put',
    dst, index, value, dim,   # dim: 0 <= dim < value_rank - 1
    bound,                    # al.int64
    dst_shape, dst_offset, dst_stride)  # 三者等长 tuple
```

### __builtin_gather_load / __builtin_scatter_store 调用形式

```python
al.custom('__builtin_gather_load',
    src, index, bound, dim,   # index rank 1-5；dim < index_rank
    src_stride,               # al.int64
    index_shape, offsets,     # 与 src_stride 等长 == index_rank
    out=output)               # out.shape == index.shape

al.custom('__builtin_scatter_store',
    dst, value, index, bound, dim,
    dst_stride, index_shape, offsets)   # 三者等长 == index_rank
```

## 代码示例

### 注册并使用自定义算子

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_elementwise_op:
    name = 'my_elementwise_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = 'my_elementwise_kernel'
    bitcode = '/path/to/my_op.bc'

    def __init__(self, input, output):
        assert input.dtype == output.dtype

@triton.jit
def my_kernel(in_ptr, out_ptr, N):
    x = tl.load(in_ptr + tl.arange(0, 128))
    result = al.custom('my_elementwise_op', x, out=tl.zeros([128], dtype=tl.float32))
    tl.store(out_ptr + tl.arange(0, 128), result)
```

### 使用内置 index_select

```python
@triton.jit
def index_select_kernel(src_ptr, index_ptr, out_ptr, N, dim, BLOCK: tl.constexpr):
    index = tl.load(index_ptr + tl.arange(0, BLOCK))
    out = tl.zeros([BLOCK, 16], dtype=tl.float32)
    result = al.custom(
        '__builtin_index_select',
        src_ptr, index, dim=0,
        bound=al.int64(N),
        end_offset=(BLOCK, 16),
        start_offset=(0, 0),
        src_stride=(N, 1),
        out=out)
    tl.store(out_ptr + tl.arange(0, BLOCK)[:, None] * 16 + tl.arange(0, 16)[None, :], result)
```

## NPU 适配要点

1. **内置算子无需 bitcode**：`__builtin_` 开头的算子实现由编译器内置提供。
2. **调试**：可设 `source` 属性指定源码路径、`compile` 属性指定编译选项。
3. **bitcode 生成**：需用 CANN 算子开发工具链将 C/C++ 算子编译为 bitcode，参考 CANN 自定义算子开发文档。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [custom_op.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/custom_op.py)
- [builtin_custom_ops.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/builtin_custom_ops.py)
- [core.py#L93-L101](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L93-L101)
