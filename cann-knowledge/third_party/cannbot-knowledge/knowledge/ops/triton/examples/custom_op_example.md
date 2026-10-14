---
type: Example
title: 自定义算子示例（register_custom_op / al.custom / builtin ops）
description: 教学示例：通过 cann.extension 注册 NPU 原生 CCE 自定义算子并在 Triton kernel 中调用，含 bitcode/source
  两种方式、indexing_map、extra_buffers 与内置 gather/scatter 算子。
tags: [task.implementation, topic.programming_model.execution_unit.vector, topic.compute.indexing]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/unittest/custom_op/custom_op_demo.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# 自定义算子示例（Custom Op on Triton-Ascend）

将 NPU 原生 CCE 算子接入 Triton kernel 的完整机制：`@al.register_custom_op` 注册 → kernel 内 `al.custom(...)` 调用。API 细节见 [ascend_ext_custom_op](../apis/ascend_ext_custom_op.md)。

## 注册与调用（完整示例）

```python
import os
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@al.register_custom_op
class min_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    symbol = 'min_custom_op_impl'
    bitcode = os.path.abspath(__file__)

@al.register_custom_op
class simple_custom_op:
    name = 'simple_custom_op'

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    symbol = 'simple_custom_op_impl'
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, y, dim=0, out=None):
        assert x.shape == y.shape, "x and y should have same shape"
        assert isinstance(dim, int), "dim should be const integer"
        assert out, "out is required"

@al.register_custom_op
class _example_custom_op:
    name = 'example_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    symbol = 'example_custom_op_impl'
    bitcode = os.path.abspath(__file__)

    def __init__(self, src, index, offset: al.int64, axis, out=None):
        assert isinstance(src, tl.tensor), "src should be tensor"
        assert index.dtype.is_int(), "index should be integer tensor"
        assert isinstance(offset, int), "offset should be integer"
        assert isinstance(axis, int), "axis should be integer"
        assert isinstance(out, tuple) and len(out) == 2, "out should be tuple of 2 items"

        rank = len(index.shape)
        self.symbol = f"{self.name}_{rank}d_{src.dtype.cname}_{index.dtype.cname}"

        self.source = f"workspace/example_custom_op_impl.cce"
        self.compile = "bisheng -O2 -std=c++17 -o $@ -c $<"

        self.arg_type['axis'] = index.dtype

@al.builtin
def example_op(src, index, offset, axis, _builder=None):
    x = tl.semantic.full(src.shape, 0, tl.float32, _builder)
    y = tl.semantic.full(index.shape, 0, tl.float32, _builder)
    return al.custom_semantic(_example_custom_op.name,
        src, index, offset, axis, out=(x, y), _builder=_builder)

@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    y = al.custom("min_custom_op", x, x_ptr, y_ptr + i, al.int64(0), (1, 2, 3), [4.1, 5.2], out=y)
    y = al.custom("simple_custom_op", x, y, dim=1, out=y)
    index = tl.full((2, 3), 0, tl.int64)
    x, y = al.custom("example_custom_op", x, index, offset=1, axis=0, out=(x, y))
    result, _ = example_op(x, index, offset=2, axis=1)
    tl.store(out_ptr + i, result, mask=i < n)
```

## 要点讲解

- **实现提供方式（三选一）**：① `bitcode = <目标文件路径>`，编译器按 `symbol` 找函数实现；② `source` + `compile`（CCE 源码 + bisheng 编译命令模板，`$@` 输出 / `$<` 输入）；③ `__init__` 中按参数动态拼 `symbol`（如 `{name}_{rank}d_{dtype}_{index_dtype}`）。
- **必需属性**：`core`（VECTOR/CUBE）、`pipe`（PIPE_MTE2/PIPE_V/PIPE_M/PIPE_FIX）、`mode`（SIMD/SIMT）、`symbol`、`bitcode`。
- **可选属性**：`name`（默认类名）、`indexing_map`（affine map 描述访存模式）、`extra_buffers`（scratch buffer，`(dtype, element_count)` 元组列表，编码进 HIVM MLIR 的 `extra_buffers_sizes` 属性）、`source`/`compile`/`arg_type`。
- **`@al.builtin` 包装**：用 `al.custom_semantic` 把 custom op 包装成类内置函数调用（返回 out tuple）。
- **调试**：报 "symbol not found" 先核对 `bitcode` 路径与 CCE 函数名；可用 `ASTSource` + `ttir_to_linalg` 手动编译 dump IR 定位。

## 内置自定义算子（gather/scatter 族）

```python
@triton.jit
def builtin_ops_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    index = tl.full([8], 0, tl.int32)
    value = tl.full([8, 64], 0, tl.float32)
    tmp = tl.full([8], 0, tl.float32)

    x = al.custom("__builtin_index_select",
                   x_ptr, index,
                   dim=0, bound=100,
                   end_offset=(2, 2), start_offset=(0, 0),
                   src_stride=(4, 1), out=x)

    al.custom("__builtin_index_put",
              x_ptr, index, value,
              dim=0, bound=12,
              dst_shape=(1, 2, 3),
              dst_offset=(4, 5, 6),
              dst_stride=(8, 4, 1))

    tmp = al.custom("__builtin_gather_load",
              y_ptr, index,
              bound=100, dim=0,
              src_stride=(1,),
              index_shape=(3,),
              offsets=(0,),
              out=tmp)

    al.custom("__builtin_scatter_store",
              out_ptr, value, index,
              1, 0, (1, ), (2, ), (1, ))

    y = al.custom("__builtin_indirect_load", x_ptr, index, mask=i < n, other=y, out=y)
    al.custom("__builtin_indirect_store", out_ptr, index, value)
    tl.store(out_ptr + i, y, mask=i < n)
```

| 内置算子 | 功能 |
|--------|------|
| `__builtin_index_select` | 按索引选择元素 |
| `__builtin_index_put` | 按索引放置元素 |
| `__builtin_gather_load` | 聚合加载 |
| `__builtin_scatter_store` | 散射存储 |
| `__builtin_indirect_load` / `__builtin_indirect_store` | 间接加载 / 间接存储 |

自定义算子可与标准 Triton 操作自由组合（如 `x + y` 后进 custom op 再 `tl.where`）。注意 NPU 原子操作（`atomic_or/xor/and/xchg/cas`）暂不支持在 loop 中使用。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [custom_op_demo.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/unittest/custom_op/custom_op_demo.py)
- [builtin_ops_demo.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/unittest/custom_op/builtin_ops_demo.py)
- [custom_op_extra_buffer_demo.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/unittest/custom_op/custom_op_extra_buffer_demo.py)
- [custom_op_indexing_map_complex_demo.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/unittest/custom_op/custom_op_indexing_map_complex_demo.py)
