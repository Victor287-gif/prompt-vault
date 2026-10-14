---
type: API
title: parallel / compile_hint / multibuffer 辅助操作
description: 三种编译器导向辅助操作：al.parallel 扩展并行迭代（bind_sub_block）、compile_hint 附加编译提示元数据、multibuffer
  双缓冲流水线。
tags: [task.lookup, topic.implementation.pipeline, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/aux_ops.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# parallel / compile_hint / multibuffer 辅助操作

三种不直接参与计算、通过向编译器传递元数据影响代码生成与优化的辅助操作：`parallel`（扩展并行迭代）、`compile_hint`（编译提示）、`multibuffer`（ping-pong 双缓冲）。

## parallel — 并行范围声明

| 特性 | tl.range | al.parallel |
|------|----------|-------------|
| 基本功能 | 范围迭代 | 范围迭代 |
| bind_sub_block | 不支持 | **支持**（多子 Vector 核参与迭代） |
| num_stages | 支持 | 支持 |
| loop_unroll_factor | 支持 | 支持 |

```python
class parallel(range):
    def __init__(self, arg1, arg2=None, step=None, num_stages=None,
                 loop_unroll_factor=None, bind_sub_block: bool = False):
```

`bind_sub_block=True` 告知编译器将循环迭代分配给多个子 Vector 核（每个 AI Core 含 2 个 Vector Core），适用于所有支持多子 Vector 核的 Ascend 平台；不设该参数时与 `tl.range` 功能相同。

## compile_hint — 编译提示

```python
@builtin
def compile_hint(ptr, hint_name, hint_val=None, _builder=None):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | `tensor` | 需要附加提示的张量 |
| `hint_name` | `str` 或 `constexpr` | 提示名称（必须为字符串） |
| `hint_val` | `None`/`bool`/`int`/`constexpr`/`list[int]` | 提示值（list 仅支持整数数组） |

| hint_val 类型 | IR 表示 |
|---------------|---------|
| `None` | `unit_attr`（仅标记存在） |
| `bool` | `bool_attr` |
| `int` | `int32_attr` |
| `constexpr`(str) | `str_attr` |
| `list[int]` | `i64_array_attr` |

常用 hint：

| hint_name | 值类型 | 说明 |
|-----------|--------|------|
| `"hivm.multi_buffer"` | `int` | 多缓冲标记（multibuffer 内部使用） |
| `"overflow_mode"` | `str` | 溢出处理（`"saturate"`/`"trunc"`），整数转换 |
| `"mem_unique"` | `None` | 核心独占内存标记 |
| `"saturate_src_unsigned"` / `"saturate_dst_unsigned"` | `bool` | saturate 模式下源/目标类型符号性（cast 自动设置） |
| `"break_vf"` | `None` | 打断 Vector Fusion，防过长 VF 融合导致 UB 溢出 |

**限制**：SIMT 模式下不支持 hint 标注（直接跳过）；同一张量可多次标注不同名称。

## multibuffer — 多缓冲（双缓冲）

```python
@builtin
def multibuffer(src: tensor, size, _builder=None):
```

| 参数 | 说明 |
|------|------|
| `src` | 需要多缓冲设置的张量 |
| `size` | 缓冲区副本数量，**当前仅支持 2** |

原理：创建两个缓冲区副本，一个计算时另一个并行搬运，隐藏搬运延迟：

```
单缓冲：[计算1][--等待--][计算2][--等待--]
双缓冲：[计算1][计算2][计算3]...
        [搬运1][搬运2][搬运3]...
```

实现上是 `compile_hint(src, "hivm.multi_buffer", size)` 的语法糖。适用场景：循环中数据加载与计算交替，计算当前批时预加载下一批。

## 代码示例

### parallel 带 bind_sub_block

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def parallel_sub_block_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    with al.scope(core_mode="vector"):
        vec_id = al.sub_vec_id()
        for i in al.parallel(0, N, BLOCK_SIZE, bind_sub_block=True):
            x = tl.load(in_ptr + i + tl.arange(0, BLOCK_SIZE))
            result = tl.sigmoid(x)
            tl.store(out_ptr + i + tl.arange(0, BLOCK_SIZE), result)
```

### compile_hint 多种值类型

```python
@triton.jit
def compile_hint_basic_kernel(in_ptr, out_ptr, xnumel,
                              XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr + xindex, xmask)

        al.compile_hint(tmp0, "hint_a")                  # unit_attr
        al.compile_hint(tmp0, "hint_b", 42)              # int
        al.compile_hint(tmp0, "hint_c", True)            # bool
        al.compile_hint(tmp0, "hint_d", [XBLOCK, XBLOCK_SUB])  # list[int]

        tl.store(out_ptr + xindex, tmp0, xmask)
```

### multibuffer 双缓冲

```python
@triton.jit
def multibuffer_kernel(in_ptr, out_ptr, xnumel,
                       XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr + xindex, xmask)
        al.multibuffer(tmp0, 2)
        tmp2 = tmp0 * 2.0
        tl.store(out_ptr + xindex, tmp2, xmask)
```

### overflow_mode 提示（经 cast 自动附加）

```python
result = al.cast(x, tl.int8, overflow_mode="saturate")
```

## NPU 适配要点

1. **CV 协同中多子 Vector 核必须用 `al.parallel(bind_sub_block=True)`**，`tl.range` 无此能力。
2. **compile_hint 非侵入**：仅加元数据，不改变计算语义，编译器可忽略；移除所有 hint 不应改变结果。类比 C++ 的 `__attribute__` / `#pragma`。
3. **multibuffer 仅支持 size=2**，不支持三缓冲及以上。
4. **SIMT 模式下 compile_hint 无效**。
5. **自定义 hint_name 需后端编译器支持**才生效。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [aux_ops.py#L99-L111](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/aux_ops.py#L99-L111)
- [aux_ops.py#L135-L151](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/aux_ops.py#L135-L151)
- [aux_ops.py#L153-L162](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/aux_ops.py#L153-L162)
- [compile_hint.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/compile_hint.md)
- [multibuffer.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/multibuffer.md)
