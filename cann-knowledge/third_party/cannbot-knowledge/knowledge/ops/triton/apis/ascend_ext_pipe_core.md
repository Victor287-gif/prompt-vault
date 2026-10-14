---
type: API
title: PIPE / CORE / MODE / IteratorType 枚举
description: 昇腾 NPU 硬件流水线与核心类型枚举：PIPE、CORE、MODE、IteratorType 的取值、语义与在自定义算子和同步操作中的用法。
tags: [task.lookup, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.pipeline, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---
# PIPE / CORE / MODE / IteratorType 枚举

昇腾 NPU 采用 Cube-Vector 分离架构，每个 AI Core 含 1 个 Cube 核心（矩阵计算）和多个 Vector 核心（向量计算）。不同操作在不同硬件流水线（PIPE）上执行；这些枚举在自定义算子注册（`register_custom_op`）、同步操作（`sync_block_set/wait`）和代码生成中起关键作用。

## PIPE 枚举：硬件流水线类型

| 枚举值 | 硬件单元 | 数据通路 | 说明 |
|--------|----------|----------|------|
| `PIPE_S` | Scalar 标量单元 | 标量寄存器 | 标量运算（整数加减、比较、分支） |
| `PIPE_V` | Vector 向量单元 | Vector 计算流水线 | 逐元素运算、激活函数 |
| `PIPE_M` | Cube 矩阵单元 | Cube 计算流水线 | 矩阵乘法（MatMul） |
| `PIPE_MTE1` | MTE1 搬运引擎 | GM → L1 | GM 到 L1 缓存搬运 |
| `PIPE_MTE2` | MTE2 搬运引擎 | GM → UB | GM 到 Unified Buffer 搬运 |
| `PIPE_MTE3` | MTE3 搬运引擎 | UB → GM | UB 到 GM 搬运 |
| `PIPE_FIX` | Fixpipe 流水线 | L0C → GM/L1/UB | Cube 结果搬出；L0C→UB 仅 910_95 |
| `PIPE_ALL` | 全部流水线 | - | 操作涉及所有流水线 |

## CORE 枚举：核心类型

| 枚举值 | 说明 | 使用场景 |
|--------|------|----------|
| `CORE.VECTOR` | 在 Vector 核心上执行 | 向量运算、逐元素操作 |
| `CORE.CUBE` | 在 Cube 核心上执行 | 矩阵乘法 |
| `CORE.CUBE_OR_VECTOR` | 可在 Cube 或 Vector 上执行 | 编译器自动选择 |
| `CORE.CUBE_AND_VECTOR` | 需同时使用 Cube 和 Vector | Cube-Vector 融合算子，需配合 `scope` 与同步操作 |

## MODE 枚举：执行模式

| 枚举值 | 说明 | 特点 |
|--------|------|------|
| `MODE.SIMD` | 单指令多数据 | 向量化并行，纯数据并行时性能更高 |
| `MODE.SIMT` | 单指令多线程 | 线程级并行，适合控制流密集/动态索引操作 |
| `MODE.MIX` | 混合模式 | SIMD 与 SIMT 混合 |

## IteratorType 枚举：迭代器语义

用于自定义算子的 `indexing_map` 中标注各维度迭代方式：

`Parallel`（并行）、`Broadcast`（广播）、`Transpose`（转置）、`Reduction`（归约）、`Interleave`（交错）、`Deinterleave`（解交错）、`Inverse`（逆序）、`Pad`（填充）、`Concat`（拼接）、`Gather`（收集）、`Cumulative`（累积）、`Opaque`（不透明）。

## 代码示例

### 在自定义算子注册中使用 PIPE/CORE/MODE

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_vector_op:
    name = 'my_vector_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, input, output):
        pass

@triton.jit
def my_kernel(in_ptr, out_ptr, N):
    x = tl.load(in_ptr + tl.arange(0, 128))
    result = al.custom('my_vector_op', x, out=tl.zeros([128], dtype=tl.float32))
    tl.store(out_ptr + tl.arange(0, 128), result)
```

### 在同步操作中显式指定 PIPE

```python
@triton.jit
def cube_vector_sync_kernel():
    with al.scope(core_mode="cube"):
        result = tl.dot(a, b)
        al.sync_block_set("cube", "vector", 0,
                          sender_pipe=al.PIPE.PIPE_FIX,
                          receiver_pipe=al.PIPE.PIPE_MTE2)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0,
                           sender_pipe=al.PIPE.PIPE_FIX,
                           receiver_pipe=al.PIPE.PIPE_MTE2)
        processed = result * 2.0
```

### IteratorType 在自定义算子中的使用

```python
@al.register_custom_op
class my_matmul_reduce:
    name = 'my_matmul_reduce'
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD
    iterator_types = [al.IteratorType.Parallel, al.IteratorType.Parallel, al.IteratorType.Reduction]
    indexing_map = al.affine_map(...)
```

## NPU 适配要点

1. **PIPE 选择对性能至关重要**：错误标注会导致编译器生成低效代码或编译失败。向量操作必须标 `PIPE_V`，矩阵操作必须标 `PIPE_M`。
2. **数据通路**：GM 片外大容量高延迟；L1 片上缓存 Cube 专用；UB 片上存储 Vector 专用；L0C 是 Cube 输出缓冲区。
3. **同步操作中的 PIPE 默认值**（不指定时按方向自动推断）：
   - cube → vector：`sender_pipe=PIPE_FIX`，`receiver_pipe=PIPE_MTE2`
   - vector → cube：`sender_pipe=PIPE_MTE3`，`receiver_pipe=PIPE_MTE2`
4. **PIPE_M vs PIPE_FIX**：`PIPE_M` 执行矩阵乘法计算；`PIPE_FIX` 负责把 Cube 结果从 L0C 搬到 GM/L1/UB，是 Cube 侧的不同阶段。A2/A3 上 fixpipe 仅支持 L0C→GM/L1，910_95 额外支持 L0C→UB。
5. **SIMT vs SIMD**：复杂控制流（条件分支、动态索引）用 SIMT；纯数据并行用 SIMD。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py)
- [semantic.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/semantic.py)
