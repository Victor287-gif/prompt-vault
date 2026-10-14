---
type: Optimization
title: compile_hint 与 Ascend 扩展 API 速查
description: triton.language.extra.cann.extension 扩展 API 速查：compile_hint、slice 操作、PropagateNan、sub_vec、custom
  算子、Buffer 模型与同步操作。
tags: [task.optimization, topic.programming_model.execution_unit.vector, topic.implementation.memory_management, topic.implementation.synchronization, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/compile_hint.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---
# compile_hint 与 Ascend 扩展 API 速查

触发场景：打断 Vector Fusion 防 UB 溢出；张量切片；NaN 传播控制；Cube-Vector 同步；注册自定义算子；直接操作片上存储。

导入方式：

```python
import triton.language.extra.cann.extension as extension   # 或简写 al
import triton.extension.buffer.language as bl              # Buffer 编程模型
from triton.language.extra.cann import libdevice           # 数学函数库
```

## API 速查表

| API | 类别 | 说明 |
|-----|------|------|
| `compile_hint` | 编译提示 | 为张量附加编译器元数据 |
| `multibuffer` | 编译提示 | ping-pong 多缓冲，隐藏搬运延迟 |
| `insert_slice` / `extract_slice` / `get_element` | 向量操作 | 子张量插入 / 提取 / 取单元素 |
| `sort` / `flip` / `cast` | 向量操作 | 末维排序（硬件加速）/ 翻转 / 增强类型转换（overflow_mode） |
| `sync_block_set` / `sync_block_wait` / `sync_block_all` | 同步 | 核心间信号与屏障 |
| `sub_vec_id` / `sub_vec_num` | 子向量 | 当前 Vector 核编号（0/1）/ 核数（通常 2） |
| `fixpipe` | Cube-Vector | L0C→UB 搬运（仅 910_95） |
| `custom` / `register_custom_op` | 自定义算子 | 调用 / 注册自定义算子 |
| `bl.alloc` / `bl.to_buffer` / `bl.to_tensor` / `bl.subview` | Buffer 模型 | 缓冲区分配与零拷贝视图转换 |
| `tl.PropagateNan` | 枚举 | NaN 传播策略（NONE / ALL） |

## compile_hint

```python
extension.compile_hint(ptr, hint_name, hint_val=None)
```

| hint_name | hint_val | 说明 |
|-----------|----------|------|
| `"break_vf"` | None | 打断 Vector Fusion，防过长 VF 链导致 UB 溢出 |
| `"hivm.multi_buffer"` | int | 多缓冲标记（multibuffer 内部使用） |
| `"overflow_mode"` | str constexpr | 整数转换溢出控制（"saturate"/"trunc"） |
| `"mem_unique"` | None | 多核并行时核心独占缓冲区 |

Flash Attention 反向中打断 VF 融合链：

```python
ds = p * (dp - d[:, None])
ds = tl.where(mask, ds, 0.0)
ds = ds.to(dtype)
extension.compile_hint(ds, "break_vf")
dq += tl.dot(ds, k)
```

注意：SIMT 模式下 `compile_hint` 无效被跳过；非侵入式不改语义；同一张量可标注多个不同名 hint。

## slice 操作

```python
result = extension.insert_slice(ful, sub, offsets, sizes, strides)
sub = extension.extract_slice(ful, offsets, sizes, strides)
elem = extension.get_element(src, indice)      # indice 长度 = 张量维度数
```

约束：`ful`/`sub` 维度数相同；`offsets`/`sizes`/`strides` 长度等于维度数；`sizes >= 1`，`strides >= 0`；offset 支持 constexpr 与运行时 tensor。

## tl.PropagateNan

| 枚举值 | 说明 | 生成 IR |
|--------|------|---------|
| `PropagateNan.NONE`（默认） | NaN 视为安静值 | `MaxNumFOp`/`MinNumFOp` |
| `PropagateNan.ALL` | 任一操作数 NaN 则结果 NaN | `MaximumFOp`/`MinimumFOp`（IEEE 754） |

适用函数：`tl.maximum`/`tl.minimum`/`tl.clamp`（参数 `propagate_nan`），`tl.max` 归约（参数 `propagate_nan=True/False`）。Flash Attention online softmax 中的配合用法：

```python
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
```

## sub_vec 与 parallel

```python
with extension.scope(core_mode="vector"):
    for i in extension.parallel(0, N, BLOCK_N, bind_sub_block=True):
        offset = i + extension.sub_vec_id() * BLOCK_N
        ...
```

`sub_vec_id()` 返回 `tl.tensor`（int64，0 或 1），仅在 `core_mode="vector"` 作用域内使用；`sub_vec_num()` 返回 `tl.constexpr`（通常 2）。

## custom 算子

```python
@extension.register_custom_op
class my_elementwise_op:
    name = 'my_elementwise_op'          # 全局唯一
    core = extension.CORE.VECTOR
    pipe = extension.PIPE.PIPE_V
    mode = extension.MODE.SIMT
    symbol = 'my_elementwise_kernel'    # 非内置必需
    bitcode = '/path/to/my_op.bc'       # 非内置必需

result = extension.custom('my_elementwise_op', x, out=tl.zeros([128], dtype=tl.float32))
```

内置算子（无需注册）：`__builtin_index_select`、`__builtin_index_put`、`__builtin_gather_load`、`__builtin_scatter_store`。设备端需 int64 时用 `extension.int64(x)` 包装 Python int（默认转 int32）。

## Buffer 模型

```python
ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], extension.ascend_address_space.UB)
data_buf = bl.to_buffer(tensor, extension.ascend_address_space.UB, bind_buffer=ub_buf)
tensor = bl.to_tensor(data_buf, writable=True)
sub_buf = bl.subview(src_buf, offsets=[0, 0], sizes=[M, N], strides=[1, 1])
```

地址空间：`UB` / `L1` / `L0C` / `GM`（`extension.ascend_address_space.*`）。注意：`to_buffer`/`to_tensor` 为零拷贝视图转换；`subview` 偏移须 32B 对齐且所有 stride 为 1；不支持 int1（bool）缓冲区。

## 同步操作（简表）

```python
extension.sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None)
extension.sync_block_wait(sender, receiver, event_id, ...)
extension.sync_block_all(mode, event_id)   # mode: all_cube/all_vector/all/all_sub_vector
```

`event_id` 0-15；sender≠receiver；set/wait 必须配对；须在 `scope` 上下文中使用。PIPE 默认推断：cube→vector 为 `PIPE_FIX`/`PIPE_MTE2`，vector→cube 为 `PIPE_MTE3`/`PIPE_MTE2`。详见 sync_and_barrier 卡。

## 910_95 特别注意

- `fixpipe` 仅 910_95 系列支持（详见 fixpipe_bias_fusion 卡）
- `cast` overflow_mode="saturate"：910_95 用硬件 `int_cast` 指令；非 910_95 经 fp32 中转，精度略有差异
- `libdevice.rint`：910_95 用 HMF 硬件实现（`__hmf_rint`）；其他平台软件实现（银行家舍入）
- 非 910_95 平台 `cast` 不支持 fp8/fp64


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L22](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L22)
- [aux_ops.py#L135-L151](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/aux_ops.py#L135-L151)
- [aux_ops.py#L153-L162](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/aux_ops.py#L153-L162)
- [vec_ops.py#L47-L92](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L47-L92)
- [vec_ops.py#L95-L137](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L95-L137)
- [vec_ops.py#L139-L177](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py#L139-L177)
- [core.py#L1935-L1975](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1935-L1975)
- [standard.py#L174](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/standard.py#L174)
- [core.py#L202-L244](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L202-L244)
- [core.py#L166-L171](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L166-L171)
- [custom_op.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/custom_op.py)
- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/extension/buffer/language/core.py)
