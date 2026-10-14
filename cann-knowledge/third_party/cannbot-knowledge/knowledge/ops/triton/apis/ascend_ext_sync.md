---
type: API
title: sync_block_set / wait / all 同步操作
description: 昇腾 NPU Cube-Vector 核心间的 Event/Flag 同步原语：sync_block_set、sync_block_wait、sync_block_all
  与 Vector 内部 debug_barrier（SYNC_IN_VF）。
tags: [task.lookup, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/scope.py
  role: implementation
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/semantic.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]

---
# sync_block_set / wait / all 同步操作

昇腾 NPU Cube-Vector 分离架构中，Cube 与 Vector 核心可并行执行。三个同步操作基于 NPU Event/Flag 硬件机制实现核心间精确的生产者-消费者同步，是 Cube-Vector 协同计算的基础。

## 关键概念

| 概念 | 说明 |
|------|------|
| Event | 硬件事件信号，用于核心间通知 |
| Event ID | 事件编号，**0-15，共 16 个独立事件通道** |
| Flag | 事件状态标志，set=已触发，wait=等待触发 |

| 操作 | 角色 | 语义 | 类比 |
|------|------|------|------|
| `sync_block_set` | 生产者 | "我已完成计算，你可以读取数据" | 发送信号/释放锁 |
| `sync_block_wait` | 消费者 | "我等待你完成后再读取" | 等待信号/获取锁 |
| `sync_block_all` | 全局 | "所有核心在此处同步" | 全局屏障 Barrier |

| 同步方向 | sender_pipe 默认 | receiver_pipe 默认 | 典型场景 |
|----------|-----------------|--------------------|----------|
| cube → vector | `PIPE_FIX` | `PIPE_MTE2` | Cube 完成 matmul，通知 Vector 后处理 |
| vector → cube | `PIPE_MTE3` | `PIPE_MTE2` | Vector 完成数据准备，通知 Cube 计算 |

## API 参考

### sync_block_set — 生产者发送同步信号

```python
@builtin
def sync_block_set(
    sender: str,              # "cube" 或 "vector"
    receiver: str,            # "cube" 或 "vector"
    event_id: int,            # 事件 ID，范围 0-15
    sender_pipe: PIPE = None,   # 自动推断
    receiver_pipe: PIPE = None, # 自动推断
    _builder=None,
) -> None
```

### sync_block_wait — 消费者等待同步信号

```python
@builtin
def sync_block_wait(
    sender: str,              # 与对应 set 一致
    receiver: str,            # 与对应 set 一致
    event_id: int,            # 与对应 set 一致
    sender_pipe: PIPE = None,
    receiver_pipe: PIPE = None,
    _builder=None,
) -> None
```

### sync_block_all — 全局屏障

```python
@builtin
def sync_block_all(
    mode: str,      # "all_cube" / "all_vector" / "all" / "all_sub_vector"
    event_id: int,  # 范围 0-15
    _builder=None,
) -> None
```

| mode | 说明 |
|------|------|
| `"all_cube"` | 所有 Cube 核心同步 |
| `"all_vector"` | 所有 Vector 核心同步 |
| `"all"` | 所有核心（Cube + Vector）同步 |
| `"all_sub_vector"` | 所有子 Vector 核心同步（910B 等多子 Vector 平台） |

### debug_barrier — Vector 内部流水线同步

```python
@builtin
def debug_barrier(sync_mode: SYNC_IN_VF, _builder=None) -> None
```

`SYNC_IN_VF` 取值（12 种）：`VV_ALL`（Vector-Vector 全同步）、`VST_VLD`（Store→Load）、`VLD_VST`（Load→Store）、`VST_VST`、`VS_ALL`、`VST_LD`、`VLD_ST`、`VST_ST`、`SV_ALL`、`ST_VLD`、`LD_VST`、`ST_VST`。典型用途：Store 后确保数据可见再 Load 时用 `VST_VLD`。

## 代码示例

### 基本 Cube-Vector 同步

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def basic_sync_kernel(a_ptr, b_ptr, c_ptr, N):
    with al.scope(core_mode="cube"):
        a = tl.load(a_ptr + tl.arange(0, 128))
        b = tl.load(b_ptr + tl.arange(0, 128))
        result = tl.dot(a, b)
        al.sync_block_set("cube", "vector", 0)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        processed = result * 2.0
        tl.store(c_ptr + tl.arange(0, 128), processed)
```

### Flash Attention 流水线同步（多 Event ID 握手）

```python
@triton.jit
def flash_attention_fwd(q_ptr, k_ptr, v_ptr, o_ptr, M, N, HEAD_DIM,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    with al.scope(core_mode="cube"):
        for start_n in range(0, N, BLOCK_N):
            qk = tl.dot(q, k)
            al.sync_block_set("cube", "vector", 0)
            al.sync_block_wait("vector", "cube", 1)
            pv = tl.dot(p, v)
            al.sync_block_set("cube", "vector", 2)

    with al.scope(core_mode="vector"):
        for start_n in range(0, N, BLOCK_N):
            al.sync_block_wait("cube", "vector", 0)
            m_new, l_new, softmax_out = _softmax(qk, m_prev, l_prev)
            al.sync_block_set("vector", "cube", 1)
            al.sync_block_wait("cube", "vector", 2)
            acc = _update_output(pv, softmax_out, acc)

    with al.scope(core_mode="cube"):
        al.sync_block_all("all", 0)
    tl.store(o_ptr + offsets, acc)
```

## NPU 适配要点

1. **Event ID 范围 0-15**：复杂流水线需合理规划分配；不相关同步共用同一 ID 会导致死锁或数据竞争。
2. **sender ≠ receiver**：不允许 cube→cube 或 vector→vector；同一核心内部同步用 `debug_barrier`（SYNC_IN_VF）。
3. **set/wait 必须成对**：`sender`、`receiver`、`event_id` 三者必须一致。
4. **PIPE 默认值**按方向自动推断（见上表），仅在需要非默认通路时显式指定。
5. **必须配合 `al.scope`**：同步操作须在 `al.scope(core_mode="cube"/"vector")` 上下文中使用。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L202-L244](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L202-L244)
- [core.py#L336-L349](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L336-L349)
- [semantic.py#L62-L87](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/semantic.py#L62-L87)
- [sync_block.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/sync_block.md)
