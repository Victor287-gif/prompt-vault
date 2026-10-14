---
type: Optimization
title: 同步与 Barrier（NPU 三层同步机制）
description: NPU 管道同步（set_flag/wait_flag/pipe_barrier）、跨核同步（sync_block_*）、UnitFlag
  机制、编译器自动同步注入，以及 910_95 Reg-based 与 910B FFTS 的差异。
tags: [task.optimization, topic.implementation.synchronization, topic.implementation.pipeline, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.debugging]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/sync_block.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# 同步与 Barrier（NPU 三层同步机制）

触发场景：编写 Cube-Vector 协作 kernel（`al.scope`）；`tl.dot` 后紧跟 Vector 后处理需保证数据就绪；多 Block 协作；循环含 `tl.dot` 且循环次数可能为 0；排查 deadlock / barrier timeout / 数据竞争。

## 同步三层架构

```
跨核同步 (Block Sync)：sync_block_set / sync_block_wait / sync_block_all —— 基于 FFTS 或 Reg-based 硬件
管道同步 (Pipe Sync)：set_flag / wait_flag / pipe_barrier —— 同一 Block 内不同 Pipeline 间 Event Flag
UnitFlag 同步：unit_flag_cond / unit_flag_mode —— 嵌入宏操作（如 mmadL1），处理循环可能不执行的依赖
```

## 管道同步（最细粒度）

NPU 多 Pipeline（MTE1/MTE2/MTE3/M/V/FIX）并行执行，有数据依赖时通过 Event Flag 保证顺序。每个 Pipe 有 8 个 Event ID（EVENT_ID0~7），`(set_pipe, wait_pipe, event_id)` 三元组配对。

| 操作 | HIVM IR | 说明 |
|------|---------|------|
| SetFlag | `hir.set_flag` | set_pipe 上置 Flag，通知 wait_pipe 数据就绪 |
| WaitFlag | `hir.wait_flag` | wait_pipe 上等待 Flag |
| PipeBarrier | `hir.pipe_barrier` | 等该 Pipe 所有先前操作完成（重量级） |

典型 Pipe 组合：MTE1→M（L1 加载完通知 Cube）、MTE2→M、M→MTE3（Cube 完通知写回）、M→FIX、M→V。

## 跨核同步（Block Sync）

Triton 层 API（`triton.language.extra.cann.extension`）：

| API | 角色 | 语义 |
|-----|------|------|
| `al.sync_block_set(sender, receiver, event_id)` | 生产者 | "我已完成，你可以读" |
| `al.sync_block_wait(sender, receiver, event_id)` | 消费者 | "等你完成我再读" |
| `al.sync_block_all(mode, event_id)` | 全局屏障 | 所有核心同步到同一点 |
| `al.debug_barrier(sync_mode)` | Vector 内部 | Vector 内部流水线阶段同步 |

参数：`sender`/`receiver` 取 `"cube"`/`"vector"` 且**不能相同**；`event_id` 范围 **0-15**（16 个独立通道）；PIPE 可自动推断（cube→vector：`PIPE_FIX`/`PIPE_MTE2`；vector→cube：`PIPE_MTE3`/`PIPE_MTE2`），也可显式指定。

`sync_block_all` 的 mode：`all_cube` / `all_vector` / `all` / `all_sub_vector`。

SyncBlockMode 枚举（HIVM 层）：ALL_CUBE=0, ALL_VECTOR=1, ALL_SUB_VECTOR=2, BARRIER_CUBE=3, BARRIER_VECTOR=4, ALL=5。

## UnitFlag —— 循环可能不执行的场景

当循环次数可能为 0 时，循环内 set_flag/wait_flag 配对被打破，UnitFlag 通过条件化同步解决。模式：DISABLED(0b00) / ENABLED_WITHOUT_UPDATE(0b10) / ENABLED_WITH_UPDATE(0b11)。

## 自动同步注入（编译器代劳）

HIVM 编译器提供自动同步注入 Pass，**用户通常不需要手写管道同步**：

| Pass | 特点 |
|------|------|
| InjectSync | 启发式，编译快，基本正确 |
| GraphSyncSolver | 图求解，编译慢，同步更优 |

InjectSync 选项：`sync-mode`（NORMAL 插 set/wait_flag；BARRIERALL 插 pipe_barrier，调试用）、`enable-unit-flag`、`assume-alive-loops`（假设循环至少执行一次，简化分析）。

## 代码模式

基本 Cube-Vector 同步：

```python
@triton.jit
def basic_sync_kernel(a_ptr, b_ptr, c_ptr, N):
    with al.scope(core_mode="cube"):
        result = tl.dot(tl.load(a_ptr + tl.arange(0, 128)), tl.load(b_ptr + tl.arange(0, 128)))
        al.sync_block_set("cube", "vector", 0)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        tl.store(c_ptr + tl.arange(0, 128), result * 2.0)
```

Flash Attention 流水线（多 event_id 并行同步）：

```python
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
```

Vector 内部同步（store 后再 load 同址）：

```python
with al.scope(core_mode="vector"):
    tl.store(out_ptr + offsets, data)
    al.debug_barrier(al.SYNC_IN_VF.VST_VLD)
    loaded = tl.load(in_ptr + offsets)
```

## 910_95 特别注意（Reg-based vs FFTS）

| 特性 | 910B 系列 | 910_95 系列 |
|------|----------|------------|
| 跨核同步机制 | FFTS | Reg-based（寄存器 SetFlag/WaitFlag） |
| ffts_base_addr | 必须设置 | **不需要**（硬件寄存器直传信号） |
| UB 容量 | 192KB（单 buffer）/ 96KB（双） | 248KB（单）/ 124KB（双） |
| L0C 容量 | 128KB | 256KB |
| L0C→UB 通路 | 不支持 | 支持（fixpipe），改变了同步 Pipe 组合 |
| multibuffer 默认 | 开启 | **关闭**，需显式 `multibuffer=True` |
| SIMT 模式 | 不支持 | 支持 |

910_95 的 Reg-based 同步延迟更低，但同步语义与 FFTS 一致。

### 何时需要手动同步

需要：①Cube-Vector 协作（`al.scope`）；②多 Block 全局屏障；③Vector 内部流水线同步（debug_barrier）。
不需要：纯 Vector kernel；简单 load→compute→store；编译器可自动推断依赖的场景。

### 常见陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| Event ID 冲突 | 死锁/数据竞争 | 不相关的同步用不同 ID |
| set/wait 不配对 | barrier timeout | 每个 set 有对应 wait 且参数一致 |
| 循环中同步断裂 | 首次迭代正常后续异常 | UnitFlag 或 `assume-alive-loops` |
| sender == receiver | 编译错误 | 同核心内部用 `debug_barrier` |
| TRITON_ALL_BLOCKS_PARALLEL 与同步冲突 | 死锁 | 有同步操作时不能开该选项 |


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)
- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py)
