---
type: Optimization
title: Grid/Program ID 与 AI Core 对应关系
description: tl.program_id/tl.num_programs 在 NPU 上的语义、Grid 合并为 1D 的机制、Grid 大小与物理核数对齐及
  Grid 合并优化。
tags: [task.optimization, topic.parallelism, topic.tiling, topic.compute.indexing]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---
# Grid/Program ID 与 AI Core 对应关系

## 关键 API 语义

| API | NPU 语义 | 说明 |
|-----|----------|------|
| `tl.program_id(axis)` | 当前 program（AI Core）在指定轴的索引 | axis ∈ {0,1,2} 对应 x/y/z；返回 `[0, num_programs(axis))` |
| `tl.num_programs(axis)` | 指定轴启动的 program 总数 | kernel 内取 Grid 大小，应与物理核数对齐 |
| `kernel[grid](...)` | AI Core 分配方案 | 推荐 1D Grid |
| coreDim | `grid_x * grid_y * grid_z`，总 program 数 | 不能超过 65535 |

`tl.num_programs(0)`（kernel 内 Grid 大小）与 host 端 `get_device_properties` 的物理核数在 Grid 对齐时相等；kernel 内用前者，host 端用后者。

## Grid 合并机制：2D/3D 被合并为 1D

```python
# 2D Grid (4, 5) 等效于 1D Grid (20,)
# flat_pid = pid_x + pid_y * 4
# 3D Grid (2, 3, 4) 等效于 1D Grid (24,)
# flat_pid = pid_x + pid_y * 2 + pid_z * 6
```

2D Grid 与展平后的 1D Grid 效果相同，无性能差异。推荐直接 1D Grid + `tl.program_id(0)`，避免多维索引复杂性。若 2D kernel 中需要全局 ID，手动计算 `flat_pid = pid_x + pid_y * tl.num_programs(0)`。

## Grid 大小与物理核数

| Grid 大小 | 行为 |
|-----------|------|
| Grid < 物理核数 | 部分核空闲，资源浪费 |
| Grid = 物理核数 | **最优配置** |
| Grid > 物理核数 | 分批调度，额外核启动开销 |

## 跨步分配模式（标准写法）

```python
@triton.jit
def strided_kernel(data_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(N, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        data = tl.load(data_ptr + offsets, mask=mask)
        tl.store(data_ptr + offsets, data * 2.0, mask=mask)
```

## 多维任务展平为 1D（推荐 NPU 模式）

将 batch/heads/blocks 等维度展平后在核内反解索引：

```python
@triton.jit
def attention_kernel(Q, K, V, Out, stride_qz, stride_qh,
                     Z: tl.constexpr, H: tl.constexpr,
                     N_CTX: tl.constexpr, BLOCK_M: tl.constexpr):
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # ... 处理当前 block
```

## Grid 合并优化（减少核启动开销）

多次小 Grid 启动合并为单次大 Grid + 核内循环：

```python
# 优化前：循环中多次 kernel[grid](...) 启动
for i in range(num_iterations):
    my_kernel[(num_cores,)](data_ptr + i * chunk_size, chunk_size, BLOCK_SIZE)

# 优化后：单次启动，核内跨步循环
grid = (num_cores,)
merged_kernel[grid](data_ptr, total_size, chunk_size, BLOCK_SIZE)
```

## TRITON_ALL_BLOCKS_PARALLEL 与 auto_blockify_size

逻辑 Grid 大于物理核数且核间可并行时：

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

编译器自动将逻辑核数收缩为物理核数并在核内添加循环；可用 `auto_blockify_size` 编译选项指定扩展的左起第一个维度大小：

```python
triton.Config({'BLOCK_SIZE': 1024, 'auto_blockify_size': 20})
```

## libentry 装饰器

`triton.runtime.libentry` 自动处理 Grid 配置和核数对齐：

```python
from triton.runtime import libentry

@libentry()
@triton.jit
def auto_kernel(data_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    ...
```

## 注意事项与调试

- Grid 合并要求合并前后数据访问模式不变且核间无依赖；kernel 内用 `program_id(1/2)` 做条件分支时合并可能导致逻辑错误
- coreDim 超限时检查 `grid_x * grid_y * grid_z <= 65535`
- `TRITON_DEBUG=1` 可查看编译中间产物排查 Grid 问题


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L1151-L1166](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1151-L1166)
- [core.py#L1170-L1178](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1170-L1178)
- [programming_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/programming_guide.md)
- [architecture_difference.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/architecture_difference.md)
