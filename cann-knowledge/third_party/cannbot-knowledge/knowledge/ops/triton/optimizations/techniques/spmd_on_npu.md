---
type: Optimization
title: SPMD 模型在 NPU 上的映射
description: Triton SPMD 编程模型从 GPU Thread Block 到 NPU AI Core 的映射机制，Grid 物理核绑定策略与
  coreDim 65535 上限规避。
tags: [task.migration, topic.architecture, topic.parallelism, topic.tiling, topic.quality.compatibility.version_migration]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# SPMD 模型在 NPU 上的映射

## 触发条件

- 从 GPU 迁移 Triton 算子，需要确定 Grid 配置
- kernel 并行度低或核启动开销高
- 编译报 `coreDim can't be greater than UINT16_MAX`

## GPU vs NPU 语义对照

| 概念 | GPU 语义 | NPU 语义 |
|------|----------|----------|
| Program 实例 | Thread Block | AI Core 执行单元（1 Cube + 2 Vector） |
| 并行粒度 | Warp（32 线程） | AI Core（最小调度单位） |
| 执行模型 | SIMT（线程独立 PC，可 divergent） | SIMD（数据并行，用 mask 表达条件） |
| Grid 本质 | 逻辑维度，与物理 SM 解耦 | 物理核强绑定，应对齐 AI Core 总数 |
| 核数上限 | 数百 SM | 20~36 AI Core；超出时分批调度，引入额外开销 |
| 同步 | `__syncthreads()` 块内线程同步 | `sync_block_all` / `sync_block_set` / `sync_block_wait` 核间同步 |

## AI Core 数量参考

| NPU 型号 | AI Core 数 | Vector Core 数 |
|----------|-----------|---------------|
| Ascend910B1 | 24 | 48 |
| Ascend910B3 | 20 | 40 |
| Ascend910_9381 | 24 | 48 |
| Ascend910_9581 (910_95) | 32 | 64 |
| Ascend910_9599 (910_95) | 36 | 72 |

```python
import torch, torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]
```

## 核心原则：Grid 固定为物理核数 + 核内循环

- 纯 Vector 算子（无 `tl.dot`）：`NUM_CORE = num_vectorcore`
- CV 融合算子（含 `tl.dot`）：`NUM_CORE = num_aicore`（执行时按 1:2 调用 Vector Core）

```python
@triton.jit
def parallel_kernel(data_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(N, BLOCK_SIZE)
    # 跨步分配：起始 pid、步长 NUM_CORE，任务均匀轮转
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        data = tl.load(data_ptr + offsets, mask=mask)
        tl.store(data_ptr + offsets, data * 2.0, mask=mask)
```

GPU→NPU 对照：

```python
# GPU: grid 与物理核解耦，硬件自动调度
grid_gpu = (triton.cdiv(N, BLOCK_SIZE),)
# NPU: grid 固定物理核数，核内循环处理多块
grid_npu = (NUM_CORE,)
```

## SIMT → SIMD：条件执行改写

```python
# GPU SIMT: 每线程独立分支 if thread_id < N: ...
# NPU SIMD: 用 mask + tl.where 表达条件
mask = offsets < N
data = tl.load(ptr + offsets, mask=mask)
result = tl.where(mask, data * 2.0, 0.0)
tl.store(ptr + offsets, result, mask=mask)
```

## coreDim 上限 65535 及规避

`coreDim = grid_x * grid_y * grid_z` 不能超过 UINT16_MAX（65535）。大数据 + 小 BLOCK 时易超限，如 `N=2^30, BLOCK_SIZE=2048 → coreDim=524288`。

三种解法：

1. `export TRITON_ALL_BLOCKS_PARALLEL=1`：编译器自动将逻辑核数收缩为物理核数。仅当逻辑核间无数据依赖时可用。
2. 增大 BLOCK_SIZE：`BLOCK_SIZE = max(32768, triton.next_power_of_2(triton.cdiv(N, 65535)))`。
3. 增大 BLOCK_SIZE 导致 UB 溢出时，引入二级分块 `BLOCK_SIZE_SUB`：外层保证 coreDim 合规，内层控制 UB 使用量。

## NPU 适配要点

1. Grid 必须对齐物理核数，禁用 GPU 的 `cdiv(N, BLOCK_SIZE)` 作 Grid
2. 优先 1D Grid：2D/3D Grid 在 NPU 上会被合并为 1D
3. 区分 Vector / CV 算子选择 `num_vectorcore` / `num_aicore`
4. Grid 超过物理核数会分批调度，引入核启动与初始化开销
5. 用 `@triton.autotune` 搜索最优 BLOCK_SIZE

## 不适用场景

- 核间存在数据依赖、必须按 program_id 顺序执行的 kernel：不能用 `TRITON_ALL_BLOCKS_PARALLEL`
- 使用 `tl.barrier()` / `tl.NamedBarrier` 的跨核同步策略：不适用 Grid 收缩


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [programming_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/programming_guide.md)
- [migrate_from_gpu.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/migrate_from_gpu.md)
- [architecture_difference.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/architecture_difference.md)
- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py)
