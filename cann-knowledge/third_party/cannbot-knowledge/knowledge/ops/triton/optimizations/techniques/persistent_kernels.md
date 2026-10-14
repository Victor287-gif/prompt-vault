---
type: Optimization
title: Persistent Kernel 模式
description: Triton Ascend Persistent Kernel（grid-stride loop）：适用场景、核数配置与 multi-round
  调度对比。
tags: [task.optimization, topic.parallelism]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# Persistent Kernel 模式

当 n_rows >> num_cores 时，使用 grid-stride loop 固定核数，消除大量 block 的启动开销和 multi-round 调度损耗。

## 适用场景

- n_rows 远大于 AI Core 数（如 batch×seq_len >> 20/24）
- 每个 program 处理多个 row
- 标准 grid 导致 block 数远多于核数

## OPT-triton-persist-001: Grid-Stride Loop

```python
grid = (min(n_blocks, NUM_CORES),)  # 固定 grid 到核数
@triton.jit
def persistent_kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    for block_id in range(pid, tl.cdiv(n, BLOCK_SIZE), tl.num_programs(0)):
        # 每个 core 处理多个 block
```

## OPT-triton-persist-002: 核数配置

- 910b2: ~24 Vector cores
- 使用 `torch_npu.npu_core_num()` 获取实际核数
- Grid 固定到核数，不过度也不浪费

## 对比

| 模式 | 优点 | 缺点 |
|------|------|------|
| 标准 grid | 简单、负载天然均衡 | 多轮调度开销 |
| Persistent | 无启动开销、L2 复用 | 需手动 stride、尾块处理 |

