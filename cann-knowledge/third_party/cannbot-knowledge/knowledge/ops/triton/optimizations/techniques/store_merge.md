---
type: Optimization
title: Store 合并优化
description: 多条同基地址、偏移连续的 tl.store 合并为一次连续写入，减少 MTE3 搬运次数与启动开销。
tags: [task.optimization, topic.data_movement.memory_access, topic.data_movement.data_copy, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# Store 合并优化

## 触发条件

kernel 中存在多次 `tl.store` 对**相同基地址、不同偏移**的存储，且：

- 各 store 共享同一个 `base_ptr`
- 各 offset 合并后构成一整块**连续**地址（不能有间隔）
- 各 data tensor 可通过 `tl.where` 等方式整合为一个大 tensor

## 核心知识

每次 `tl.store` 触发一次独立的 MTE3（UB→GM）搬运，每次搬运都有启动开销。地址连续时可先在 UB 中拼接数据，再用一次 `tl.store` 完成写入，减少指令数与启动延迟。

## 代码模式

优化前（3 次独立 store，每次只写 1 个元素）：

```python
ps1_idx_0 = pid * 3
ps1_idx_1 = ps1_idx_0 + 1
ps1_idx_2 = ps1_idx_0 + 2

tl.store(problem_sizes1_ptr + ps1_idx_0, data0, mask=ps1_idx_0 < num_experts * 3)
tl.store(problem_sizes1_ptr + ps1_idx_1, data1, mask=ps1_idx_1 < num_experts * 3)
tl.store(problem_sizes1_ptr + ps1_idx_2, data2, mask=ps1_idx_2 < num_experts * 3)
```

优化后（1 次合并 store）：

```python
offs_n = tl.arange(0, 3)[None, :]

data = tl.zeros((BLOCK_SIZE, 3), dtype=tl.int32)
data = tl.where(offs_n == 0, data0, data)
data = tl.where(offs_n == 1, data1, data)
data = tl.where(offs_n == 2, data2, data)

base_offsets = pid[:, None] * 3 + offs_n
tl.store(problem_sizes1_ptr + base_offsets, data, mask=mask[:, None])
```

含固定值时用 `tl.full` 初始化，省掉一次 `tl.where`：

```python
data = tl.full((BLOCK_SIZE, 3), FILL_VALUE, dtype=tl.int32)
data = tl.where(offs_n == 0, data0, data)
data = tl.where(offs_n == 1, data1, data)
```

方法要点：offset 用 `tl.arange` 直接生成整块连续偏移；data 用 `tl.zeros`/`tl.full` 建大 tensor 再逐位 `tl.where` 填充；原 mask 按逻辑合并。

## 910_95 特别注意

合并后大 data tensor 占更多 UB，大小 = `BLOCK_SIZE * N * sizeof(dtype)`（N 为合并的 store 数），需确保加其他 UB 占用后不溢出：

| 硬件 | UB 可用容量 | multibuffer 开启后 |
|------|-----------|----------------|
| 910B | 192 KB | 96 KB |
| 910_95 | 248 KB | 124 KB（910_95 默认 multibuffer=False，手动开启后需重估 BLOCK_SIZE） |

910_95 上 MTE3 搬运要求 32B 对齐。合并后的连续 store 天然更易满足对齐，是额外收益；但合并后总字节数仍可能不足 32B（如 3×int32=12B），此时主要收益是减少搬运次数。