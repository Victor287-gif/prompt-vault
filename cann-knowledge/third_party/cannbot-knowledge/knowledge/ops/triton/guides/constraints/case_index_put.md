---
type: Guide
title: Index Put 索引赋值优化案例
description: 索引赋值（index_put）优化：批量加载索引数据到UB后循环内通过get_element复用（避免重复访问全局内存），显著降低内存访问延迟，适用于需要在循环中多次访问同一片数据的不规则内存访问场景
tags: [task.design, topic.compute.indexing, topic.tiling, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Index Put 索引赋值优化案例

> 本卡源自 designer 设计案例，仅含**定性**配置结论（如「性能最优」
> 「并行度低」），无定量实测数值，故按 SPEC §9 不作为 `perf_card`。
> 全局约束见 [global_constraints.md](global_constraints.md)。

## 任务特征
- **操作类型**：索引赋值，根据索引映射将数据写入目标缓冲区
- **数据尺寸**：输入分数(16384, 4)，组缓冲区(8, 65536)
- **特点**：不规则内存访问，需逐元素处理以避免写冲突

## 优化：批量加载 + 数据复用

### 错误：简单方式：循环内重复加载

```python
for i in tl.range(0, BLOCK_SIZE):
    if start_idx + i < total_elements:
        # 每次循环都从全局内存加载索引
        unit_idx = tl.load(unit_indices_ptr + start_idx + i)
        pos_idx = tl.load(position_map_ptr + start_idx + i)
```

**问题**：每次循环都访问全局内存，延迟高，效率低。

### 正确：优化方式：批量加载到UB，循环内复用

```python
# 循环外：批量加载一片索引数据到UB（统一缓冲区）
unit_indices_tile = tl.load(unit_indices_ptr + offsets, mask=mask, other=0)
position_map_tile = tl.load(position_map_ptr + offsets, mask=mask, other=0)

# 循环内：通过get_element从UB中取数，复用数据
for i in tl.range(0, BLOCK_SIZE):
    if start_idx + i < total_elements:
        # 从UB中取数，避免重复访问全局内存
        unit_idx = tl.get_element(unit_indices_tile, [i])
        pos_idx = tl.get_element(position_map_tile, [i])
        # 后续处理...
```

### 优化内容
- 在循环外，通过一次`tl.load`操作将整个BLOCK_SIZE的索引数据批量加载到UB
- 在循环内，通过`tl.get_element`从UB中逐个取出索引值
- 将多次全局内存访问转换为一次批量加载+多次片上缓存访问
- 显著降低内存访问延迟

### 总结
**[通用优化]** 当需要在循环中多次访问同一片数据时，应先批量加载到片上缓存（UB），然后通过get_element逐个取用，实现数据复用，减少全局内存访问次数，提升性能。
