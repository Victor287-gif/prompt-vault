---
type: Guide
title: Int8 到 FP16 类型转换优化案例
description: 大shape类型转换（int8→fp16）优化：通过二次切分（BLOCK_SIZE+TILE_SIZE）提高UB利用率，核数2048时性能最优，适用于shape较大（百万级元素）的elementwise类型转换场景
tags: [task.design, topic.compute.elementwise.arithmetic, topic.tiling, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Creation_Ops/cast.md
  role: primary
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Int8 到 FP16 类型转换优化案例

> 本卡源自 designer 设计案例，仅含**定性**配置结论（如「性能最优」
> 「并行度低」），无定量实测数值，故按 SPEC §9 不作为 `perf_card`。
> 全局约束见 [global_constraints.md](global_constraints.md)。

## 任务特征
- **操作类型**：Elementwise，类型转换操作
- **数据尺寸**：(128, 1024, 1024)，shape较大
- **数据类型**：输入int8，输出fp16
- **任务特点**：可以按照轴的顺序（可flatten为一根轴），外层并行，内层向量化，若UB存不下，可考虑多次切分

## 优化：二次切分 + 用满UB

```python
# Triton 内核实现：将BLOCK_SIZE分块，每次搬运TILE_SIZE大小的数据
configs = [
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 65536}), # 核数2048, 性能最优！用满UB且无二次切分
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 32768}), # 核数2048，但UB未用满
    triton.Config({"BLOCK_SIZE": 2097152, "TILE_SIZE": 65536}), # 核数64, 并行度低
    triton.Config({"BLOCK_SIZE": 4194304, "TILE_SIZE": 65536}), # 核数32, 并行度更低
]

# 内核操作：
block_start = pid * BLOCK_SIZE
for i in range(0, BLOCK_SIZE, TILE_SIZE):
    offsets = block_start + tl.arange(0, TILE_SIZE)
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask)
    output_data = tl.cast(input_data, tl.float16)
    tl.store(output_ptr + offsets, output_data, mask=mask)
```

### 优化内容
- triton 内核部分使用for循环，尝试进行二次切分，每次搬运TILE_SIZE大小的数据，提高UB的利用率
- 在一定范围内提高核数，并尝试用满UB
- 核内没有二次切分时性能最优（BLOCK_SIZE = TILE_SIZE = 65536）

### 总结
1. 当数据的shape较大时，为了获得更佳的性能，切分值设置尽量能被shape的大小整除
2. 对于单纯的Elementwise操作，将多根轴的元素展开为一根轴，然后在这根轴上进行切分
3. 将block分配给每个线程块，若UB存不下，可考虑多次切分（二次切分）
