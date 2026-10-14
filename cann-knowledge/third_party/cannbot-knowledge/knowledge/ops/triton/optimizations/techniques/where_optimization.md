---
type: Optimization
title: Where 条件优化（单位置 False 场景）
description: condition 仅一个位置为 False 的 tl.where 退化为标量离散访存，改用 get_element + insert_slice
  保持向量化。
tags: [task.optimization, topic.compute.ordering, topic.programming_model.execution_unit.scalar, topic.data_movement.memory_access, topic.implementation.masking]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/where.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# Where 条件优化（单位置 False 场景）

## 触发条件

`tl.where(condition, A, B)` 且 condition 同时满足：

1. 形式为 `X != y`：`X` 是地址偏移向量（如 `base + tl.arange(0, BLOCK_SIZE)`），`y` 是标量/常量
2. **仅一个位置为 False**：`y` 在 `X` 取值范围内只出现一次
3. A、B、X shape 相同

直观理解：当 `y ∈ [base, base + BLOCK_SIZE)` 时，`X == y` 仅在偏移 `_offs = y - base` 处成立，where 只在该位置选 B。

**不适用**：condition 有多个位置为 False 时不能用此优化。

## 核心知识

`tl.where` 逐元素比较在 NPU 上导致离散访存、退化为标量操作（910B 上性能损失可达 10x-100x）。单位置 False 的语义等价于"把 A 的某一位置换成 B 的对应值"，可用：

- `get_element`：从 B 提取单个元素（标量操作，只执行一次）
- `insert_slice`：将该元素写入 A 的指定位置（向量化操作）

回退机制（extension 不可用时落到 Triton 核心库）：

```python
try:
    import triton.language.extra.cann.extension as extension
except Exception:
    extension = tl
```

## 代码模式

优化前（where 逐元素比较，退化标量）：

```python
X = base + tl.arange(0, BLOCK_SIZE)
A = tl.load(A_ptr + X)
B = tl.load(B_ptr + X)
A = tl.where(X != y, A, B)
```

优化后（单次提取 + 单次写入）：

```python
X = base + tl.arange(0, BLOCK_SIZE)
A = tl.load(A_ptr + X)
B = tl.load(B_ptr + X)

if base <= y < base + BLOCK_SIZE:
    _offs = y - base
    _val = extension.get_element(B, (_offs,))
    _tensor = tl.full((1,), _val, dtype=A.dtype)
    A = extension.insert_slice(A, _tensor, offsets=(_offs,), sizes=(1,), strides=(1,))
```

要点：`y` 不在范围内时 condition 恒 True，结果就是 A 本身，**不需要任何操作也不必保留原 where**；`tl.full` 的 dtype 必须与 A 一致。

多维扩展（每维各给索引）：

```python
if base_m <= y_m < base_m + BLOCK_M and base_n <= y_n < base_n + BLOCK_N:
    _val = extension.get_element(B, (y_m - base_m, y_n - base_n))
    _tensor = tl.full((1, 1), _val, dtype=A.dtype)
    A = extension.insert_slice(A, _tensor,
                               offsets=(y_m - base_m, y_n - base_n),
                               sizes=(1, 1), strides=(1, 1))
```

## 910_95 特别注意

- 910_95 为 Reg-based 架构（支持 SIMT VF），910B 为 Mem-based；**两种架构下单位置 False 的 where 都应替换**，SIMT 下退化行为更复杂
- `get_element`/`insert_slice` dtype 支持：int8/16/32/64、uint8/16/32/64、fp16、fp32、bf16 均支持；**bool 不支持**（A/B 为 bool 时不能用此优化）
- `insert_slice` 约束：`offsets`/`sizes`/`strides` 长度必须等于张量维度数；`sub` 的 shape 必须与 `sizes` 匹配；插入区域不得越界


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [get_element.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/get_element.md)
- [insert_slice.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/insert_slice.md)
- [vec_ops.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py)
