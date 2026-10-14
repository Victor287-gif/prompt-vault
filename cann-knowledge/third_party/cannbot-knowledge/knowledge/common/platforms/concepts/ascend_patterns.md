---
type: Concept
title: Ascend NPU 特有语法与模式
description: Triton-Ascend 与标准 GPU Triton 的关键差异：Grid 配置、constexpr 规则、禁用语法、并行模式与 where/slice
  操作替代方案。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.scalar, topic.parallelism, topic.implementation.masking, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/include/Dialect/TritonAscend/IR/TritonAscendOps.td
  role: constraint
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# Ascend NPU 特有语法与模式

> Triton-Ascend 与标准 GPU Triton 的关键差异汇总。实现任何 kernel 前必读。

## 1. Grid 配置规范

- Grid 必须是 `tuple` 类型，最多 3 维
- 各维度乘积 ≤ 65535（`x * y * z ≤ 65535`）
- 不能使用 4 维 Grid
- 大 shape 时用 grid-stride loop 代替大 grid

```python
# 正确
grid = (48,)            # 1D，乘积=48
grid = (24, 2)          # 2D，乘积=48

# 错误
grid = 48               # 不是 tuple
grid = (256, 256)       # 乘积=65536 > 65535
```

## 2. `tl.constexpr` 规则

`tl.constexpr` **仅允许在 kernel 函数参数声明中使用**，禁止在 host 侧使用。

## 3. 禁用语法

| 禁止 | 替代方案 |
|------|----------|
| `return`/`break`/`continue` | mask 控制执行逻辑 |
| `lambda` 表达式 | `tl.where` / 内联函数 |
| 链式布尔 `a and b and c` | `mask1 & mask2 & mask3` |
| `tensor[i]` 直接索引 | `tl.load` / `tl.store` |
| `tl.swizzle2d` | `grouped_launch_diagonal` |
| int64 CMP/ADD | 转 fp32 / int32 |
| 负数偏移 | `tl.maximum(offset, 0)` + mask |

## 4. 核心数与并行模式

- 910b2: 通常 20 Cube + 48 Vector = 68 AI Core
- 使用 `torch_npu.npu_core_num()` 获取实际核数
- Grid 按核数配置，超出用 grid-stride loop

## 5. where/split/slice 操作

| 操作 | 正确写法 |
|------|----------|
| 条件选择 | `tl.where(cond, x, y)` — cond 必须是 mask tensor |
| 张量切片 | `tl.load(ptr + start + tl.arange(0, size))` |
| 维度分割 | 手动计算偏移，分两次 load |

## 6. Tensor 创建

| 操作 | API |
|------|-----|
| 零张量 | `tl.zeros([SIZE], dtype=tl.float32)` |
| 全填充 | `tl.full([SIZE], value, dtype=tl.float32)` |
| 整数序列 | `tl.arange(0, SIZE)` |
| 2D 广播 | `tl.arange(0, M)[:, None]` + `tl.arange(0, N)[None, :]` |

