---
type: API
title: 比较、逻辑、移位与 where 操作 API
description: Triton-Ascend 比较/逻辑/移位/where API：有序浮点比较语义、logical_and 与按位与的区别、移位位宽检查、where
  非短路求值与 bf16 比较提升 fp32。
tags: [task.lookup, topic.compute.ordering, topic.implementation.masking, topic.programming_model.execution_unit.vector, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Comparing_Ops
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Comparing_Ops/eq.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# 比较、逻辑、移位与 where 操作 API

比较操作返回 `int1` 张量，NPU 上由 Vector 单元执行；bf16 比较需先提升 fp32（Vector 单元无 bf16 FCMP 指令）。

## 比较操作

`tl.eq/ne/lt/le/gt/ge`（或运算符 `==`/`!=`/`<`/`<=`/`>`/`>=`），返回广播后形状的 int1 张量。

| 操作 | 浮点 IR | 有符号整数 IR | 无符号整数 IR |
|------|---------|-------------|-------------|
| `==` / `!=` | `fcmpOEQ` / `fcmpUNE` | `icmpEQ` / `icmpNE` | 同有符号 |
| `<` / `<=` | `fcmpOLT` / `fcmpOLE` | `icmpSLT` / `icmpSLE` | `icmpULT` / `icmpULE` |
| `>` / `>=` | `fcmpOGT` / `fcmpOGE` | `icmpSGT` / `icmpSGE` | `icmpUGT` / `icmpUGE` |

**浮点比较是有序（ordered）比较**：NaN 与任何值比较（含自身）均为 False；`!=` 为 True。检测 NaN 用 `x != x`。

## 逻辑与按位操作

| 操作 | 运算符 | 函数 | 说明 |
|------|--------|------|------|
| 按位与/或/异或 | `&` / `\|` / `^` | `and_` / `or_` / `xor_` | 仅整数；先整数类型提升到更宽类型 |
| 按位取反 | `~x` | `invert(x)` | 等价 `x ^ all_ones`；不支持浮点/指针 |
| 逻辑与/或 | - | `tl.logical_and` / `tl.logical_or` | 返回 int1；非布尔输入先转布尔（`!= 0`） |
| 逻辑非 | - | `not_(x)` | 返回 int1；浮点不支持 |
| 算术取负 | `-x` | `minus(x)` | 等价 `0 - x`；不支持指针/布尔 |

**logical_and vs `&` 语义差异**（常见陷阱）：

| 特性 | `logical_and` | `&`（按位与） |
|------|-------------|------------|
| 输入 | 任意数值（非布尔先转布尔） | 仅整数 |
| 输出 | int1 | 提升后的整数 |

对 int1 张量两者结果相同；对 int32 不同（`3 & 5 = 1` 而 `logical_and(3, 5) = True`）。

## 移位操作

| 操作 | 函数 | 说明 |
|------|------|------|
| `x << y` | `shl` | 整数左移 |
| `x >> y`（无符号） | `lshr` | 逻辑右移，高位填 0 |
| `x >> y`（有符号） | `ashr` | 算术右移，高位填符号位 |

`>>` 按左操作数符号性自动选 ashr/lshr。移位量 ≥ 操作数位宽时**行为未定义**（Triton 发警告但不阻止编译）。

## tl.where

```python
triton.language.where(condition, x, y) -> tensor
```

| 参数 | 说明 |
|------|------|
| `condition` | int1 张量，True 选 x、False 选 y；形状决定输出形状 |
| `x` / `y` | **必须相同 dtype**，自动广播到 condition 形状 |

**非短路求值**：x 和 y 总是被求值。若 x/y 的计算涉及可能越界的内存访问，必须用 `tl.load`/`tl.store` 的 `mask` 参数而非 where。

## NPU 适配要点

1. **bf16 比较提升**：经 `_promote_bfloat16_to_float32` 提升 fp32 执行，比较本身精确；但 bf16 仅 7 位尾数，提升前的精度损失可能使结果与直接 fp32 计算不同。
2. **int1 特殊处理**：`tl.load` 加载布尔指针时实际加载 int8 并标记 `was_bool_to_int8`；`logical_and`/`logical_or`/`invert` 检测该标记先转回 int1。
3. **i64 限制**：`tl.where` 不支持 int64（见支持矩阵）。

## 代码示例

```python
@triton.jit
def masked_relu_kernel(x_ptr, mask_ptr, out_ptr, N: tl.constexpr,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    boundary_mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=boundary_mask, other=0.0)
    custom_mask = tl.load(mask_ptr + offsets, mask=boundary_mask, other=0)
    active = (x > 0) & (custom_mask != 0)     # int1 上 & 与 logical_and 等价
    result = tl.where(active, x, 0.0)
    tl.store(out_ptr + offsets, result, mask=boundary_mask)

# 嵌套 where 实现 clamp
result = tl.where(x < min_val, min_val, tl.where(x > max_val, max_val, x))
```

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [semantic.py#L552-L643](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L552-L643)
- [semantic.py#L412-L516](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L412-L516)
- [semantic.py#L443-L501](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L443-L501)
- [semantic.py#L504-L516](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L504-L516)
- [semantic.py#L1738](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1738)
- [core.py#L1886-L1905](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1886-L1905)
- [core.py#L888-L951](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L888-L951)
