---
type: Optimization
title: 入参静态化 优化模式
description: 在 Triton NPU kernel 中，将固定数值的入参声明为 tl.constexpr，可以让编译器在编译时进行更多的常量折叠和常量传播优化，从而提升
  kernel 的执行效率。
tags: [task.optimization, topic.compilation, topic.tiling, topic.compute.reduction]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/triton/autotune.md
  role: primary
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# 入参静态化 优化模式

## 概述

在 Triton NPU kernel 中，将固定数值的入参声明为 `tl.constexpr`，可以让编译器在编译时进行更多的常量折叠和常量传播优化，从而提升 kernel 的执行效率。

## 触发条件

**当代码中存在以下固定数值参数时，应考虑将其声明为 `tl.constexpr`**：

1. **固定的 BLOCK_SIZE**：如 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 等
2. **固定的 STRIDE**：如 `stride_m`、`stride_n` 等
3. **模型配置超参数**：如 MoE 场景中的 `num_experts`、`topk_numel`、`seq_len` 等。这些值在模型训练/推理过程中通常是固定配置（如 `num_experts=128`），不应仅凭变量名判断为运行时变量。若该参数来自 Python 层的固定配置，应优先尝试声明为 `tl.constexpr`
4. **启动级常量**（launch-level constants）：如 `repeat` 次数 `r`、操作轴 `axis`、reduce 维度 `dim`、padding 大小 `pad_l`/`pad_r` 等。此类参数虽然在 `forward()` 内的多次 `kernel[grid]()` 之间可能变化，但在**单次启动内固定**。Triton Ascend 编译器会在每次启动时根据传入的 `constexpr` 值进行**启动级特化**（launch-level specialization），生成特化代码，典型收益包括：
   - 触发 `for i in range(r)` 等循环的编译期 unroll，消除标量循环开销
   - 消除基于启动级常量的分支判断（如 `if mode == 0` 在特化后只剩一个分支）
5. **其他在 kernel 生命周期内不会变化的常量参数**

如果已有入参中的某个参数对性能影响很大，且在kernel生命周期内不会变化，如若不确定则应该**询问用户是否可以将该参数设置为 `tl.constexpr`**。

## 优化方法

### 原始代码

```python
@triton.jit
def kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_an,  # 这些是入参，但实际运行时是固定值
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # ...
```

### 优化后代码

```python
@triton.jit
def kernel(
    A, B, C,
    M, N, K,
    stride_am: tl.constexpr,  # 声明为 constexpr
    stride_an: tl.constexpr,  # 声明为 constexpr
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # ...
```

## 关键点

1. **常量性质**：只有那些在 kernel 运行时不会变化的参数才适合声明为 `tl.constexpr`
2. **性能影响**：对于性能敏感的参数（如 BLOCK_SIZE），应优先考虑声明为 `tl.constexpr`
3. **用户确认**：如果不确定某个参数是否可以设为 constexpr，应询问用户

## 性能收益

将固定参数声明为 `tl.constexpr` 可以：
- 启用编译时常量折叠
- 帮助编译器进行更 aggressive 的常量传播
- 减少运行时分支判断开销

## ⚠️ 反向场景：do_not_specialize

与 `tl.constexpr` 相反，某些参数**不应**被特化，否则会导致过度编译。

**适用场景**：当参数在每次 kernel 启动时取值不同，且取值种类较多时，应将其加入 `do_not_specialize` 列表，避免编译器为每个不同值生成一份 kernel 二进制。

**典型参数**：
- `T`（序列长度）：变长场景下每次调用可能不同
- `stride_hz`（步长参数）：不同张量布局下值不同
- 其他随调用变化的动态值

**示例**：
```python
# 优化前：T 和 stride_hz 会被特化，每次不同值触发重新编译
@triton.jit(do_not_specialize=['T'])
def kernel(..., T, stride_hz, ...):

# 优化后：加入 do_not_specialize，避免过度编译
@triton.jit(do_not_specialize=['T', 'stride_hz'])
def kernel(..., T, stride_hz, ...):
```

**判断原则**：
- 参数值在 kernel 生命周期内不变 → 考虑 `tl.constexpr`
- 参数值在每次调用间变化且种类多 → 考虑 `do_not_specialize`
- 参数值在每次调用间变化但种类少（如枚举值）→ 保持默认特化

---
