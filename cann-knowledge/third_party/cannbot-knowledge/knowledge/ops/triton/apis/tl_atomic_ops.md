---
type: API
title: 原子操作 API 及 NPU 限制
description: Triton-Ascend 原子操作 API：atomic_add/max/min/and/or/xor/xchg/cas 签名、NPU
  dtype 支持表、sem/scope 仅支持默认值的限制与浮点 atomic_max 硬件直支持。
tags: [task.lookup, topic.implementation.atomic, topic.data_movement.memory_access, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Atomic_Ops
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Atomic_Ops/atomic_add.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']

---
# 原子操作 API 及 NPU 限制

所有原子操作返回操作前 `pointer` 指向的旧值。NPU 上 dtype 范围与内存序保证与 GPU 差异显著。

## 通用签名

```python
tl.atomic_xxx(pointer, val, mask=None, sem=None, scope=None) -> tensor
tl.atomic_cas(pointer, cmp, val, sem=None, scope=None) -> tensor  # old==cmp 时写入 val
```

| 参数 | 说明 |
|------|------|
| `pointer` / `val` | 操作地址与右操作数（`val` 类型为 `pointer.dtype.element_ty`） |
| `mask` | 指定操作范围；None 时自动创建全 True mask |
| `sem` | 内存语义：`"acquire"`/`"release"`/`"acq_rel"`（默认）/`"relaxed"` |
| `scope` | 同步范围：`"gpu"`（默认）/`"cta"`/`"sys"` |

## NPU 内存序限制（重要）

| 参数 | GPU | NPU |
|------|-----|-----|
| `sem` | 全支持 | **仅 `"acq_rel"`（默认值）** |
| `scope` | 全支持 | **仅 `"gpu"`（默认值）** |

传入其他值可能导致未定义行为。

## 各操作与 NPU dtype 支持

| 操作 | 语义 | NPU dtype 支持与限制 |
|------|------|---------------------|
| `tl.atomic_add` | `*p += val` | int8/int16/int32/uint8/uint16/uint32/fp16/fp32/bf16 ✓；**int64/fp64 ✗**。浮点用 FADD，整数用 ADD |
| `tl.atomic_max` | `*p = max(*p, val)` | **浮点由硬件 MAX 操作码直接支持**（GPU 需 bitcast 转整数实现）；有符号 MAX / 无符号 UMAX |
| `tl.atomic_min` | `*p = min(*p, val)` | 同上，硬件 MIN/UMIN 直接支持浮点 |
| `tl.atomic_and` | `*p &= val` | 仅整数；**不支持在 loop 中使用** |
| `tl.atomic_or` | `*p \|= val` | 仅整数；不支持 loop 中使用 |
| `tl.atomic_xor` | `*p ^= val` | 仅整数；不支持 loop 中使用 |
| `tl.atomic_xchg` | `*p = val`，返回旧值 | 不支持 loop 中使用 |
| `tl.atomic_cas` | 比较并交换 | 910B：int8/uint8/int16/int32/int64/fp16/bf16/fp32；910_95 增 uint16/uint32/uint64/fp8e4m3/fp8e5m2；不支持 loop 中使用 |

与 GPU 对比：NPU 额外支持 int8/int16/uint8/uint16/uint32 的 atomic_add；不支持 int64/fp64 的 atomic_add（64 位原子加可用 atomic_cas 自旋锁替代）。

## 代码示例

```python
@triton.jit
def atomic_histogram_kernel(x_ptr, hist_ptr, N: tl.constexpr,
                            NUM_BINS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x = x % NUM_BINS
    ones = tl.where(mask, 1, 0)
    tl.atomic_add(hist_ptr + x, ones, mask=mask)

# atomic_cas 自旋锁
@triton.jit
def spin_lock_kernel(lock_ptr, data_ptr, result_ptr, N: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    old = tl.atomic_cas(lock_ptr, 0, 1)
    if old == 0:
        data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        tl.store(result_ptr + offsets, data * 2.0, mask=mask)
        tl.atomic_xchg(lock_ptr, 0)
```

## NPU 适配要点

1. **并发安全性**：同一地址的原子操作串行化，无数据竞争；但仅 `"acq_rel"` 序，不保证不同地址间操作顺序。
2. **atomic_max/min 的 NaN**：硬件直接执行浮点 max，NaN 行为遵循 IEEE 754（结果可能是 NaN 或另一操作数）。
3. **离散 mask store 关联**：离散 mask 的 `tl.store` 在 NPU 上会被拆解为 atomic {load, select, store} 组合，corner case 可能有正确性问题（见内存操作卡）。
4. **不支持多核 add + 保存中间结果**（atomic_add 限制）。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L1760-L1877](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1760-L1877)
- [semantic.py#L1348-L1542](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1348-L1542)
- [semantic.py#L1365-L1384](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1365-L1384)
