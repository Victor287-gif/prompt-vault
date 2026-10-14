---
type: API
title: 内存操作 API（tl.load / tl.store / block_ptr）
description: Triton-Ascend 内存操作 API：tl.load、tl.store、tl.make_block_ptr、tl.advance
  的签名、参数语义、dtype 支持与 NPU 特有差异（care_padding、离散访存）。
tags: [task.lookup, topic.data_movement.memory_access, topic.implementation.masking, topic.alignment]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-12T03:07:12Z'
platforms: [a2, a3]

---
# 内存操作 API（tl.load / tl.store / block_ptr）

NPU 上内存操作负责 GM 与 Unified Buffer（UB）间的数据搬运，由 Vector 单元发起。语义与 GPU 有显著差异：`care_padding` 扩展参数、离散访存性能差、cache 类参数无效。

## tl.load

```python
triton.language.load(pointer, mask=None, other=None, boundary_check=(),
                     padding_option="", cache_modifier="", eviction_policy="",
                     volatile=False, care_padding=True) -> tensor
```

| 参数 | 说明 |
|------|------|
| `pointer` | 单指针 / 指针张量 / `make_block_ptr` 产出的块指针 |
| `mask` | `mask[idx]` 为 False 时不读取。单指针可传标量 mask，指针张量可传可广播的 mask；block_ptr 不可传 |
| `other` | mask 为 False 处的填充值，仅在 `mask != None` 时可传 |
| `boundary_check` | 需边界检查的维度元组，仅 block_ptr 场景可传 |
| `padding_option` | `""` / `"zero"` / `"nan"`，越界填充值；仅 `boundary_check` 非空时可传，当前部分场景支持不全，建议用 `mask`+`other` 替代 |
| `cache_modifier` / `eviction_policy` / `volatile` | **对 Ascend 硬件无效**（无 L1/L2 cache 控制语义），传入不报错但无效果 |
| `care_padding` | **Ascend 扩展**，默认 True。`other` 非 None 时不生效；`other=None` 且 True 时 masked 位置填 0；False 时 masked 位置为随机值但性能更好 |

**约束**：
- 单指针：`mask`/`other` 必须标量，不允许 `boundary_check`/`padding_option`
- 指针张量：`mask`/`other` 隐式广播到 `pointer.shape`，不允许 `boundary_check`/`padding_option`
- block_ptr：`mask`/`other` 必须为 None，可指定 `boundary_check`/`padding_option`
- `mask=None` 而 `other` 非 None 非法

**dtype 支持（Ascend）**：int8/int16/int32/int64/fp16/fp32/bf16/bool ✓；uint8/16/32/64、fp64 ✗（硬件限制）

## tl.store

```python
triton.language.store(pointer, value, mask=None, boundary_check=(),
                      cache_modifier="", eviction_policy="") -> None
```

- `value` 隐式广播到 `pointer.shape` 并转换为 `pointer.dtype.element_ty`
- 不允许向 const 指针 store；block_ptr 场景 `mask` 必须为 None
- dtype 支持同 `tl.load`

## tl.make_block_ptr / tl.advance

```python
triton.language.make_block_ptr(base, shape, strides, offsets, block_shape, order) -> tensor
triton.language.advance(base, offsets) -> tensor
```

| 参数 | 说明 |
|------|------|
| `base` / `shape` / `strides` | 父张量基址、形状、步幅 |
| `offsets` / `block_shape` | 块偏移量、块形状 |
| `order` | 原始数据格式的维度顺序 |

`advance(base, offsets)` 按维度推进块指针偏移量。块指针不允许算术运算，不允许通过调整 stride 顺序实现转置。

## 代码示例

```python
@triton.jit
def block_ptr_kernel(x_ptr, y_ptr, N: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    x_block = tl.make_block_ptr(base=x_ptr, shape=(N, N), strides=(N, 1),
                                offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
                                block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
    y_block = tl.make_block_ptr(base=y_ptr, shape=(N, N), strides=(N, 1),
                                offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
                                block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(y_block, x + 1.0, boundary_check=(0, 1))
    x_block = tl.advance(x_block, (BLOCK_M, 0))
```

## NPU 适配要点

1. **离散访存 vs 连续访存**：离散访存（gather/scatter 模式）导致更多内存事务与更高延迟；离散 mask 的 store 会被拆解为 atomic {load, select, store} 组合，corner case 可能有正确性问题。优先连续访存；`make_block_ptr` + `boundary_check` 通常比指针张量 + mask 更高效。
2. **care_padding 使用条件**：仅在后续计算不依赖 masked 位置的值时（如后续 mask 会过滤）才用 `care_padding=False` 换性能。
3. **ND2NZ 转换**：`make_block_ptr` 语义与 GPU 一致，但 NPU 底层可能自动插入 ND2NZ 数据格式转换，影响性能。
4. **不支持类型处理**：uint 系列 / fp64 需在 host 端转换或用 `tl.cast` 转为支持类型。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L1589-L1652](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1589-L1652)
- [core.py#L1678-L1726](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1678-L1726)
- [core.py#L1729-L1741](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1729-L1741)
- [core.py#L1744-L1753](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1744-L1753)
- [semantic.py#L1100-L1213](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1100-L1213)
- [semantic.py#L1253-L1340](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1253-L1340)
