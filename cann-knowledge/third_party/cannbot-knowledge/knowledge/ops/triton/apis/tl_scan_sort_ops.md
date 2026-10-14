---
type: API
title: 扫描与排序操作 API（associative_scan / cumsum / histogram / sort）
description: Triton-Ascend 扫描与排序 API 签名与约束：associative_scan、cumsum/cumprod、histogram、sort（bitonic
  实现，2 的幂限制）及 NPU 性能特征。
tags: [task.lookup, topic.compute.ordering, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Scan_Sort_Ops
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Scan_Sort_Ops/sort.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# 扫描与排序操作 API

扫描（前缀）与排序操作在 NPU 上由 Vector 单元执行，中间结果存 UB；扫描 O(log n) 步每步读写 UB，比逐元素操作慢得多，大张量下可能成为瓶颈。

## tl.associative_scan

```python
triton.language.associative_scan(input, axis, combine_fn, reverse=False) -> tensor
```

| 参数 | 说明 |
|------|------|
| `input` | 张量或张量元组（多输出扫描），所有输入形状相同 |
| `axis` | 扫描轴 |
| `combine_fn` | 必须 `@triton.jit`，两标量入参返回组合结果；**必须满足结合律** |
| `reverse` | True 沿轴反向扫描 |

语义：`result[i] = f(f(...f(x[0], x[1])...), x[i])`（前缀累积）。

## tl.cumsum / tl.cumprod

```python
tl.cumsum(input, axis=0, reverse=False)
tl.cumprod(input, axis=0, reverse=False)
```

分别等价 `associative_scan(input, axis, lambda a,b: a+b / a*b, reverse)`。**NPU 注意**：bf16 输入自动提升 fp32 执行。反向扫描性能与正向对称。

## tl.histogram

```python
triton.language.histogram(input, num_bins) -> tensor  # 返回 (num_bins,) int32
```

**约束**：输入必须 1D 整数张量；桶宽 1、起始 0（第 i 桶统计值为 i 的元素数）；NPU 上仅 i32/i64（见支持矩阵）。浮点数据需先离散化（如 `(x * scale).to(tl.int32)`），或用 atomic_add 统计。

## tl.sort

```python
triton.language.sort(x, dim=None, descending=False) -> tensor
```

**约束**：
- **仅支持沿最后一维排序**（dim=None 时取最后一维）
- 排序维度大小必须是 **2 的幂**（bitonic merge sort 实现）
- **NPU 当前版本不可用**（毕昇编译器限制，见支持矩阵）

算法：reshape 为超立方体 `[2,2,...,2]` → 迭代 bitonic merge（多轮 compare-and-swap）→ reshape 回原形。时间复杂度 O(n log²n)，原地排序 O(1) 额外空间。非 2 幂数据可填充到最近 2 的幂，排序后截取。

## 代码示例

```python
@triton.jit
def cumsum_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def scan_combine(a, b):
    return a + b

@triton.jit
def custom_scan_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.associative_scan(x, axis=0, combine_fn=scan_combine)
    tl.store(out_ptr + offsets, result, mask=mask)
```

## NPU 适配要点

1. **cumsum vs reduce**：cumsum 保留全部前缀结果（输出同形）；reduce 只返回最终值。
2. **反向 cumsum（后缀和）**：`tl.cumsum(x, axis=0, reverse=True)`。
3. **combine_fn 结合律**：不满足结合律会导致错误结果（执行顺序由编译器决定）。
4. **性能**：大张量扫描是多轮 UB 读写，可能成为瓶颈；优先考虑能否用归约替代。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L2150-L2200](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L2150-L2200)
- [core.py#L2190-L2200](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L2190-L2200)
- [standard.py#L305-L331](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/standard.py#L305-L331)
- [standard.py#L430-L451](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/standard.py#L430-L451)
- [semantic.py#L1792-L1809](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1792-L1809)
- [semantic.py#L1840-L1843](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1840-L1843)
