---
type: API
title: 归约操作 API（reduce / sum / max / argmax / xor_sum）
description: Triton-Ascend 归约 API 签名与参数语义；NPU 上窄整数不提升 int32 避免 UB 溢出、bf16 提升 fp32、argmax
  NaN 行为等适配差异。
tags: [task.lookup, topic.compute.reduction, topic.programming_model.execution_unit.vector, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/sum.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# 归约操作 API

NPU 上归约由 Vector 单元硬件归约指令执行，数据与中间结果均在 UB 中，归约范围过大可能 UB 溢出。

## tl.reduce

```python
triton.language.reduce(input, axis, combine_fn, keep_dims=False) -> tensor
```

| 参数 | 说明 |
|------|------|
| `input` | 张量或张量元组（多输出归约） |
| `axis` | 归约轴；None 归约所有维度；支持负数轴 |
| `combine_fn` | 必须 `@triton.jit`，接受两组标量返回组合结果；**必须满足结合律**（不要求交换律） |
| `keep_dims` | 归约维度保留为长度 1，默认 False |

## tl.sum / tl.max / tl.min

```python
tl.sum(input, axis=None, keep_dims=False)
tl.max(input, axis=None, return_indices=False, return_indices_tie_break_left=True,
       keep_dims=False, propagate_nan=False)
tl.min(input, axis=None, return_indices=False, return_indices_tie_break_left=True,
       keep_dims=False)
```

| 参数 | 说明 |
|------|------|
| `return_indices` | 同时返回最值索引，返回 `(值, 索引)` 元组 |
| `return_indices_tie_break_left` | 多个相等最值时返回最左索引（默认 True；False 为快速模式不保证索引） |
| `propagate_nan` | True 时底层用 `maximum`（传播 NaN）；False 用 `maxnumf`（NaN 不参与比较） |

**NPU 类型提升**：
- bf16 输入自动提升 fp32 执行
- **窄整数 int8/int16 不提升 int32**（避免 UB 溢出），结果可能溢出，需用户保证数值范围或先 `x.to(tl.int32)`
- 位宽 < 32 的浮点类型提升 fp32

## tl.argmax / tl.argmin

```python
tl.argmax(input, axis, tie_break_left=True, keep_dims=False)
tl.argmin(input, axis, tie_break_left=True, keep_dims=False)
```

`axis` 必须指定。实现为 `max(input, axis, return_indices=True)[1]`。NaN 行为：`propagate_nan=False`（max 默认）时 NaN 不会被选为最值。

## tl.xor_sum

```python
tl.xor_sum(input, axis=None, keep_dims=False)
```

**仅支持整数类型**，浮点输入抛 `ValueError`。

## 代码示例

```python
@triton.jit
def argmax_kernel(x_ptr, idx_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
    idx = tl.argmax(x, axis=0)
    tl.store(idx_ptr + pid, idx)

@triton.jit
def combine(a, b):
    return a + b * b

@triton.jit
def custom_reduce_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    result = tl.reduce(x, axis=0, combine_fn=combine)
    tl.store(out_ptr + pid, result)
```

## NPU 适配要点

1. **int8/int16 溢出风险**：NPU 跳过了窄整数到 int32 的提升（GPU 会提升防溢出，NPU 为省 UB 不提升）。安全做法：`tl.sum(x.to(tl.int32), axis=0)`。
2. **归约仅在单 program 内**：跨 block 归约需两阶段 kernel（局部归约 store → 二次归约）或用 atomic 操作。
3. **fp32 累加器**：归约中间结果用 fp32 避免精度损失。
4. **UB 容量**：归约数据先加载到 UB，大张量归约需关注 UB 占用。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [standard.py#L136-L302](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/standard.py#L136-L302)
- [core.py#L2047-L2096](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L2047-L2096)
- [core.py#L2100-L2106](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L2100-L2106)
- [core.py#L2109](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L2109)
