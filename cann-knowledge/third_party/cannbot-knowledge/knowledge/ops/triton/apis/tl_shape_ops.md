---
type: API
title: 张量形状操作 API（broadcast / reshape / trans / join / split 等）
description: Triton-Ascend 形状操作 API 签名与约束：broadcast、cat、reshape、trans/permute、expand_dims、join/split、ravel、flip、interleave；can_reorder
  语义与 NPU 性能影响。
tags: [task.lookup, topic.data_movement.layout_transform, topic.compute.indexing]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# 张量形状操作 API

形状操作在 NPU 上行为与 GPU 基本一致；`trans`/`permute` 是元数据操作（不改步幅不搬数据），但后续访问模式影响性能。

## API 签名与约束

| API | 签名 | 关键约束 |
|-----|------|---------|
| `tl.broadcast(input, other)` | → `(tensor, tensor)` | 双张量广播到兼容形状（右对齐，维度 1 可扩展） |
| `tl.broadcast_to(input, *shape)` | → tensor | 广播到指定形状 |
| `tl.cat(input, other, can_reorder=False)` | → tensor | **当前仅支持 `can_reorder=True`**，输入必须 1D；元素顺序可能与输入不一致 |
| `tl.reshape(input, *shape, can_reorder=False)` | → tensor | 元素总数不变；`can_reorder=True` 允许编译器重排元素优化布局 |
| `tl.view(input, *shape)` | → tensor | **已弃用**，等价 `reshape(..., can_reorder=False)`，新代码用 reshape |
| `tl.trans(input, *dims)` | → tensor | 不传 dims 默认 `(1, 0)` 2D 转置 |
| `tl.permute(input, *dims)` | → tensor | dims 必须是 `0..n-1` 的排列；**NPU 不支持不相邻轴转置** |
| `tl.expand_dims(input, axis)` | → tensor | 插入长度 1 维度；不允许重复 axis |
| `tl.join(a, b)` | → tensor | 先广播到同形，结果形状 `a.shape + [2]`；split 的逆操作 |
| `tl.split(a)` | → `(tensor, tensor)` | 最后一维大小必须为 2 |
| `tl.ravel(x)` | → tensor | 等价 `reshape(x, [x.numel], can_reorder=False)` |
| `tl.flip(x, dim=None)` | → tensor | **仅支持最后一维**；翻转维度大小与 `x.numel` 必须是 2 的幂（bitonic 实现） |
| `tl.interleave(a, b)` | → tensor | 等价 `join(a,b).reshape(a.shape[:-1] + [2*a.shape[-1]])`；两输入同形 |

## can_reorder 语义（reshape / cat）

- `False`（默认）：保持元素逻辑顺序，编译器可能插入额外 UB 数据搬运
- `True`：允许重排元素优化数据布局，避免不必要的搬运；仅当后续操作不依赖元素顺序时安全（如 reshape 后立即归约，或元素将被原子写入不同位置）

## NPU 适配要点

1. **trans/permute 是元数据操作**：不移动数据；但连续访问转置后张量 = 对原张量跨步访问，NPU 跨步访问效率低于连续访问。建议在矩阵乘前用 `trans` 调整布局让 Cube 获得连续输入。
2. **多维 cat 替代方案**：`cat` 仅支持 1D；多维拼接先 reshape 为 1D、cat、再 reshape 回去，或用 `join` 沿新维度连接。
3. **i64 限制**：`tl.permute`/`tl.trans` 不支持 int64（见支持矩阵）。

## 代码示例

```python
@triton.jit
def join_split_kernel(a_ptr, b_ptr, out_a_ptr, out_b_ptr,
                      N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    joined = tl.join(a, b)          # [BLOCK, 2]
    processed = joined * 2.0
    out_a, out_b = tl.split(processed)
    tl.store(out_a_ptr + offsets, out_a, mask=mask)
    tl.store(out_b_ptr + offsets, out_b, mask=mask)
```

```python
# 转置后乘（trans 是元数据操作，不搬数据）
x_t = tl.trans(x)
result = x_t * y

# reshape 后立即归约时可安全开启 can_reorder，避免多余 UB 搬运
flat = tl.reshape(x, (BLOCK_M * BLOCK_N,), can_reorder=True)
s = tl.sum(flat, axis=0)
```

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L1240-L1271](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1240-L1271)
- [core.py#L1274-L1323](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1274-L1323)
- [core.py#L1407-L1447](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1407-L1447)
- [core.py#L1457-L1484](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1457-L1484)
- [core.py#L1326-L1404](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1326-L1404)
- [semantic.py#L796-L846](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L796-L846)
- [semantic.py#L768-L775](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L768-L775)
- [standard.py#L60-L561](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/standard.py#L60-L561)
