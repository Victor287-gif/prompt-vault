---
type: API
title: 内存操作扩展（index_put / gather_out_to_ub / scatter_ub_to_out / index_select_simd）
description: GM↔UB 间的 SIMD 索引读写操作：index_put、gather_out_to_ub、scatter_ub_to_out、index_select_simd
  的签名、约束与边界检查语义。
tags: [task.lookup, topic.data_movement.memory_access, topic.compute.indexing, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/mem_ops.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# 内存操作扩展（index_put / gather_out_to_ub / scatter_ub_to_out / index_select_simd）

GM 与 UB 之间的高效索引读写操作，利用 SIMD 硬件实现 gather/scatter。与标准 `tl.load/store` + 指针算术不同，直接支持多维索引、边界检查和 UB 零拷贝语义，是注意力机制、嵌入查找等场景的关键组件。

## 操作对比与数据流向

| 操作 | 方向 | 模式 | 说明 |
|------|------|------|------|
| `index_put` | UB → GM | SIMD | 按索引将值写入 GM |
| `gather_out_to_ub` | GM → UB | SIMD | 按索引从 GM 收集到 UB |
| `scatter_ub_to_out` | UB → GM | SIMD | 按索引从 UB 散列到 GM |
| `index_select_simd` | GM → UB | SIMD | 并行索引选择（零拷贝） |

通用参数：`dim`（索引维度）、`index_boundary`（索引上界，硬件边界检查）、`src_stride`/`dst_stride`（各维步长，**int64**）、`start_offset`/`end_offset`（各维起止偏移，**int32**）。

## API 参考

### index_put — UB → GM 索引写入

```python
@_tensor_member_fn
@builtin
def index_put(ptr, index, value, dim, index_boundary,
              end_offset, start_offset, dst_stride, _builder=None)
```

| 参数 | 说明 |
|------|------|
| `ptr` | GM 目标张量指针 |
| `index` | UB 索引张量（整数；rank != 1 时自动 flatten 为 1D） |
| `value` | UB 值张量（2~5D） |
| `dim` | 散列维度，`0 <= dim < rank(value) - 1`（**不支持最后一维**） |

约束：`ptr` 与 `value` rank 相同；`ptr.dtype` **仅支持 fp16/bf16/fp32**；`index.numel == value.shape[dim]`；`end_offset`/`start_offset`/`dst_stride` 长度 == value rank。

语义（2D，dim=0）：`out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]`

### gather_out_to_ub — GM → UB 索引收集

```python
@_tensor_member_fn
@builtin
def gather_out_to_ub(src, index, index_boundary, dim,
                     src_stride, end_offset, start_offset, other=None, _builder=None)
```

| 参数 | 说明 |
|------|------|
| `src` | GM 源张量指针（dtype 仅 fp16/bf16/fp32） |
| `index` | UB 整数索引张量（1~5D） |
| `dim` | 收集维度，`0 <= dim < rank(index)` |
| `other` | 索引越界时的默认值（可选标量） |

约束：`src` 与 `index` rank 相同。返回形状同 `index.shape` 的 UB 张量。

语义（2D，dim=0）：`out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]`

### scatter_ub_to_out — UB → GM 索引散列

```python
@_tensor_member_fn
@builtin
def scatter_ub_to_out(ptr, value, index, index_boundary, dim,
                      dst_stride, end_offset, start_offset, _builder=None)
```

- `value` 可为 UB 张量**或标量**（标量时自动按 `index.shape` 与 `ptr.dtype` 创建全值张量）。
- 约束：`ptr`/`index`/`value` rank 相同；`index` 为 1~5D 整数；`0 <= dim < rank(index)`。

语义（2D，dim=0）：`out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]`

### index_select_simd — 并行索引选择（零拷贝）

```python
@_tensor_member_fn
@builtin
def index_select_simd(src, dim, index, src_shape, src_offset, read_shape, _builder=None) -> tensor
```

| 参数 | 说明 |
|------|------|
| `src` | GM 源张量指针 |
| `dim` | 索引选择维度（**不能是最后一维**） |
| `index` | 1D 索引张量（UB） |
| `src_shape` | 源张量完整形状（`List[int | tensor]`，支持动态形状） |
| `src_offset` | 各维读取起始偏移（`src_offset[dim]` 可为 `-1`，被忽略） |
| `read_shape` | 各维读取大小（`read_shape[dim]` 必须为 `-1`，由 index 长度决定） |

边界处理：`src_offset + read_shape > src_shape` 时自动截断到边界；**不检查 index 越界**（用户保证合法）。返回 UB 张量，dim 维被替换为 index 长度。

## 代码示例

### index_put

```python
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def simple_index_put_kernel(value_ptr, index_ptr, dst_ptr):
    index_local = tl.arange(0, 2)
    x1_local = tl.arange(0, 2)[None, :]
    index_tile = tl.load(index_ptr + index_local)
    value_tile = tl.load(value_ptr + index_local[:, None] * 2 + x1_local)
    al.index_put(ptr=dst_ptr, index=index_tile, value=value_tile, dim=0,
                 index_boundary=4, end_offset=(2, 2), start_offset=(0, 0), dst_stride=(2, 1))

dst = torch.zeros((4, 2), device='npu', dtype=torch.float32)
value = torch.tensor([[1., 2.], [3., 4.]], device='npu')
index = torch.tensor([2, 0], device='npu')
simple_index_put_kernel[(1,)](value, index, dst)
```

### gather_out_to_ub（带 other 默认值）

```python
gathered = al.gather_out_to_ub(
    src=src_ptr, index=index, index_boundary=10, dim=0,
    src_stride=(16,), end_offset=(4,), start_offset=(0,), other=0.0)
```

### index_select_simd（静态形状）

```python
@triton.jit
def index_select_static_kernel(src_ptr, output_ptr, indices_ptr):
    indices = tl.load(indices_ptr + tl.arange(0, 4))
    result = al.index_select_simd(
        src_ptr, dim=1, index=indices,
        src_shape=[8, 100, 256], src_offset=[4, -1, 128], read_shape=[4, -1, 128])
    tl.store(output_ptr + tl.arange(0, 4)[:, None] * 128 + tl.arange(0, 128)[None, :], result)
```

## NPU 适配要点

1. **GM/UB 区分**：`ptr`/`src` 是 GM 指针，`index`/`value` 在 UB，避免中间转储。
2. **参数整数类型**：stride 用 int64、offset 用 int32，与硬件指令参数要求一致。
3. **index_boundary**：越界时硬件特殊处理（用默认值或跳过）。
4. **gather_out_to_ub vs index_select_simd**：前者通用（多维索引+边界检查+other 默认值）；后者更高效零拷贝但限制多（index 必须 1D、dim 不能最后一维、不查越界）。
5. **dtype 限制**：`ptr.dtype` 仅 fp16/bf16/fp32，不支持整数类型直接 scatter/gather。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [mem_ops.py#L40-L177](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/mem_ops.py#L40-L177)
- [mem_ops.py#L180-L329](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/mem_ops.py#L180-L329)
- [mem_ops.py#L332-L482](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/mem_ops.py#L332-L482)
- [mem_ops.py#L485-L636](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/mem_ops.py#L485-L636)
- [builtin_custom_ops.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/builtin_custom_ops.py)
