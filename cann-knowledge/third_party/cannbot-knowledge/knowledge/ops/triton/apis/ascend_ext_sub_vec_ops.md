---
type: API
title: sub_vec_id / sub_vec_num 子向量核操作
description: 获取 AI Core 内子 Vector 核编号与数量，用于 CV 融合算子中多 Vector 核心的负载划分（Cube:Vector =
  1:2）。
tags: [task.lookup, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.cube, topic.parallelism]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]

---
# sub_vec_id / sub_vec_num 子向量核操作

昇腾 NPU 一个 AI Core 含 1 个 Cube 核心和 2 个 Vector 核心（Cube:Vector = 1:2）。`sub_vec_id()` 返回当前 Vector 核编号（0 或 1），`sub_vec_num()` 返回每 AI Core 的 Vector 核数（通常 2），是多 Vector 核并行计算的基础。

## 硬件结构

```
┌─────────────────────────────────────────┐
│              AI Core                     │
│  ┌──────────┐  ┌──────────┐             │
│  │  Cube    │  │ Vector 0 │ (sub_vec_id=0) │
│  │  (1个)   │  ├──────────┤             │
│  │          │  │ Vector 1 │ (sub_vec_id=1) │
│  └──────────┘  └──────────┘             │
│  sub_vec_num() = 2                       │
└─────────────────────────────────────────┘
```

CV（Cube-Vector）融合算子中：Cube 负责矩阵乘法，多个 Vector 核并行负责后处理（Softmax、量化、激活），通过 `sub_vec_id()` 区分负载、通过 `sync_block_set/wait` 协调执行。

## API 参考

### sub_vec_id — 当前 Vector 核编号

```python
@builtin
def sub_vec_id(_builder=None) -> tl.tensor:
    """Get the Vector Core index on the AI Core."""
```

返回 `tl.tensor`（int64），值 0 或 1，可参与 kernel 内计算与条件判断。

### sub_vec_num — Vector 核数量

```python
@builtin
def sub_vec_num(_builder=None) -> tl.constexpr:
    """Get the Vector Core Num on one AI Core."""
```

返回 `tl.constexpr`（编译期常量）。实现上通过 `NPUUtils` 在编译时查询硬件信息：`aivector_core_num / aicore_num`，可适应不同硬件比例平台。

## 代码示例

### CV 融合算子中的负载划分

```python
@triton.jit
def cv_fusion_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    with al.scope(core_mode="cube"):
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + ...)
            b = tl.load(b_ptr + ...)
            acc = tl.dot(a, b, acc)
        ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
        al.fixpipe(acc, ub_buf)
        al.sync_block_set("cube", "vector", 0)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        vec_id = al.sub_vec_id()
        vec_num = al.sub_vec_num()

        half_m = BLOCK_M // vec_num
        start_row = vec_id * half_m
        end_row = start_row + half_m

        acc_tensor = bl.to_tensor(ub_buf)
        my_slice = acc_tensor[start_row:end_row, :]
        bias_slice = tl.load(bias_ptr + tl.arange(0, BLOCK_N))
        result = my_slice + bias_slice
        tl.store(c_ptr + ..., result)
```

### parallel 循环中的 bind_sub_block

```python
@triton.jit
def parallel_sub_vec_kernel(data_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    with al.scope(core_mode="vector"):
        vec_num = al.sub_vec_num()
        for i in al.parallel(0, N, BLOCK_N, bind_sub_block=True):
            vec_id = al.sub_vec_id()
            offset = i + vec_id * BLOCK_N
            data = tl.load(data_ptr + offset + tl.arange(0, BLOCK_N))
            result = data * 2.0
            tl.store(out_ptr + offset + tl.arange(0, BLOCK_N), result)
```

### 动态负载分配

```python
@triton.jit
def dynamic_workload_kernel(input_ptr, output_ptr, total_elements, BLOCK_SIZE: tl.constexpr):
    with al.scope(core_mode="vector"):
        vec_id = al.sub_vec_id()
        vec_num = al.sub_vec_num()
        per_vec = total_elements // vec_num
        my_start = vec_id * per_vec
        my_end = my_start + per_vec
        for offset in range(my_start, my_end, BLOCK_SIZE):
            idx = offset + tl.arange(0, BLOCK_SIZE)
            mask = idx < my_end
            data = tl.load(input_ptr + idx, mask=mask)
            result = tl.sigmoid(data)
            tl.store(output_ptr + idx, result, mask=mask)
```

## NPU 适配要点

1. **仅限 Vector 上下文**：应在 `al.scope(core_mode="vector")` 中调用；Cube 上下文中调用 `sub_vec_id()` 行为未定义。
2. **返回类型差异**：`sub_vec_id()` 返回运行时 `tl.tensor`（int64）；`sub_vec_num()` 返回编译期 `tl.constexpr`。
3. **bind_sub_block 配合 parallel**：`al.parallel(..., bind_sub_block=True)` 让编译器把循环迭代分配给不同子 Vector 核（910B 最多 2 个）。
4. **负载均衡**：多 Vector 核场景需均匀划分数据量，避免负载不均浪费性能。
5. **防写冲突**：通过 `sub_vec_id()` 保证各核写不同内存区域；可用 `sync_block_all("all_sub_vector", event_id)` 同步所有子 Vector 核。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L166-L171](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L166-L171)
- [core.py#L359-L368](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L359-L368)
- [semantic.py#L90-L91](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/semantic.py#L90-L91)
