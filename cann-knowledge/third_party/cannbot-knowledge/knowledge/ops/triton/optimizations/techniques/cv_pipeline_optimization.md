---
type: Optimization
title: CV 流水线优化（Cube-Vector 融合）
description: Cube/Vector 分离架构下的指令重排序、编译参数组合、fixpipe 数据通路与精度回退策略，覆盖 FlashAttention
  等矩阵乘+后处理融合算子。
tags: [task.optimization, topic.compute.composite, topic.implementation.pipeline, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/builder.py
  role: constraint
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/multibuffer.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# CV 流水线优化（Cube-Vector 融合）

触发条件：算子同时包含 `tl.dot`（Cube 矩阵乘）和向量运算。典型：FlashAttention（QK + Softmax + PV）、Matmul+Bias+GELU、Matmul+LayerNorm/Quantize。纯 Vector 或纯 Cube 算子无收益。

## Cube/Vector 分离架构

每个 AI Core = 1 个 Cube Core + 2 个 Vector Core（1:2 比例）：

| 硬件单元 | 执行运算 | 数据通路 |
|----------|----------|----------|
| Cube Core | 矩阵乘（`tl.dot`） | L1 → L0C |
| Vector Core | 逐元素、归约、激活 | UB |

```
GM --PIPE_MTE1--> L1 --Cube--> L0C --fixpipe(PIPE_FIX)--> UB -> Vector   (910_95，零 GM 访问)
                                 \--PIPE_FIX--> GM --PIPE_MTE2--> UB     (A2/A3，2 次 GM 访问)
```

CV 融合 vs 分离执行：分离执行 Cube→GM→Vector 需 2 次 GM 访问；CV 融合 Cube→L0C→UB→Vector 零 GM 访问（仅 910_95），代价是核心间同步开销。后处理很简单时同步开销可能抵消收益，建议 msprof 对比。

## 分核策略

CV 融合算子的 grid = **Cube 核数（`aicore_num`）**，不是 `vectorcore_num`；纯 Vector 算子用 `vectorcore_num`：

```python
import torch_npu
import triton.runtime.driver as driver

properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
grid = (properties["num_aicore"],)   # CV 融合算子
```

## 指令重排序：让 Cube 提前执行

核心思想：将无数据依赖的 Vector 计算延后，Cube 计算提前启动，实现 Cube/Vector 流水并行。FlashAttention 经典模式：

优化前（`tl.dot(p, v)` 被 Vector 计算阻塞）：

```python
m_new = tl.maximum(m, tl.max(s, 1))
alpha = tl.math.exp2((m - m_new) * qk_scale)
p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
p_sum = tl.sum(p, 1)
acc *= alpha[:, None]
v = load_if(v_block_ptr, False, True)
acc += tl.dot(p.to(dtype), v)
l = l * alpha + p_sum
```

优化后（`tl.dot(p, v)` 提前，与 `p_sum`/`alpha` 并行）：

```python
m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
v = load_if(v_block_ptr, False, True)
pv = tl.dot(p.to(dtype), v)              # Cube 提前启动
p_sum = tl.sum(p, 1)                     # 与 Cube 并行
alpha = tl.math.exp2((m - m_new) * qk_scale)
acc = acc * alpha[:, None] + pv          # 依赖 Cube 结果，放最后
l = l * alpha + p_sum
```

要点：重排只改变发射时机不改语义；`tl.max`/`tl.maximum` 加 `propagate_nan` 保持 NaN 传播（online softmax 中被 mask 为 -inf 的行可能产生 NaN，不传播会被静默忽略）。

## 编译参数组合

纯 Vector 算子：

```python
vector_kernel[grid](..., enable_flatten=True, multibuffer=True)
```

CV 融合算子：

```python
cv_kernel[grid](
    ...,
    enable_auto_bind_sub_block=True,
    enable_flatten=False,
    set_workspace_multibuffer=2,
    sync_solver=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    multibuffer=True,
    enable_mixed_cv=True,
)
```

### 精度回退策略（出现精度问题时按序逐项回退，每次只退一项并验证）

1. 移除 `enable_mixed_cv=True`
2. `sync_solver=True` → `False`
3. 移除 `limit_auto_multi_buffer_of_local_buffer="no-limit"`
4. 移除 `set_workspace_multibuffer=2`
5. `enable_auto_bind_sub_block=True` → `False`
6. `multibuffer=True` → `False`
7. 移除 `propagate_nan` 参数
8. 撤销指令重排序，恢复原始顺序

纯 Vector 算子只需回退 `enable_flatten` 和 `multibuffer`。

## fixpipe 与同步机制（CV 融合底层）

`al.fixpipe`：L0C→UB 直通数据通路（**仅 910_95/950**，A2/A3 需 GM 中转），支持 NZ2ND/NZ2DN/NZ2NZ 布局转换。详见 fixpipe_bias_fusion 卡。

`sync_block_set/wait/all` 核心间同步，sender/receiver 取 `"cube"/"vector"`（不能相同），`event_id` 0-15 共 16 个通道需避免冲突，set/wait 必须配对。`al.scope(core_mode="cube"/"vector")` 划定核心执行作用域。详见 sync_and_barrier 卡。

`al.sub_vec_id()` / `al.sub_vec_num()`：在 `core_mode="vector"` 作用域内获取当前 Vector 核索引（0/1）与数量（通常 2），用于两个 Vector 核分工：

```python
with al.scope(core_mode="vector"):
    al.sync_block_wait("cube", "vector", 0)
    vec_idx = al.sub_vec_id()
    rows_per_vec = BLOCK_M // al.sub_vec_num()
    start_row = vec_idx * rows_per_vec     # 每个 Vector 核处理一半行
```

`compile_hint("break_vf")`：打断过长的 Vector Function 融合链，防止 UB 溢出并确保后续 `tl.dot` 调度到 Cube：

```python
ds = ds.to(dtype)
extension.compile_hint(ds, "break_vf")
dq += tl.dot(ds, k)
```

## 910_95 限制清单

| 限制项 | 说明 |
|--------|------|
| fixpipe 仅 910_95/950 | A2/A3 需 GM 中转，性能损失较大 |
| 512B 对齐 | CV 融合算子要求 Tensor 尾轴大小能被 512B 整除 |
| sync_block event_id | 范围 0-15，同循环内不同同步点用不同 id |
| multibuffer 默认 False | 910_95 需显式 `multibuffer=True`（910B 默认 True） |

## 性能收益预估

| 优化手段 | 预估收益 |
|----------|----------|
| 指令重排序（CV 流水并行） | 10%-30% |
| 编译参数优化（multibuffer 等） | 5%-15% |
| fixpipe 数据通路（仅 910_95） | 10%-20% |
| 综合 | 20%-40% |

收益条件：Cube/Vector 计算比例接近 1:1 时收益最大；后处理越复杂收益越高。

不适用场景：纯 Vector 算子（无 Cube 计算）；矩阵乘无后处理；后处理依赖全局信息（跨核心同步开销大）。


## 相关源

- 主源：docs_triton_ascend/05-Performance-Optimization/05-cv-fusion.md
- 合并来源：docs_for_triton_agent/05-cv-pipeline-optimization.md（指令重排序、编译参数、精度回退）

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L273-L333](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L273-L333)
- [sync_block.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/sync_block.md)
- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py)
- [programming_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/programming_guide.md)
