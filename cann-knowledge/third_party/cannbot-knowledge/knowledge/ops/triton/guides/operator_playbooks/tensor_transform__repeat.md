---
type: Guide
title: Repeat 算子（transformation-memory） 实现手册
description: '算子类别: transformationmemory。'
tags: [task.implementation, topic.data_movement.layout_transform, topic.tiling, topic.implementation.masking, topic.compilation, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Repeat 算子（transformation-memory） 实现手册

> 类别约束见 [tensor_transform_constraints.md](../constraints/tensor_transform_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §4 Repeat 算子（transformation-memory）

**算子类别**: `transformation-memory`
**典型特征**: 数据搬运为主，计算极简（仅坐标映射），输出 shape != 输入 shape

### §4.0 关键区分（必须先看懂）

Repeat 类算子在 PyTorch 里有两个不同 API，优化策略完全不同：

| API | 语义 | 典型调用 |
|-----|------|---------|
| `torch.repeat` / `Tensor.repeat` | 按**维度**重复整个张量 | `x.repeat(2, 1, 3)` |
| `torch.repeat_interleave` | 沿**单一维度**，把每个元素重复 N 次 | `x.repeat_interleave(2, dim=-1)` |

**性能基准**：
- `torch.repeat`: 几何平均加速比 **0.88x**，49/49 cases 全过（`repeat_v2_20260526`）
- `repeat_interleave`: 几何平均加速比 **1.50x**，75 cases 全过

> ⚠️ **禁止混淆**: Layer 2 以下方案按算子类型选择。`torch.repeat` 不要套用 `repeat_interleave` 的逐元素展开思路。

### §4.1 Layer 1: 设计约束

#### L1.1 禁止单 kernel 展平多维 repeat
（特化分派原则见 §1 G3）
- **禁止**用单一 1D kernel 处理多维 `torch.repeat`（如 `Tensor.repeat([2,3,4])`）
- 单 kernel 展平会导致坐标解码 overhead 极大，且无法利用多维并行
- **必须**按维度逐层串行处理，每层一个 kernel
- **注意**：`repeat_interleave(input, repeats, dim)` 只沿单一维度重复，不在此限制范围内

#### L1.2 必须使用 constexpr 循环展开
- **必须**将 repeat 次数声明为 `tl.constexpr`

#### L1.3 禁止在 kernel 内对向量做整数除法/取余
- **禁止**在 Triton kernel 内对 `tl.arange` 产生的向量做 `//` 或 `%` 运算
- Ascend NPU 上 int32 向量除法退化为**标量循环**，BLOCK=1024 时每个 block 执行 1024 次标量除法
- 正确方案：用标量索引做除法（如 `block_idx // num_inner_blocks`），或对连续内存块直接拷贝

#### L1.4 多核分配必须按输出元素
（见 §1 G6）
- **必须**按输出元素总数分配核数，而非按输入元素
- 每个 program 处理一段连续的输出元素

#### L1.5 输入必须 contiguous
（见 §1 G7）

### §4.2 Layer 2: 算法骨架

#### L2.0 子类型判定

根据任务描述判断：
- 若语义是 `Tensor.repeat(repeats)` → 走 **L2.1 torch.repeat 推荐架构**（outer×inner 连续块）
- 若语义是 `repeat_interleave(input, repeats, dim)` → 走 **L2.4 repeat_interleave 架构**

#### L2.1 torch.repeat 推荐架构：outer×inner 连续块路径（优先）

这是经过验证的**最优架构**（`repeat_v2_20260526`，0.88x）。核心思想：把 repeat 问题转换为"按 outer_idx 加载一段连续 inner_size 数据，然后连续写 r 次"。

**Host 侧流程**：

```python
def forward(self, x, repeats):
    x = x.contiguous()
    shape = list(x.shape)
    ndim = len(shape)

    # 将 repeats 扩展到与 ndim 相同长度（前面补 1）
    repeats = [1] * (ndim - len(repeats)) + list(repeats)
    out = x

    # 从最低维到最高维逐维度处理
    for dim_idx in range(ndim - 1, -1, -1):
        r = repeats[dim_idx]
        if r <= 1:
            continue

        outer_size = math.prod(shape[:dim_idx])
        inner_size = out.numel() // outer_size

        # 选择 BLOCK：按 inner_size 向上取 2 的幂，目标使 num_inner_blocks 尽量小
        BLOCK = _get_block_size(inner_size)
        num_inner_blocks = (inner_size + BLOCK - 1) // BLOCK
        total_blocks = outer_size * num_inner_blocks

        # 构造输出 tensor
        out_shape = list(out.shape)
        out_shape[dim_idx] *= r
        output = torch.empty(out_shape, dtype=out.dtype, device=out.device)

        # Grid 分发
        if total_blocks <= VEC_CORE_NUM:
            grid = (outer_size, num_inner_blocks)   # 2D grid
        else:
            grid_cores = total_blocks if total_blocks < VEC_CORE_NUM else VEC_CORE_NUM
            grid = (grid_cores,)                     # 1D grid，按 core 负载均衡

        launch_kernel(out, output, outer_size, inner_size, num_inner_blocks,
                      r=r, BLOCK=BLOCK, num_cores=grid_cores)

        out = output
        shape[dim_idx] *= r

    return out
```

**Kernel 侧（两种 grid 模式）**

**模式 A：Small Grid（2D grid）**

```python
@triton.jit
def repeat_small_kernel(x_ptr, out_ptr, inner_size, r: tl.constexpr, BLOCK: tl.constexpr):
    outer_idx = tl.program_id(0).to(tl.int32)
    local_block = tl.program_id(1).to(tl.int32)

    block_start = local_block * BLOCK
    offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
    mask = offs < inner_size

    in_offset = outer_idx * inner_size
    val = tl.load(x_ptr + in_offset + offs, mask=mask)

    # r 为 constexpr，编译期展开
    for repeat_idx in range(r):
        out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
        tl.store(out_ptr + out_offset + offs, val, mask=mask)
```

**模式 B：Large Grid（1D grid + 标量循环分配）**

```python
@triton.jit
def repeat_large_kernel(x_ptr, out_ptr, outer_size, inner_size, num_inner_blocks,
                        r: tl.constexpr, BLOCK: tl.constexpr, num_cores: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    total_blocks = outer_size * num_inner_blocks

    blocks_per_core = total_blocks // num_cores
    remainder = total_blocks - blocks_per_core * num_cores

    if pid < remainder:
        my_blocks = blocks_per_core + 1
        start_block = pid * (blocks_per_core + 1)
    else:
        my_blocks = blocks_per_core
        start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core

    for block_idx in range(start_block, start_block + my_blocks):
        outer_idx = block_idx // num_inner_blocks          # 标量除法
        local_block = block_idx - outer_idx * num_inner_blocks

        block_start = local_block * BLOCK
        offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
        mask = offs < inner_size

        in_offset = outer_idx * inner_size
        val = tl.load(x_ptr + in_offset + offs, mask=mask)

        for repeat_idx in range(r):
            out_offset = outer_idx * inner_size * r + repeat_idx * inner_size
            tl.store(out_ptr + out_offset + offs, val, mask=mask)
```

**为什么这个架构快？**

1. **输入输出都连续**
   - 输入：每个 program 加载 `inner_size` 中一段连续数据
   - 输出：`outer_idx * inner_size * r + repeat_idx * inner_size + offs`
   - r 个副本在输出内存中**紧挨着**，无跨步写

2. **Grid 并行度高**
   - `total_blocks = outer_size * num_inner_blocks`
   - 2D grid 或按 core 数分发，充分利用多核

3. **无向量除法**
   - 所有除法/取余都是标量（`block_idx // num_inner_blocks`）
   - `block_idx` 是 program 内标量，不触发向量降级

4. **从低维到高维处理**
   - 每次只扩展当前最低维
   - 保持 `inner_size` 是一段逻辑上连续的数据

#### L2.2 torch.repeat 备选架构：lastdim + slice 路径

当实现环境受限，或外层循环已固定为高维到低维时，可作为备选。但**不是首选**，性能通常不如 L2.1。

**Lastdim 路径（零除法）**

```python
val = tl.load(in_ptr + in_offs, mask=in_mask, other=0.0)
val_2d = tl.broadcast_to(tl.reshape(val, (BLOCK_IN, 1)), (BLOCK_IN, repeats))
out_val = tl.reshape(val_2d, (BLOCK_IN * repeats,))
tl.store(out_ptr + out_offs, out_val, mask=out_mask)
```

**Slice 路径（标量除法）**

```python
for s in range(SLICES_PER_BLOCK):
    slice_idx = block_start_slice + s
    if slice_idx < num_out_slices:
        in_slice_idx = slice_idx // repeats          # 标量除法
        out_base = slice_idx * post_size
        in_base = in_slice_idx * post_size
        for p_start in range(0, post_size, BLOCK_POST):
            p_off = p_start + tl.arange(0, BLOCK_POST)
            val = tl.load(in_ptr + in_base + p_off, ...)
            tl.store(out_ptr + out_base + p_off, val, ...)
```

> ⚠️ **注意**：slice 路径的缺点是同一个输入 slice 的多个副本在输出内存中**不连续**（间隔 `DIM_SIZE * POST_SIZE`），会导致跨步写。只有在 `post_size` 很小或 `repeats=1` 时才能接受。

#### L2.3 repeat_interleave 架构

`repeat_interleave(input, repeats, dim)` 只沿单一维度重复，可直接用一个 kernel 处理。

```python
# 输出坐标 out_idx 映射到输入坐标 in_idx = out_idx // repeats
# 或对于可变 repeats，先前缀和再二分查找
```

### §4.3 Layer 3: 关键技巧

#### L3.1 BLOCK 选择公式

```python
def _get_block_size(inner_size: int) -> int:
    """按 inner_size 向上取 2 的幂，目标使 num_inner_blocks 尽量小。"""
    if inner_size <= 64:   return 64
    if inner_size <= 128:  return 128
    if inner_size <= 256:  return 256
    if inner_size <= 512:  return 512
    if inner_size <= 1024: return 1024
    if inner_size <= 2048: return 2048
    if inner_size <= 4096: return 4096
    return 8192
```

原则（pow2 BLOCK 见 §1 G2）：
- BLOCK 不超过 inner_size 太多（避免 mask 浪费）
- num_inner_blocks 尽量小，减少 kernel 内循环次数

#### L3.2 Grid 选择阈值

```python
import torch_npu
import triton.runtime.driver as driver

VEC_CORE_NUM = driver.active.utils.get_device_properties(torch_npu.npu.current_device())["num_vectorcore"]

if total_blocks <= VEC_CORE_NUM:
    grid = (outer_size, num_inner_blocks)   # 2D grid，每个 block 处理一个 outer × inner_block
else:
    grid_cores = min(total_blocks, VEC_CORE_NUM)
    grid = (grid_cores,)                     # 1D grid，按 core 负载均衡
```

#### L3.3 int32 索引
（见 §1 G5）

#### L3.4 负载均衡公式
（见 §1 G6）

### §4.4 Repeat 性能基准
- `torch.repeat`: **0.88x** 几何平均，49/49 cases 全过
- `repeat_interleave`: **1.50x** 几何平均，75 cases 全过

---
