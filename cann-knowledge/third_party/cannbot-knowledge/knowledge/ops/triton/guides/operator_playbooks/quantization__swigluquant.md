---
type: Guide
title: SwigluQuant 算子（quantization-activation） 实现手册
description: '算子类别: quantizationactivation。'
tags: [task.implementation, topic.compute.quantization, topic.compute.elementwise.activation, topic.implementation.masking, topic.tiling, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# SwigluQuant 算子（quantization-activation） 实现手册

> 类别约束见 [quantization_constraints.md](../constraints/quantization_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 SwigluQuant 算子（quantization-activation）

**算子类别**: `quantization-activation`
**典型特征**: 对 SwiGLU 激活结果做逐行静态或动态 INT8/INT4 量化，支持 smooth scales、offsets、group index
**性能基准**: 52 cases 精度全过；纯 Triton 实现几何平均 **0.5067x** vs torch，implementation latency 约 **0.265 ms**

> **评判口径**: framework 参考实现为高度优化的 `torch_npu.npu_swiglu_quant`，`speedup_vs_torch < 1` 属于正常现象；优化目标应聚焦在 **implementation latency** 降低，同时保证 52/52 精度全过。

### §3.0 算子语义

**输入**:
- `x`: 任意前缀 shape + 最后一维 `2 * half_dim` 的激活张量
- `smooth_scales`: 可选，1D `[num_groups]` 或 2D `[num_groups, half_dim]` 缩放
- `offsets`: 可选，仅 static 模式有效，1D 或 2D 偏移
- `group_index`: 可选，映射每行属于哪个 group
- `activate_left`: bool，决定 SwiGLU 用 `sigmoid(left) * right` 还是 `sigmoid(right) * left`
- `quant_mode`: 0=static, 1=dynamic
- `dst_type`: `torch.int8` 或 `torch.quint4x2`

**输出**:
- `out_quant`: 量化后的 int8/int4 张量
- `out_scales`: dynamic 模式返回每行 scale；static 模式返回零张量占位

### §3.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须拆分 static / dynamic 两个 kernel
- **必须**为 `quant_mode=0`（static）和 `quant_mode=1`（dynamic）分别编写 `@triton.jit` kernel，host 侧通过 `_route` 二选一启动
- **Why:**
  - static 模式直接量化，dynamic 模式需要 Pass 1 求 `max_abs`、Pass 2 量化
  - 合并为单一 kernel 会在内部引入大量 `tl.where(quant_mode==1, ...)` 分支，向量利用率低、编译器难以优化
- **How to apply:** 所有 SwigluQuant 场景

#### L1.2 必须使用 1D Grid + 核内循环，且 grid 必须钳制到 VEC_CORE_NUM
（通用模式见 §1 Q2）
- **必须** `grid = (min(M, VEC_CORE_NUM),)`，kernel 内 `for row in range(pid, M, num_programs)` 循环处理多行
- **禁止** `grid = (M,)`（每行一个 program，M 大时严重过饱和，全部串行排队，实测性能 0.01x，损失 10-50 倍）
- **禁止** 多维 grid（未带来收益，增加调度复杂度）
- static kernel 内部按 `rows_per_prog` 循环处理多行；dynamic kernel 内部按 `tiles_per_prog` 循环处理多个 tile
- **正确骨架**：
  ```python
  VEC_CORE_NUM = triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)
  grid = (min(M, VEC_CORE_NUM),)
  kernel[grid](..., num_cores=VEC_CORE_NUM, ...)
  # kernel 内：
  for row in range(pid, M, num_programs):
      ...
  ```
- **Why:** grid=(M,) 当 M=数百~数千 token 时远超 vector core 数（~40），program 全部排队串行，无核内循环摊销。grid 钳制 + 核内循环是本算子脱离 0.01x 的单点关键修复。

#### L1.3 禁止在 kernel 内或 forward 中调用 torch / torch_npu 计算
（见 §1 Q6）
- **禁止**使用 `torch_npu.npu_swiglu_quant` 预计算 dynamic scale
- **Why:** 违反 Triton-Ascend 算子实现约束；虽然可能提升 `speedup_vs_torch`，但失去纯 Triton 意义，无法沉淀为可复用 kernel
- **How to apply:** dynamic scale 必须在 Triton kernel 内部通过 `tl.max` + vector division 计算

#### L1.4 禁止 `tl.sigmoid`，必须手写 sigmoid
- **必须**使用 `x / (1.0 + tl.exp(-x))`
- **禁止**使用 `tl.sigmoid`
- **Why:** `tl.sigmoid` 在 Ascend 后端会产生系统性数值偏差，导致大量 case 精度 fail
- **How to apply:** 所有场景

#### L1.5 INT4 输出必须按 (odd << 4) | even 打包
- **必须**将相邻两个 INT8 值各取低 4 位：`packed = ((q_odd & 0x0F) << 4) | (q_even & 0x0F)`
- **Why:** 与 `torch.quint4x2` 存储布局一致
- **How to apply:**
  ```python
  q_2d = tl.reshape(q_val, (BLOCK_SIZE_COL // 2, 2))
  q_even, q_odd = tl.split(q_2d)
  packed = ((q_odd & 0x0F) << 4 | (q_even & 0x0F)).to(tl.int8)
  ```

#### L1.6 dynamic 模式下 scale 必须使用 vector division 路径
- **必须**先按行将 `max_abs` 存入全局内存，再以 tile 为单位 load 一个短向量 `max_vec`，执行 `int_scale / max_vec`
- **禁止**在量化的内层循环里对每行单独做标量除法 `int_scale / max_abs`
- **Why:**
  - Triton-Ascend 标量除法与向量除法的 ULP 行为不同
  - 标量路径会在大 fp32 dynamic-int8 shape 上产生 off-by-1
- **How to apply:**
  - Pass 1: 每行写 `scales_ptr[row_idx] = max_abs`
  - tile 级别：`scale_vec = int_scale / tl.maximum(max_vec, 1e-10)`，写回 `scales_ptr`
  - Pass 2: `row_scale = tl.load(scales_ptr + row_idx)`，再 `nearbyint(swiglu * row_scale)`

#### L1.7 clip 必须在 int32 阶段使用 `tl.maximum/tl.minimum`
- **必须**按 `nearbyint → .to(tl.int32) → tl.maximum/tl.minimum → .to(tl.int8)` 顺序执行
- **禁止**对 int32 使用 `tl.clamp`
- **Why:** `tl.clamp` 仅支持浮点；`tl.maximum/tl.minimum` 在 int32 上效率更优
- **How to apply:** 所有量化路径

#### L1.8 Grid 大小必须动态读取实际 vector core 数量
（见 §1 Q1）
- **禁止**硬编码 `num_cores`

#### L1.9 禁止在 kernel 内使用 `continue` / `break`
（见 §1 Q7）

#### L1.10 多维输入必须在 host 侧扁平化为 2D
（见 §1 Q5）
- **必须**在 `forward()` 中将输入 `x` 通过 `x.view(prefix_dims, 2 * half_dim)` 扁平化

#### L1.11 offsets 仅在 static 模式下有效
- **必须**在 host 侧设置 `has_offsets = offsets is not None and quant_mode == 0`
- **Why:** dynamic 模式不存在 offsets 语义，错误启用会导致精度失败或多余计算
- **How to apply:** host 侧分支决策时处理

### §3.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分支决策树

```python
def forward(self, x, smooth_scales=None, offsets=None, group_index=None,
            activate_left=False, quant_mode=0, group_list_type=0, dst_type=None):
    prefix_shape = x.shape[:-1]
    prefix_dims = x.numel() // x.shape[-1]
    half_dim = x.shape[-1] // 2
    x_ptr = x.view(prefix_dims, 2 * half_dim)

    is_int4 = dst_type is not None and (str(dst_type) == 'int4' or dst_type == torch.quint4x2)
    int_scale = 7 if is_int4 else 127
    clip_min, clip_max = (-8, 7) if is_int4 else (-128, 127)

    has_smooth_scales = smooth_scales is not None
    has_offsets = offsets is not None and quant_mode == 0
    has_group_index = group_index is not None
    num_groups = group_index.shape[0] if has_group_index else 0

    # block_size_col 按 half_dim 选择
    block_size_col = select_block_size_col(is_int4, half_dim)

    if quant_mode == 1:
        out_scales = torch.empty(prefix_dims, dtype=torch.float32, device=x.device)
        grid = (min((prefix_dims + BLOCK_ROWS - 1) // BLOCK_ROWS, VEC_CORE_NUM),)
        swiglu_quant_dynamic_kernel[grid](...)
    else:
        out_scales = torch.zeros(prefix_shape, dtype=torch.float32, device=x.device)
        grid = (min(prefix_dims, VEC_CORE_NUM),)
        swiglu_quant_static_kernel[grid](...)

    reshape and return
```

#### L2.2 Static kernel 骨架

```python
pid = tl.program_id(0)
num_progs = tl.num_programs(0)
rows_per_prog = (prefix_dims + num_progs - 1) // num_progs
start_row = pid * rows_per_prog
end_row = tl.minimum(start_row + rows_per_prog, prefix_dims)

for row_idx in range(start_row, end_row):
    gid = 0
    in_bounds = True
    if has_group_index:
        gid = row_idx // group_size
        gid = tl.minimum(gid, num_groups - 1)
        in_bounds = row_idx < num_groups * group_size

    scale_scalar = tl.load(smooth_scales_ptr + gid) if has_smooth_scales else 0.0
    offset_scalar = tl.load(offsets_ptr + gid) if has_offsets else 0.0

    for col_start in range(0, half_dim, BLOCK_SIZE_COL):
        num_cols = tl.minimum(BLOCK_SIZE_COL, half_dim - col_start)
        col_arange = tl.arange(0, BLOCK_SIZE_COL)
        mask = col_arange < num_cols

        left_base = row_idx * x_stride_row + col_start
        right_base = left_base + half_dim
        left = tl.load(x_ptr + left_base + col_arange, mask=mask, other=0.0).to(tl.float32)
        right = tl.load(x_ptr + right_base + col_arange, mask=mask, other=0.0).to(tl.float32)

        sig_left = left / (1.0 + tl.exp(-left))
        sig_right = right / (1.0 + tl.exp(-right))
        swiglu = tl.where(activate_left, sig_left * right, sig_right * left)

        if has_smooth_scales:
            if smooth_scales_is_2d:
                scale_block = tl.load(smooth_scales_ptr + gid * half_dim + col_start + col_arange,
                                      mask=mask, other=1.0).to(tl.float32)
                swiglu = tl.where(in_bounds, swiglu * scale_block, swiglu)
            else:
                swiglu = tl.where(in_bounds, swiglu * scale_scalar, swiglu)

        if has_offsets:
            if offsets_is_2d:
                offset_block = tl.load(offsets_ptr + gid * half_dim + col_start + col_arange,
                                       mask=mask, other=0.0).to(tl.float32)
                swiglu = tl.where(in_bounds, swiglu + offset_block, swiglu)
            else:
                swiglu = tl.where(in_bounds, swiglu + offset_scalar, swiglu)

        q_val = tl.extra.ascend.libdevice.nearbyint(swiglu)
        q_val = q_val.to(tl.int32)
        q_val = tl.maximum(q_val, clip_min_val)
        q_val = tl.minimum(q_val, clip_max_val)

        if is_int4:
            # reshape + split + pack
            ...
        else:
            q_val = q_val.to(tl.int8)
            tl.store(out_ptr + row_idx * half_dim + col_start + col_arange,
                     q_val, mask=mask)
```

#### L2.3 Dynamic kernel 骨架

```python
pid = tl.program_id(0)
num_progs = tl.num_programs(0)
num_tiles = (prefix_dims + BLOCK_ROWS - 1) // BLOCK_ROWS
tiles_per_prog = (num_tiles + num_progs - 1) // num_progs
start_tile = pid * tiles_per_prog
end_tile = tl.minimum(start_tile + tiles_per_prog, num_tiles)

for tile_idx in range(start_tile, end_tile):
    start_row = tile_idx * BLOCK_ROWS
    end_row = tl.minimum(start_row + BLOCK_ROWS, prefix_dims)

    # Pass 1: per-row max_abs
    for row_idx in range(start_row, end_row):
        gid, in_bounds = compute_group(row_idx, ...)
        scale_scalar = load_scalar_smooth(gid) if has_smooth_scales else 0.0
        max_abs = 0.0
        for col_start in range(0, half_dim, BLOCK_SIZE_COL):
            swiglu = compute_row_swiglu(row_idx, col_start, activate_left,
                                        has_smooth_scales, smooth_scales_is_2d, ...)
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(swiglu), axis=0))
        tl.store(scales_ptr + row_idx, max_abs)

    # Vector scale computation for this tile
    row_arange = tl.arange(0, BLOCK_ROWS)
    row_mask = row_arange < (end_row - start_row)
    max_vec = tl.load(scales_ptr + start_row + row_arange, mask=row_mask, other=1.0).to(tl.float32)
    scale_vec = int_scale_val / tl.maximum(max_vec, 1e-10)
    tl.store(scales_ptr + start_row + row_arange, scale_vec, mask=row_mask)

    # Pass 2: quantize
    for row_idx in range(start_row, end_row):
        row_scale = tl.load(scales_ptr + row_idx)
        for col_start in range(0, half_dim, BLOCK_SIZE_COL):
            swiglu = compute_row_swiglu(row_idx, col_start, activate_left,
                                        has_smooth_scales, smooth_scales_is_2d, ...)
            q_val = tl.extra.ascend.libdevice.nearbyint(swiglu * row_scale)
            q_val = clip(q_val, clip_min_val, clip_max_val)
            store_int8_or_pack_int4(q_val, ...)
```

#### L2.4 Block Size 选择策略

static/dynamic 共用 `BLOCK_SIZE_COL`，按 `half_dim` 选择：

```python
if is_int4:
    if half_dim >= 2048:   block_size_col = 4096
    elif half_dim >= 1024: block_size_col = 2048
    elif half_dim >= 512:  block_size_col = 1024
    elif half_dim >= 256:  block_size_col = 512
    else:                   block_size_col = 256
else:
    if half_dim > 2048:    block_size_col = 4096
    elif half_dim > 1024:  block_size_col = 2048
    else:                   block_size_col = 1024
```

dynamic kernel 可固定 `BLOCK_ROWS=4` 并配合 autotune 尝试 `{2, 4, 6}`。

> Layer 3 关键 kernel 见 [quantization__swigluquant_kernels.md](quantization__swigluquant_kernels.md)。
