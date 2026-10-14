---
type: Guide
title: Interpolate nearest 路径关键 kernel
description: Interpolate 算子 nearest 路径的 Triton kernel 实现与易错点。
tags: [task.implementation, topic.data_movement.layout_transform, topic.implementation.masking, topic.compute.indexing, topic.tiling, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Interpolate nearest 路径关键 kernel

> 主卡见 [tensor_transform__interpolate.md](tensor_transform__interpolate.md)，
> 类别约束见 [tensor_transform_constraints.md](../constraints/tensor_transform_constraints.md)。

#### L3.1 nearest 2x 上采样 `nearest_2x_upsample_kernel`

```python
@triton.jit
def nearest_2x_upsample_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    BLOCK_IN: tl.constexpr = BLOCK_W // 2

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        src_h = h // 2
        input_base = (n * C + c) * H_in * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            in_w_start = w_start // 2
            in_offs = tl.arange(0, BLOCK_IN)
            in_idx = in_w_start + in_offs
            in_mask = in_idx < W_in

            vals = tl.load(input_ptr + input_base + src_h * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)

            # [a, b, c] -> [a, a, b, b, c, c]
            vals = tl.reshape(vals, [BLOCK_IN, 1])
            ones = tl.full([1, 2], 1.0, tl.float32)
            vals_dup = vals * ones
            out_vals = tl.reshape(vals_dup, [BLOCK_W])

            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out
            tl.store(output_ptr + output_row_off + w_idx, out_vals, mask=mask)
```

#### L3.2 nearest 0.5x 下采样 `nearest_half_downsample_kernel`

```python
@triton.jit
def nearest_half_downsample_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        src_h = 2 * h
        input_base = (n * C + c) * H_in * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            in_w_start = 2 * w_start
            in_offs = tl.arange(0, BLOCK_W)
            in_idx = in_w_start + 2 * in_offs   # 步长 2，但仍为 contiguous vector
            in_mask = in_idx < W_in

            vals = tl.load(input_ptr + input_base + src_h * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)

            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out
            tl.store(output_ptr + output_row_off + w_idx, vals, mask=mask)
```

#### L3.8 nearest 通用路径 `nearest_vec_kernel`

```python
@triton.jit
def nearest_vec_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    use_size: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    rows_per_prog = (total_rows + num_programs - 1) // num_programs
    row_start = pid * rows_per_prog
    row_end = row_start + rows_per_prog
    if row_end > total_rows:
        row_end = total_rows

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    last_n = -1
    last_c = -1
    last_src_yi = -1
    in_row = tl.zeros((MAX_IN_W,), dtype=tl.float32)

    for row in range(row_start, row_end):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        if use_size == 1:
            src_yi = (h * H_in) // H_out
        else:
            src_y = tl.cast(scale_h, tl.float32) * tl.cast(h, tl.float32)
            src_yi = tl.cast(src_y, tl.int32)
        if src_yi < 0:
            src_yi = 0
        if src_yi >= H_in:
            src_yi = H_in - 1

        input_row_off = (n * C + c) * H_in * W_in + src_yi * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        # 跨行行复用：相邻输出行可能映射到同一输入行
        if (n != last_n) | (c != last_c) | (src_yi != last_src_yi):
            in_row = tl.load(input_ptr + input_row_off + r_offs,
                             mask=r_mask, other=0.0).to(tl.float32)
            last_n = n
            last_c = c
            last_src_yi = src_yi

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            if use_size == 1:
                src_xi = (w_idx * W_in) // W_out
            else:
                w_f = tl.cast(w_idx, tl.float32)
                src_x = tl.cast(scale_w, tl.float32) * w_f
                src_xi = tl.cast(src_x, tl.int32)
            src_xi = tl.where(src_xi < 0, 0, src_xi)
            src_xi = tl.where(src_xi >= W_in, W_in - 1, src_xi)

            val = tl.gather(in_row, src_xi, 0)
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```
