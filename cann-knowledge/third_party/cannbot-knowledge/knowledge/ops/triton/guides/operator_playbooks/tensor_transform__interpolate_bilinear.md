---
type: Guide
title: Interpolate bilinear 路径关键 kernel
description: Interpolate 算子 bilinear 路径的 Triton kernel 实现与易错点。
tags: [task.implementation, topic.data_movement.layout_transform, topic.implementation.masking, topic.compute.indexing, topic.tiling, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Interpolate bilinear 路径关键 kernel

> 主卡见 [tensor_transform__interpolate.md](tensor_transform__interpolate.md)，
> 类别约束见 [tensor_transform_constraints.md](../constraints/tensor_transform_constraints.md)。

#### L3.3 bilinear 0.5x 下采样退化为 avg pool `bilinear_half_pool_kernel`

```python
@triton.jit
def bilinear_half_pool_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    rows_per_prog = (total_rows + num_programs - 1) // num_programs
    row_start = pid * rows_per_prog
    row_end = row_start + rows_per_prog
    if row_end > total_rows:
        row_end = total_rows

    QUARTER = 0.25

    for row in range(row_start, row_end):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        y0 = 2 * h
        y1 = y0 + 1
        if y1 >= H_in:
            y1 = H_in - 1

        input_base = (n * C + c) * H_in * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            # 一次 contiguous load 2*BLOCK_W 元素，禁止拆成 4 次 strided load
            in_start = 2 * w_start
            in_offs = tl.arange(0, 2 * BLOCK_W)
            in_idx = in_start + in_offs
            in_mask = in_idx < W_in

            row0 = tl.load(input_ptr + input_base + y0 * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)
            row1 = tl.load(input_ptr + input_base + y1 * W_in + in_idx,
                           mask=in_mask, other=0.0).to(tl.float32)

            row0_pairs = tl.reshape(row0, [BLOCK_W, 2])
            row1_pairs = tl.reshape(row1, [BLOCK_W, 2])
            row0_sum = tl.sum(row0_pairs, axis=1)
            row1_sum = tl.sum(row1_pairs, axis=1)

            val = (row0_sum + row1_sum) * QUARTER
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.4 通用 bilinear `bilinear_general_vec_kernel`

```python
@triton.jit
def bilinear_general_vec_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        h_f = tl.cast(h, tl.float32)
        sh = tl.cast(scale_h, tl.float32)
        if align_corners == 1:
            src_y = sh * h_f
        else:
            half = tl.full((), 0.5, tl.float32)
            src_y = sh * (h_f + half) - half

        y0 = tl.cast(src_y, tl.int32)
        y0_f = tl.cast(y0, tl.float32)
        if align_corners != 1:
            y0 = tl.where(src_y < y0_f, y0 - 1, y0)    # L1.11 负坐标 floor 修正
            y0_f = tl.cast(y0, tl.float32)
        y1 = y0 + 1
        y0 = tl.where(y0 < 0, 0, y0)
        y1 = tl.where(y1 >= H_in, H_in - 1, y1)

        ylambda = src_y - y0_f
        ylambda = tl.where(ylambda < 0.0, 0.0, ylambda)
        ylambda = tl.where(ylambda > 1.0, 1.0, ylambda)
        y0lambda = 1.0 - ylambda

        input_base = (n * C + c) * H_in * W_in
        row0 = tl.load(input_ptr + input_base + y0 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + y1 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            w_f = tl.cast(w_idx, tl.float32)
            sw = tl.cast(scale_w, tl.float32)
            if align_corners == 1:
                src_x = sw * w_f
            else:
                half = tl.full((), 0.5, tl.float32)
                src_x = sw * (w_f + half) - half

            x0 = tl.cast(src_x, tl.int32)
            x0_f = tl.cast(x0, tl.float32)
            if align_corners != 1:
                x0 = tl.where(src_x < x0_f, x0 - 1, x0)
                x0_f = tl.cast(x0, tl.float32)
            x1 = x0 + 1
            x0 = tl.where(x0 < 0, 0, x0)
            x1 = tl.where(x1 >= W_in, W_in - 1, x1)

            xlambda = src_x - x0_f
            xlambda = tl.where(xlambda < 0.0, 0.0, xlambda)
            xlambda = tl.where(xlambda > 1.0, 1.0, xlambda)
            x0lambda = 1.0 - xlambda

            i00 = tl.gather(row0, x0, 0)
            i01 = tl.gather(row0, x1, 0)
            i10 = tl.gather(row1, x0, 0)
            i11 = tl.gather(row1, x1, 0)

            top = x0lambda * i00 + xlambda * i01
            bot = x0lambda * i10 + xlambda * i11
            val = y0lambda * top + ylambda * bot

            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.4b bilinear 下采样/general 预计算 x 坐标版 `bilinear_general_precomputed_x_kernel`（L1.14 强制）

L3.4 的 `bilinear_general_vec_kernel` 在 kernel 内重算 x 坐标，性能差。**必须**用本节版本：x0/x1/xlambda 由 host 预计算传入。

```python
@triton.jit
def bilinear_general_precomputed_x_kernel(
    input_ptr, output_ptr,
    x0_ptr, x1_ptr, xlambda_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        # y 坐标仍在 kernel 内算（每行不同）
        h_f = tl.cast(h, tl.float32)
        sh = tl.cast(scale_h, tl.float32)
        if align_corners == 1:
            src_y = sh * h_f
        else:
            half = tl.full((), 0.5, tl.float32)
            src_y = sh * (h_f + half) - half
        y0 = tl.cast(src_y, tl.int32)
        y0_f = tl.cast(y0, tl.float32)
        if align_corners != 1:
            y0 = tl.where(src_y < y0_f, y0 - 1, y0)
            y0_f = tl.cast(y0, tl.float32)
        y1 = y0 + 1
        y0 = tl.where(y0 < 0, 0, y0)
        y1 = tl.where(y1 >= H_in, H_in - 1, y1)
        ylambda = src_y - y0_f
        ylambda = tl.where(ylambda < 0.0, 0.0, ylambda)
        ylambda = tl.where(ylambda > 1.0, 1.0, ylambda)
        y0lambda = 1.0 - ylambda

        input_base = (n * C + c) * H_in * W_in
        row0 = tl.load(input_ptr + input_base + y0 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + y1 * W_in + r_offs, mask=r_mask, other=0.0).to(tl.float32)
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out
            # x 坐标从 host 预计算表 load（L1.14）
            x0 = tl.load(x0_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            x1 = tl.load(x1_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            xlambda = tl.load(xlambda_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            x0lambda = 1.0 - xlambda

            i00 = tl.gather(row0, x0, 0)
            i01 = tl.gather(row0, x1, 0)
            i10 = tl.gather(row1, x0, 0)
            i11 = tl.gather(row1, x1, 0)
            top = x0lambda * i00 + xlambda * i01
            bot = x0lambda * i10 + xlambda * i11
            val = y0lambda * top + ylambda * bot
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

> host 分派：bilinear 的 general/下采样分支（非 2x 上采样、非 0.5x 下采样）调用本 kernel，先 `x0,x1,xl = _precompute_bilinear_x_coords(W_in, W_out, align_corners, scale_w)` 再传入。

#### L3.5 上采样 2D 垂直分块 + host 预计算 x 坐标 `bilinear_2d_tiled_precomputed_x_kernel`

**关键**：x 坐标（x0/x1/xlambda）对所有输出行相同，**必须 host 预计算**为三张量表传入 kernel，禁止 kernel 内每个 w tile 重算。kernel 内重算 x 坐标的旧版本（vec_kernel）会导致上采样 shape 极慢。

```python
@triton.jit
def bilinear_2d_tiled_precomputed_x_kernel(
    input_ptr, output_ptr,
    x0_ptr, x1_ptr, xlambda_ptr,
    N, C, H_in, W_in, H_out, W_out,
    align_corners: tl.constexpr,
    scale_h, scale_w,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MAX_KH: tl.constexpr,
):
    pid = tl.program_id(0)
    num_h_tiles = (H_out + BLOCK_H - 1) // BLOCK_H
    total_tiles = N * C * num_h_tiles
    num_programs = tl.num_programs(0)

    r_offs = tl.arange(0, MAX_IN_W)
    kh_offs = tl.arange(0, MAX_KH)

    for tile in range(pid, total_tiles, num_programs):
        n = tile // (C * num_h_tiles)
        tmp = tile - n * C * num_h_tiles
        c = tmp // num_h_tiles
        h_tile = tmp - c * num_h_tiles
        h_start = h_tile * BLOCK_H

        input_base = (n * C + c) * H_in * W_in
        output_base = ((n * C + c) * H_out + h_start) * W_out

        h_f = tl.cast(h_start, tl.float32)
        sh = tl.cast(scale_h, tl.float32)
        if align_corners == 1:
            src_y = sh * h_f
        else:
            half = tl.full((), 0.5, tl.float32)
            src_y = sh * (h_f + half) - half
        y_min = tl.cast(src_y, tl.int32)
        y_min_f = tl.cast(y_min, tl.float32)
        if align_corners != 1:
            y_min = tl.where(src_y < y_min_f, y_min - 1, y_min)

        # 一次性 load MAX_KH 行输入到 flat UB buffer
        y_idx = y_min + kh_offs
        y_idx = tl.where(y_idx < 0, 0, y_idx)
        y_idx = tl.where(y_idx >= H_in, H_in - 1, y_idx)
        rr = y_idx[:, None]
        cc = r_offs[None, :]
        block2d = tl.load(input_ptr + input_base + rr * W_in + cc,
                          mask=(cc < W_in), other=0.0).to(tl.float32)
        flat_buf = tl.reshape(block2d, [MAX_KH * MAX_IN_W])

        for dh in range(BLOCK_H):
            h = h_start + dh
            h_valid = h < H_out

            h_f = tl.cast(h, tl.float32)
            if align_corners == 1:
                src_y = sh * h_f
            else:
                src_y = sh * (h_f + half) - half
            y0 = tl.cast(src_y, tl.int32)
            y0_f = tl.cast(y0, tl.float32)
            if align_corners != 1:
                y0 = tl.where(src_y < y0_f, y0 - 1, y0)
            y1 = y0 + 1
            y0 = tl.where(y0 < 0, 0, y0)
            y1 = tl.where(y1 >= H_in, H_in - 1, y1)
            ylambda = src_y - tl.cast(y0, tl.float32)
            ylambda = tl.where(ylambda < 0.0, 0.0, ylambda)
            ylambda = tl.where(ylambda > 1.0, 1.0, ylambda)
            y0lambda = 1.0 - ylambda

            ry0 = y0 - y_min
            ry1 = y1 - y_min

            for w_start in range(0, W_out, BLOCK_W):
                w_offs = tl.arange(0, BLOCK_W)
                w_idx = w_start + w_offs
                mask = (w_idx < W_out) & h_valid

                # x 坐标从 host 预计算的张量 load，禁止 kernel 内重算
                x0 = tl.load(x0_ptr + w_idx, mask=mask, other=0).to(tl.int32)
                x1 = tl.load(x1_ptr + w_idx, mask=mask, other=0).to(tl.int32)
                xlambda = tl.load(xlambda_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
                x0lambda = 1.0 - xlambda

                fx0 = ry0 * MAX_IN_W + x0
                fx1 = ry0 * MAX_IN_W + x1
                fx2 = ry1 * MAX_IN_W + x0
                fx3 = ry1 * MAX_IN_W + x1

                i00 = tl.gather(flat_buf, fx0, 0)
                i01 = tl.gather(flat_buf, fx1, 0)
                i10 = tl.gather(flat_buf, fx2, 0)
                i11 = tl.gather(flat_buf, fx3, 0)

                top = x0lambda * i00 + xlambda * i01
                bot = x0lambda * i10 + xlambda * i11
                val = y0lambda * top + ylambda * bot

                tl.store(output_ptr + output_base + dh * W_out + w_idx, val, mask=mask)
```

**Opt10：x 表 load 复用（2D tiled kernel）**：上面 kernel 中 `x0/x1/xlambda` 的 `tl.load` 在 `for dh` 内、`for w_start` 内，每个 dh 重复 load 同一份 x 表。若 `BLOCK_H=2`，x 表被 load 2 次。优化：把 x 表 load 提到 `for dh` 循环外、在 `for w_start` 同级预先 load 整个 W_out 的 x0/x1/xlambda 到 UB buffer（一次性 `tl.load(x0_ptr + tl.arange(0, W_out_POW2))`），dh 循环内按 w_start 切片复用。对 BLOCK_H 较大或 W_out 较大的上采样 shape 提升明显（归档 Phase4 实测救回多个 bilinear ac=True 上采样 shape）。

**host 预计算函数**（必须配套使用）：
```python
def _precompute_bilinear_x_coords(W_in, W_out, align_corners, scale_w):
    import math
    x0 = torch.empty(W_out, dtype=torch.int32)
    x1 = torch.empty(W_out, dtype=torch.int32)
    xlambda = torch.empty(W_out, dtype=torch.float32)
    for i in range(W_out):
        if align_corners:
            src_x = scale_w * i
        else:
            src_x = scale_w * (i + 0.5) - 0.5
        x0_i = int(math.floor(src_x))
        x0_i = max(0, x0_i)
        x0_i = x0_i if x0_i < W_in - 1 else W_in - 1
        x1_i = x0_i + 1
        x1_i = x1_i if x1_i < W_in else W_in - 1
        xl = float(src_x - x0_i)
        xl = 0.0 if xl < 0.0 else xl
        xl = 1.0 if xl > 1.0 else xl
        xlambda[i] = xl
        x0[i] = x0_i
        x1[i] = x1_i
    return x0, x1, xlambda
```
