---
type: Guide
title: Interpolate bicubic_area 路径关键 kernel
description: Interpolate 算子 bicubic_area 路径的 Triton kernel 实现与易错点。
tags: [task.implementation, topic.data_movement.layout_transform, topic.implementation.masking, topic.compute.indexing, topic.tiling, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Interpolate bicubic_area 路径关键 kernel

> 主卡见 [tensor_transform__interpolate.md](tensor_transform__interpolate.md)，
> 类别约束见 [tensor_transform_constraints.md](../constraints/tensor_transform_constraints.md)。

#### L3.6 bicubic ac=False `bicubic_vec_kernel`

```python
@triton.jit
def bicubic_vec_kernel(
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
    A = -0.75

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

        in_y = tl.cast(src_y, tl.int32)
        in_y_f = tl.cast(in_y, tl.float32)
        in_y = tl.where(src_y < in_y_f, in_y - 1, in_y)    # L1.11 负数 floor 修正
        in_y_f = tl.cast(in_y, tl.float32)                 # 修正后重新取 float
        t_y = src_y - in_y_f

        yy0 = in_y - 1
        yy1 = in_y
        yy2 = in_y + 1
        yy3 = in_y + 2
        yy0 = tl.where(yy0 < 0, 0, yy0)
        yy0 = tl.where(yy0 >= H_in, H_in - 1, yy0)
        yy1 = tl.where(yy1 < 0, 0, yy1)
        yy1 = tl.where(yy1 >= H_in, H_in - 1, yy1)
        yy2 = tl.where(yy2 < 0, 0, yy2)
        yy2 = tl.where(yy2 >= H_in, H_in - 1, yy2)
        yy3 = tl.where(yy3 < 0, 0, yy3)
        yy3 = tl.where(yy3 >= H_in, H_in - 1, yy3)

        # 垂直方向 4 个三次权重，用 tl.where 分段（L1.8）
        tt0 = t_y + 1.0
        tt0_abs = tl.where(tt0 < 0.0, -tt0, tt0)
        wy0 = tl.where(tt0_abs < 1.0,
                       ((A + 2.0) * tt0_abs - (A + 3.0)) * tt0_abs * tt0_abs + 1.0,
                       tl.where(tt0_abs < 2.0,
                                ((A * tt0_abs - 5.0 * A) * tt0_abs + 8.0 * A) * tt0_abs - 4.0 * A,
                                0.0))
        tt1 = t_y
        tt1_abs = tl.where(tt1 < 0.0, -tt1, tt1)
        wy1 = tl.where(tt1_abs < 1.0,
                       ((A + 2.0) * tt1_abs - (A + 3.0)) * tt1_abs * tt1_abs + 1.0,
                       tl.where(tt1_abs < 2.0,
                                ((A * tt1_abs - 5.0 * A) * tt1_abs + 8.0 * A) * tt1_abs - 4.0 * A,
                                0.0))
        tt2 = 1.0 - t_y
        tt2_abs = tl.where(tt2 < 0.0, -tt2, tt2)
        wy2 = tl.where(tt2_abs < 1.0,
                       ((A + 2.0) * tt2_abs - (A + 3.0)) * tt2_abs * tt2_abs + 1.0,
                       tl.where(tt2_abs < 2.0,
                                ((A * tt2_abs - 5.0 * A) * tt2_abs + 8.0 * A) * tt2_abs - 4.0 * A,
                                0.0))
        tt3 = 2.0 - t_y
        tt3_abs = tl.where(tt3 < 0.0, -tt3, tt3)
        wy3 = tl.where(tt3_abs < 1.0,
                       ((A + 2.0) * tt3_abs - (A + 3.0)) * tt3_abs * tt3_abs + 1.0,
                       tl.where(tt3_abs < 2.0,
                                ((A * tt3_abs - 5.0 * A) * tt3_abs + 8.0 * A) * tt3_abs - 4.0 * A,
                                0.0))
        w_sum_y = wy0 + wy1 + wy2 + wy3

        input_base = (n * C + c) * H_in * W_in
        row0 = tl.load(input_ptr + input_base + yy0 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + yy1 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row2 = tl.load(input_ptr + input_base + yy2 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row3 = tl.load(input_ptr + input_base + yy3 * W_in + r_offs,
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

            in_x = tl.cast(src_x, tl.int32)
            in_x_f = tl.cast(in_x, tl.float32)
            in_x = tl.where(src_x < in_x_f, in_x - 1, in_x)
            in_x_f = tl.cast(in_x, tl.float32)
            t_x = src_x - in_x_f

            xx0 = in_x - 1
            xx1 = in_x
            xx2 = in_x + 1
            xx3 = in_x + 2
            xx0 = tl.where(xx0 < 0, 0, xx0)
            xx0 = tl.where(xx0 >= W_in, W_in - 1, xx0)
            xx1 = tl.where(xx1 < 0, 0, xx1)
            xx1 = tl.where(xx1 >= W_in, W_in - 1, xx1)
            xx2 = tl.where(xx2 < 0, 0, xx2)
            xx2 = tl.where(xx2 >= W_in, W_in - 1, xx2)
            xx3 = tl.where(xx3 < 0, 0, xx3)
            xx3 = tl.where(xx3 >= W_in, W_in - 1, xx3)

            tx0 = t_x + 1.0
            tx0_abs = tl.where(tx0 < 0.0, -tx0, tx0)
            wx0 = tl.where(tx0_abs < 1.0,
                           ((A + 2.0) * tx0_abs - (A + 3.0)) * tx0_abs * tx0_abs + 1.0,
                           tl.where(tx0_abs < 2.0,
                                    ((A * tx0_abs - 5.0 * A) * tx0_abs + 8.0 * A) * tx0_abs - 4.0 * A,
                                    0.0))
            tx1 = t_x
            tx1_abs = tl.where(tx1 < 0.0, -tx1, tx1)
            wx1 = tl.where(tx1_abs < 1.0,
                           ((A + 2.0) * tx1_abs - (A + 3.0)) * tx1_abs * tx1_abs + 1.0,
                           tl.where(tx1_abs < 2.0,
                                    ((A * tx1_abs - 5.0 * A) * tx1_abs + 8.0 * A) * tx1_abs - 4.0 * A,
                                    0.0))
            tx2 = 1.0 - t_x
            tx2_abs = tl.where(tx2 < 0.0, -tx2, tx2)
            wx2 = tl.where(tx2_abs < 1.0,
                           ((A + 2.0) * tx2_abs - (A + 3.0)) * tx2_abs * tx2_abs + 1.0,
                           tl.where(tx2_abs < 2.0,
                                    ((A * tx2_abs - 5.0 * A) * tx2_abs + 8.0 * A) * tx2_abs - 4.0 * A,
                                    0.0))
            tx3 = 2.0 - t_x
            tx3_abs = tl.where(tx3 < 0.0, -tx3, tx3)
            wx3 = tl.where(tx3_abs < 1.0,
                           ((A + 2.0) * tx3_abs - (A + 3.0)) * tx3_abs * tx3_abs + 1.0,
                           tl.where(tx3_abs < 2.0,
                                    ((A * tx3_abs - 5.0 * A) * tx3_abs + 8.0 * A) * tx3_abs - 4.0 * A,
                                    0.0))
            w_sum_x = wx0 + wx1 + wx2 + wx3

            p00 = tl.gather(row0, xx0, 0)
            p01 = tl.gather(row0, xx1, 0)
            p02 = tl.gather(row0, xx2, 0)
            p03 = tl.gather(row0, xx3, 0)
            r0 = wx0 * p00 + wx1 * p01 + wx2 * p02 + wx3 * p03
            p10 = tl.gather(row1, xx0, 0)
            p11 = tl.gather(row1, xx1, 0)
            p12 = tl.gather(row1, xx2, 0)
            p13 = tl.gather(row1, xx3, 0)
            r1 = wx0 * p10 + wx1 * p11 + wx2 * p12 + wx3 * p13
            p20 = tl.gather(row2, xx0, 0)
            p21 = tl.gather(row2, xx1, 0)
            p22 = tl.gather(row2, xx2, 0)
            p23 = tl.gather(row2, xx3, 0)
            r2 = wx0 * p20 + wx1 * p21 + wx2 * p22 + wx3 * p23
            p30 = tl.gather(row3, xx0, 0)
            p31 = tl.gather(row3, xx1, 0)
            p32 = tl.gather(row3, xx2, 0)
            p33 = tl.gather(row3, xx3, 0)
            r3 = wx0 * p30 + wx1 * p31 + wx2 * p32 + wx3 * p33

            val = wy0 * r0 + wy1 * r1 + wy2 * r2 + wy3 * r3
            w_sum_total = w_sum_x * w_sum_y
            val = tl.where(w_sum_total != 0.0, val / w_sum_total, val)

            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.7 bicubic ac=True `bicubic_ac_true_vec_kernel`

```python
@triton.jit
def bicubic_ac_true_vec_kernel(
    input_ptr, output_ptr,
    y_idx_m1_ptr, y_idx_0_ptr, y_idx_p1_ptr, y_idx_p2_ptr,
    y_w_m1_ptr, y_w_0_ptr, y_w_p1_ptr, y_w_p2_ptr,
    x_idx_m1_ptr, x_idx_0_ptr, x_idx_p1_ptr, x_idx_p2_ptr,
    x_w_m1_ptr, x_w_0_ptr, x_w_p1_ptr, x_w_p2_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
    MAX_IN_W: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = N * C * H_out
    num_programs = tl.num_programs(0)
    ZERO = tl.full((), 0.0, tl.float32)

    r_offs = tl.arange(0, MAX_IN_W)
    r_mask = r_offs < W_in

    for row in range(pid, total_rows, num_programs):
        n = row // (C * H_out)
        tmp = row - n * C * H_out
        c = tmp // H_out
        h = tmp - c * H_out

        yy0 = tl.load(y_idx_m1_ptr + h).to(tl.int32)
        yy1 = tl.load(y_idx_0_ptr + h).to(tl.int32)
        yy2 = tl.load(y_idx_p1_ptr + h).to(tl.int32)
        yy3 = tl.load(y_idx_p2_ptr + h).to(tl.int32)
        wy0 = tl.load(y_w_m1_ptr + h).to(tl.float32)
        wy1 = tl.load(y_w_0_ptr + h).to(tl.float32)
        wy2 = tl.load(y_w_p1_ptr + h).to(tl.float32)
        wy3 = tl.load(y_w_p2_ptr + h).to(tl.float32)

        input_base = (n * C + c) * H_in * W_in
        row0 = tl.load(input_ptr + input_base + yy0 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row1 = tl.load(input_ptr + input_base + yy1 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row2 = tl.load(input_ptr + input_base + yy2 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        row3 = tl.load(input_ptr + input_base + yy3 * W_in + r_offs,
                       mask=r_mask, other=0.0).to(tl.float32)
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            xx0 = tl.load(x_idx_m1_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            xx1 = tl.load(x_idx_0_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            xx2 = tl.load(x_idx_p1_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            xx3 = tl.load(x_idx_p2_ptr + w_idx, mask=mask, other=0).to(tl.int32)
            wx0 = tl.load(x_w_m1_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            wx1 = tl.load(x_w_0_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            wx2 = tl.load(x_w_p1_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            wx3 = tl.load(x_w_p2_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)

            p00 = tl.gather(row0, xx0, 0)
            p01 = tl.gather(row0, xx1, 0)
            p02 = tl.gather(row0, xx2, 0)
            p03 = tl.gather(row0, xx3, 0)
            p10 = tl.gather(row1, xx0, 0)
            p11 = tl.gather(row1, xx1, 0)
            p12 = tl.gather(row1, xx2, 0)
            p13 = tl.gather(row1, xx3, 0)
            p20 = tl.gather(row2, xx0, 0)
            p21 = tl.gather(row2, xx1, 0)
            p22 = tl.gather(row2, xx2, 0)
            p23 = tl.gather(row2, xx3, 0)
            p30 = tl.gather(row3, xx0, 0)
            p31 = tl.gather(row3, xx1, 0)
            p32 = tl.gather(row3, xx2, 0)
            p33 = tl.gather(row3, xx3, 0)

            # 16 项标量逐项累加，匹配 PyTorch C++ 顺序（L1.7）
            val = ZERO
            w_sum_total = ZERO
            val = val + wy0 * wx0 * p00
            w_sum_total = w_sum_total + wy0 * wx0
            val = val + wy0 * wx1 * p01
            w_sum_total = w_sum_total + wy0 * wx1
            val = val + wy0 * wx2 * p02
            w_sum_total = w_sum_total + wy0 * wx2
            val = val + wy0 * wx3 * p03
            w_sum_total = w_sum_total + wy0 * wx3
            val = val + wy1 * wx0 * p10
            w_sum_total = w_sum_total + wy1 * wx0
            val = val + wy1 * wx1 * p11
            w_sum_total = w_sum_total + wy1 * wx1
            val = val + wy1 * wx2 * p12
            w_sum_total = w_sum_total + wy1 * wx2
            val = val + wy1 * wx3 * p13
            w_sum_total = w_sum_total + wy1 * wx3
            val = val + wy2 * wx0 * p20
            w_sum_total = w_sum_total + wy2 * wx0
            val = val + wy2 * wx1 * p21
            w_sum_total = w_sum_total + wy2 * wx1
            val = val + wy2 * wx2 * p22
            w_sum_total = w_sum_total + wy2 * wx2
            val = val + wy2 * wx3 * p23
            w_sum_total = w_sum_total + wy2 * wx3
            val = val + wy3 * wx0 * p30
            w_sum_total = w_sum_total + wy3 * wx0
            val = val + wy3 * wx1 * p31
            w_sum_total = w_sum_total + wy3 * wx1
            val = val + wy3 * wx2 * p32
            w_sum_total = w_sum_total + wy3 * wx2
            val = val + wy3 * wx3 * p33
            w_sum_total = w_sum_total + wy3 * wx3

            val = tl.where(w_sum_total != 0.0, val / w_sum_total, val)
            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

#### L3.9 area 通用路径 `area_vec_kernel`

```python
@triton.jit
def area_vec_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_W: tl.constexpr,
    MAX_KW: tl.constexpr,
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

        istartH = (h * H_in) // H_out
        iendH = ((h + 1) * H_in + H_out - 1) // H_out
        kH = iendH - istartH

        input_base = (n * C + c) * H_in * W_in
        output_row_off = ((n * C + c) * H_out + h) * W_out

        for w_start in range(0, W_out, BLOCK_W):
            w_offs = tl.arange(0, BLOCK_W)
            w_idx = w_start + w_offs
            mask = w_idx < W_out

            istartW = (w_idx * W_in) // W_out
            iendW = ((w_idx + 1) * W_in + W_out - 1) // W_out

            sum_val = tl.full([BLOCK_W], 0.0, tl.float32)
            for ih in range(istartH, iendH):
                row_off = input_base + ih * W_in
                in_row = tl.load(input_ptr + row_off + r_offs,
                                 mask=r_mask, other=0.0).to(tl.float32)
                for iw_offset in range(MAX_KW):
                    iw = istartW + iw_offset
                    valid = (iw < iendW) & mask
                    pixel = tl.gather(in_row, iw, 0)
                    sum_val = tl.where(valid, sum_val + pixel, sum_val)

            kW = iendW - istartW
            kH_f = tl.cast(kH, tl.float32)
            kW_f = tl.cast(kW, tl.float32)
            area = kH_f * kW_f
            val = tl.where(area > 0.0, sum_val / area, sum_val)

            tl.store(output_ptr + output_row_off + w_idx, val, mask=mask)
```

### §2.4 Interpolate 性能基准

| Mode | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| area (downsample) | 5 | 18x ~ 83x | 极快，部分超 profiler 分辨率 |
| bilinear (downsample) | ~15 | 1.5x ~ 6x | 表现优异 |
| bilinear (upsample, ac=F) | ~10 | 0.28x ~ 1.8x | 中等表现 |
| bilinear (upsample, ac=T) | ~8 | 0.18x ~ 0.6x | 较弱，gather 开销 |
| bicubic (ac=F) | 4 | 0.16x ~ 3.4x | 高方差 |
| bicubic (ac=T) | 3 | 0.07x ~ 1.9x | 高方差 |
| nearest | 8 | 0.29x ~ 6.4x | 中等表现 |
| 全量 73 cases | 73 | 1.3834x | 达标 0.8x 项目目标 |

**关键结论**:
1. 下采样路径是性能优势区（1.5x~83x），输出少、输入行复用充分
2. 上采样路径较弱（0.18x~1.8x），输出多、每行独立计算
3. 多策略分派是核心：不同 mode 和采样比例的最优数据访问模式完全不同
4. 边界 floor 修正对 bicubic ac=False 精度至关重要

---
