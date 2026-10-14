---
type: Guide
title: SwigluQuant 关键 kernel 实现
description: SwigluQuant 的 static/dynamic 双 kernel 实现、INT4 打包与 scale 向量除法路径。
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
# SwigluQuant 关键 kernel 实现

> 主卡见 [quantization__swigluquant.md](quantization__swigluquant.md)。

### §3.3 Layer 3: 关键技巧（Agent 可参考但不可复制）

#### L3.1 手写 sigmoid 保证精度

**错误**:
```python
sig_left = tl.sigmoid(left)
```

**正确**:
```python
sig_left = left / (1.0 + tl.exp(-left))
```

**可替代方向**: 可尝试 `tl.extra.ascend.libdevice` 中其他 exp 近似，但需逐 case 验证精度。

#### L3.2 标量 smooth/offset 的冗余边界合并

对标量** smooth/offset 路径，可将边界判断合并到 `tl.load`：

```python
# 原实现
scale_scalar = tl.load(smooth_scales_ptr + gid)
swiglu = tl.where(in_bounds, swiglu * scale_scalar, swiglu)

# 优化后
scale_scalar = tl.load(smooth_scales_ptr + gid, mask=in_bounds, other=1.0)
swiglu = swiglu * scale_scalar
```

**收益**: 实现延时几何平均下降约 2.6%。

**注意**: 对 **2D smooth/offset** 做同样合并会触发 UB overflow，应保持 `tl.where(in_bounds, swiglu * scale_block, swiglu)`。

#### L3.3 dynamic 模式 vector scale division

**错误（标量路径，ULP 漂移）**:
```python
max_abs = tl.max(tl.abs(swiglu), axis=0)
row_scale = int_scale / tl.maximum(max_abs, 1e-10)
q = nearbyint(swiglu * row_scale)
```

**正确（向量路径，精度对齐 torch）**:
```python
# Pass 1: 写 max_abs
max_abs = tl.max(tl.abs(swiglu), axis=0)
tl.store(scales_ptr + row_idx, max_abs)

# tile 级别重新 load 向量
max_vec = tl.load(scales_ptr + start_row + row_arange, mask=row_mask, other=1.0)
scale_vec = int_scale / tl.maximum(max_vec, 1e-10)
tl.store(scales_ptr + start_row + row_arange, scale_vec, mask=row_mask)

# Pass 2: 读 row_scale
row_scale = tl.load(scales_ptr + row_idx)
q = nearbyint(swiglu * row_scale)
```

**可替代方向**: 可尝试 `BLOCK_ROWS={2,4,6}` autotune，但历史测试显示收益有限。

#### L3.4 INT4 打包模板

```python
q_val = tl.extra.ascend.libdevice.nearbyint(swiglu).to(tl.int32)
q_val = tl.maximum(q_val, clip_min_val)
q_val = tl.minimum(q_val, clip_max_val)

q_2d = tl.reshape(q_val, (BLOCK_SIZE_COL // 2, 2))
q_even, q_odd = tl.split(q_2d)

q_even_4 = q_even & 0x0F
q_odd_4 = q_odd & 0x0F
packed = ((q_odd_4 << 4) | q_even_4).to(tl.int8)

pair_arange = tl.arange(0, BLOCK_SIZE_COL // 2)
pair_mask = pair_arange < (num_cols // 2)
out_base = row_idx * (half_dim // 2) + (col_start // 2)
tl.store(out_ptr + out_base + pair_arange, packed, mask=pair_mask)
```

**可替代方向**: 可尝试 interleave 而非 split，但当前实现已验证正确。

#### L3.5 编译选项

- `multibuffer=True` 在该算子上有稳定收益，保持开启
- `unit_flag=True` 历史上测得性能下降，应避免

#### L3.6 用 `tl.load(..., mask=..., other=...)` 减少 `tl.where`

```python
# 推荐
scale_block = tl.load(ptr + off, mask=mask, other=1.0)

# 不推荐
scale_block = tl.load(ptr + off, mask=mask)
scale_block = tl.where(mask, scale_block, 1.0)
```

### §3.4 SwigluQuant 优化尝试记录与性能基准

#### 有效优化点

| 优化点 | 效果 | 关键代码位置 |
|--------|------|-------------|
| 标量 smooth/offset 边界合并到 `tl.load` | 延时下降约 2.6% | static/dynamic kernel 标量分支 |
| vector scale division | 大 fp32 dynamic shape 精度对齐 | dynamic kernel tile 级别 scale 计算 |
| `multibuffer=True` | 稳定收益 | `_route` 中 kernel 启动参数 |
| 手写 sigmoid | 避免系统性精度偏差 | sigmoid 计算处 |

#### 尝试后无效或劣化的优化点（负面经验/失败方向）

| 优化点 | 结果 | 原因 |
|--------|------|------|
| recompute 核 autotune `BLOCK_ROWS={2,4,6}` | 无提升 | `BLOCK_ROWS=4` 已接近最优 |
| 把 int32 `max/min` clip 改为 fp32 clip | 失败 | 增大 fp32 中间结果 live range，case 27 UB overflow |
| 把 recompute `BLOCK_ROWS` 从 4 降到 2 | 劣化 | 0.2434 ms > 0.2382 ms |
| 静态 `block_size_col` 限制到 2048 | 异常 | benchmark framework 延时异常，结果不可信 |

#### 参考性能数据

| 指标 | 数值 |
|------|------|
| Implementation Avg Latency | 0.2654 ms |
| Framework Avg Latency | 0.1239 ms |
| 几何平均加速比 | 0.5067x |
| 相对 Phase 3 基线 | 0.8990x |

**关键结论**:
1. **static/dynamic 分 kernel 是架构基础**：合并会导致控制流复杂、向量利用率低
2. **手写 sigmoid 是精度前提**：`tl.sigmoid` 在该后端有系统性偏差
3. **vector scale division 是 dynamic 模式精度关键**：标量除法会在大 fp32 shape 上 off-by-1
4. **INT4 打包必须严格按 (odd << 4) \| even 布局**：与 `torch.quint4x2` 一致
5. **UB 压力是主要瓶颈**：动态核 `BLOCK_ROWS` 受限；避免长期保留 fp32 中间结果
6. **speedup_vs_torch < 1 不代表失败**：framework 是优化后的 NPU 原生算子，重点看 implementation latency

---
