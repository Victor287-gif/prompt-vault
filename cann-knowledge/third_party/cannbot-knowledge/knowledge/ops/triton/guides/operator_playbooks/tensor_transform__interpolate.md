---
type: Guide
title: Interpolate 算子（transformation-compute） 实现手册
description: '算子类别: transformationcompute（数据重排伴随权重计算。'
tags: [task.implementation, topic.data_movement.layout_transform, topic.implementation.masking, topic.compute.indexing, topic.tiling, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Interpolate 算子（transformation-compute） 实现手册

> 类别约束见 [tensor_transform_constraints.md](../constraints/tensor_transform_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 Interpolate 算子（transformation-compute）

**算子类别**: `transformation-compute`（数据重排伴随权重计算；nearest/bilinear/bicubic/area 重采样）
**典型特征**: 4D NCHW 张量，支持 `nearest` / `bilinear` / `bicubic` / `area` 四种 mode，`align_corners` True/False，`size`/`scale_factor` 指定输出尺寸，fp32/fp16/bf16
**性能基准**: 73/73 pass，几何平均加速比 **1.3834x** vs torch（超 0.8x 项目目标、0.6x 用户目标）

### §2.0 首次生成必读：为什么必须把主要框架写对

Interpolate 是一个**计算特征高度异构**的算子：四种 mode × align_corners × 上/下采样 × 整数倍/分数倍比例，最优数据访问模式完全不同。**首次生成如果把框架写偏（例如用单一 kernel 统所有路径、漏掉 0.5x 下采样特化、用 gather 做整数倍上采样），后续迭代很难通过局部修bug把性能救回来**——历史上出现过迭代 58 轮仍无法精度/性能双达标的情况。

本章按 Layer 1→3 组织：
- **L1 是硬性约束**，首次生成必须全部满足；
- **L2 是 host 分派骨架 + 辅助函数**，必须一次写对；
- **L3 是关键 kernel 实现**，贴出优化的重点代码和那些不容易首次生成就生成出来的代码。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须按 (mode, align_corners, 采样比例) 多策略分派
- **必须**在 host 侧 `_route` 方法中按 `(mode, align_corners, H_out vs H_in, W_out vs W_in)` 分派到特化 kernel。
- **禁止**用单一 kernel 统一处理所有 mode 和采样比例。
- **Why:** 不同场景的最优数据访问模式完全不同（2x 上采样用 broadcast，0.5x 下采样用 strided/avgpool，通用用 gather），强行统一会丢失所有特化机会。
- **How to apply:** 见 L2.1 分派决策树。

#### L1.2 行预载必须用 pow2 MAX_IN 作为 constexpr
- **必须**用 `MAX_IN_W = next_power_of_2(W_in)`（或受 cap 限制）把输入宽度 pad 到 2 的幂，作为 constexpr 传入。
- **Why:** Ascend 上 fixed-shape vector load 必须是编译期常量长度；若 `tl.arange(0, W_in)` 中 W_in 非 constexpr 会触发 dynamic-shape load，退化为标量循环。
- **How to apply:** 见 L2.2 `_select_max_in_w` 与 L3 kernel 中的 `tl.arange(0, MAX_IN_W)`。

#### L1.3 2x 上采样必须用 reshape broadcast 复制像素
- **必须**在 `H_out == 2*H_in 且 W_out == 2*W_in` 时用 `nearest_2x_upsample_kernel`，通过 `vals.reshape([BLOCK_IN,1]) * ones([1,2])` 把 `[a,b,c]` 复制成 `[a,a,b,b,c,c]`。
- **禁止**用 gather 实现整数倍上采样。
- **Why:** 整数倍上采样的每个输入像素映射到固定的输出块，reshape broadcast 是 contiguous 计算，远快于 gather 的随机访问。
- **How to apply:** 见 L3.1 `nearest_2x_upsample_kernel`。

#### L1.4 0.5x 下采样必须用 strided contiguous load
- **必须**在 `H_in == 2*H_out 且 W_in == 2*W_out` 时用 strided load（步长 2），禁止用 gather。
- **Why:** 0.5x 下采样的输出像素 (h,w) 映射到输入 (2h, 2w)，是规则的步长访问，contiguous strided load 比 gather 快。
- **How to apply:** 见 L3.2 `nearest_half_downsample_kernel`。

#### L1.5 bilinear 0.5x 下采样 (ac=False) 必须退化为 2x2 avg pool
- **必须**在 `mode='bilinear' and not align_corners and H_in==2*H_out and W_in==2*W_out` 时用 `bilinear_half_pool_kernel`，把 bilinear 退化为 2x2 均值池化。
- **Why:** ac=False + 0.5x 下采样时，bilinear 的 4 个采样点正好是 2x2 输入块，权重全是 0.25，等价于均值池化。用 `reshape([BLOCK_W,2]) + tl.sum` 实现远快于通用 bilinear 的 gather。
- **⚠️ 关键细节**: kernel 内部必须**一次 contiguous load `2*BLOCK_W` 元素**，再 `reshape([BLOCK_W,2]) + tl.sum`。禁止 `x0=w_idx*2; x1=x0+1` 做 4 次离散 strided load——strided load 无法 coalesce，实测性能差 100 倍，单点决定 geomean 能否过 0.6x。
- **How to apply:** 见 L3.3 `bilinear_half_pool_kernel`。

#### L1.6 上采样必须用 2D 垂直分块复用输入行 + host 预计算 x 坐标
- **必须**在 `H_out > H_in`（上采样）时用 `bilinear_2d_tiled_precomputed_x_kernel`，BLOCK_H=2、MAX_KH=3，一次性 load 3 行输入供 2 行输出复用。
- **必须** host 侧用 `_precompute_bilinear_x_coords(W_in, W_out, align_corners, scale_w)` 预计算 x0/x1/xlambda 三张量表，传入 kernel。**禁止**在 kernel 内每个 w tile 重算 x 坐标（cast/floor/where 链）。
- **Why:** 上采样时多个输出行共享同一批输入行，2D 垂直分块复用输入行减少 HBM 读取；x 坐标对所有行相同，host 预计算一次比 kernel 内每个 tile 重算快得多。历史上 kernel 内重算 x 坐标的版本（vec_kernel）比 precomputed_x 版本慢，导致部分上采样 shape 极慢（<0.1x），几何平均从 0.5x 跌到 0.29x。
- **How to apply:** 见 L3.5 `bilinear_2d_tiled_precomputed_x_kernel` 与 L2.2 `_precompute_bilinear_x_coords`。

#### L1.7 align_corners=True 必须在 host 预计算坐标 + 标量累积
- **必须**ac=True 路径在 host 用 `numpy.float32` 预算坐标和权重，传入 kernel 作为指针。
- **必须**bicubic ac=True 用**标量逐项累加**（16 项 `val = val + w*pixel`）匹配 PyTorch C++ 标量计算顺序。
- **Why:** PyTorch C++ 的 ac=True 坐标用 float32 计算 (`scale = (H_in-1)/(H_out-1)`)，Triton kernel 内若重新计算会有精度差异导致 verify 失败。bicubic 的 16 项累加若向量化会改变累加顺序，与 PyTorch 标量顺序不一致。
- **How to apply:** 见 L2.2 `_precompute_bicubic_ac_true_all` 与 L3.6 `bicubic_ac_true_vec_kernel`。

#### L1.8 三次权重必须用 tl.where 向量化分段，禁止 if
- **必须**用 `tl.where(at < 1.0, w1, tl.where(at < 2.0, w2, 0.0))` 计算三次权重，禁止 `if` 分支。
- **Why:** Triton kernel 内 `if` 对 vector 数据无效，必须用 `tl.where` 做向量分段。
- **How to apply:** 见 L3.5 `bicubic_vec_kernel` 中的权重计算。

#### L1.9 grid_size 必须动态限制为核数
（见 §1 G4）

#### L1.10 NPU scale 必须用 vdiv 匹配 CANN 精度（仅低精度 ac=True）
- **必须**在 `align_corners=True 且 dtype in (fp16, bf16)` 时用 `_npu_scale`（NPU vdiv）计算 scale，其他场景用 `_cpu_scale`（CPU float32）避免 host-device sync。
- **Why:** 低精度 + ac=True 路径对 scale 精度敏感，CANN 的 vdiv 与 CPU float32 除法结果有微小差异，会导致 verify 不过。其他场景 CPU 计算足够且无同步开销。
- **How to apply:** 见 L2.2 `_npu_scale`、`_cpu_scale`、`_need_npu_scale`。

#### L1.11 ac=False 负坐标 floor 必须修正
- **必须**在 ac=False 计算 `src_y = scale * (i + 0.5) - 0.5` 后，用 `tl.where(src_y < y0_f, y0 - 1, y0)` 修正 floor。
- **Why:** `tl.cast(src_y, tl.int32)` 向零截断而非 floor，src_y 为负时会导致 t 偏移错误，verify 失败（case 13/44 等 upsampling 首行/首列）。
- **How to apply:** 见 L3.5 `bicubic_vec_kernel` 与 L3.7 说明。

#### L1.12 所有 kernel 调用加 multibuffer=True
- **必须**所有 kernel 调用附加 `multibuffer=True`。
- **Why:** 插值 kernel 为内存密集型，大量 global load/store；`multibuffer=True` 是 Ascend 内存密集型 kernel 的默认必备选项。
- **How to apply:** 见 L2.1 所有 kernel 调用末尾的 `multibuffer=True`。

#### L1.13 禁止在 forward 中使用 torch 计算
- **必须**`ModelNew.forward()` 中只负责 shape 计算、分派、host 预计算；所有像素级计算必须在 `@triton.jit` kernel 内完成。
- **Why:** `validate_triton_impl.py` Type-3 检查会 flag 任何 forward 中的 torch 计算为退化。

#### L1.14 bilinear 下采样/general 路径也必须用 host 预计算 x 坐标
- **必须** bilinear 的 **所有** 路径（上采样 + 下采样 + general）都用 `_precompute_bilinear_x_coords(W_in, W_out, align_corners, scale_w)` 预计算 x0/x1/xlambda，传入 kernel。
- **必须** 下采样/general 路径用 `bilinear_general_precomputed_x_kernel`（见 L3.4b），**禁止** 用 `bilinear_general_vec_kernel`（kernel 内重算 x 坐标）。
- **Why:** x 坐标对所有输出行相同，host 预计算一次比 kernel 内每个 w tile 重算（cast/floor/where/clamp 链）快得多。历史上 general 路径用 vec_kernel 会导致 bilinear ac=True 下采样 shape 极慢（<0.3x），Phase 4 改用 precomputed_x 后 11 个 shape 救回，geomean 从 0.53x 升到 0.73x。
- **How to apply:** 见 L3.4b `bilinear_general_precomputed_x_kernel` 与 L2.2 `_precompute_bilinear_x_coords`。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧多策略分派 `_route`

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cores = _get_num_cores()
        self._coord_cache = {}

    def _route(self, x, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias):
        N, C, H_in, W_in = x.shape

        if size is not None:
            H_out, W_out = int(size[0]), int(size[1])
        elif scale_factor is not None:
            H_out = int(H_in * scale_factor)
            W_out = int(W_in * scale_factor)
        else:
            raise ValueError("either size or scale_factor should be defined")

        output = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
        total_rows = N * C * H_out
        grid_cap = int(os.environ.get('TRITON_INTERP_GRID_CAP', str(self.num_cores)))
        grid = (min(total_rows, grid_cap),)   # L1.9
        BLOCK_W = self._select_block_w(W_out)  # pow2, cap 512
        MAX_IN_W = self._select_max_in_w(W_in) # pow2, cap 4096 (L1.2)

        need_npu_scale = _need_npu_scale(x.dtype, align_corners)

        if mode == 'nearest':
            use_size = 1 if size is not None else 0
            if H_out == 2 * H_in and W_out == 2 * W_in:
                nearest_2x_upsample_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W=BLOCK_W, multibuffer=True)               # L1.3
            elif H_in == 2 * H_out and W_in == 2 * W_out:
                nearest_half_downsample_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W=BLOCK_W, multibuffer=True)               # L1.4
            else:
                scale_h, scale_w = _get_scales(
                    [(H_in, H_out), (W_in, W_out)], use_npu=need_npu_scale)
                nearest_vec_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    use_size=use_size, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)

        elif mode == 'bilinear' and align_corners and H_out > 1 and W_out > 1:
            ac = 1
            scale_h, scale_w = _get_scales(
                [(H_in - 1, H_out - 1), (W_in - 1, W_out - 1)],
                use_npu=need_npu_scale)
            if H_out > H_in:
                BLOCK_H = 2
                MAX_KH = 3
                num_h_tiles = (H_out + BLOCK_H - 1) // BLOCK_H
                total_tiles = N * C * num_h_tiles
                grid_2d = (min(total_tiles, grid_cap),)
                bilinear_2d_tiled_vec_kernel[grid_2d](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W,
                    BLOCK_H=BLOCK_H, MAX_KH=MAX_KH, multibuffer=True) # L1.6
            else:
                bilinear_general_vec_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)

        elif mode == 'bilinear':
            ac = 1 if align_corners else 0
            if align_corners:
                pairs = [(H_in - 1 if H_out > 1 else 0, H_out - 1 if H_out > 1 else 1),
                         (W_in - 1 if W_out > 1 else 0, W_out - 1 if W_out > 1 else 1)]
            else:
                pairs = [(H_in, H_out), (W_in, W_out)]
            scale_h, scale_w = _get_scales(pairs, use_npu=need_npu_scale)
            if align_corners and H_out == 1:
                scale_h = 0.0
            if align_corners and W_out == 1:
                scale_w = 0.0
            if not align_corners and H_in == 2 * H_out and W_in == 2 * W_out:
                bilinear_half_pool_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W=BLOCK_W, multibuffer=True)               # L1.5
            elif H_out > H_in:
                BLOCK_H = 2
                MAX_KH = 3
                num_h_tiles = (H_out + BLOCK_H - 1) // BLOCK_H
                total_tiles = N * C * num_h_tiles
                grid_2d = (min(total_tiles, grid_cap),)
                bilinear_2d_tiled_vec_kernel[grid_2d](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W,
                    BLOCK_H=BLOCK_H, MAX_KH=MAX_KH, multibuffer=True) # L1.6
            else:
                bilinear_general_vec_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)

        elif mode == 'bicubic':
            ac = 1 if align_corners else 0
            if align_corners:
                y_all, x_all = self._precompute_bicubic_ac_true_all(
                    H_in, W_in, H_out, W_out)                         # L1.7
                (y_coords, y_idx_m1, y_idx_0, y_idx_p1, y_idx_p2,
                 y_w_m1, y_w_0, y_w_p1, y_w_p2) = y_all
                (x_coords, x_idx_m1, x_idx_0, x_idx_p1, x_idx_p2,
                 x_w_m1, x_w_0, x_w_p1, x_w_p2) = x_all
                device = x.device
                bicubic_ac_true_vec_kernel[grid](
                    x, output,
                    y_idx_m1.to(device), y_idx_0.to(device), y_idx_p1.to(device), y_idx_p2.to(device),
                    y_w_m1.to(device), y_w_0.to(device), y_w_p1.to(device), y_w_p2.to(device),
                    x_idx_m1.to(device), x_idx_0.to(device), x_idx_p1.to(device), x_idx_p2.to(device),
                    x_w_m1.to(device), x_w_0.to(device), x_w_p1.to(device), x_w_p2.to(device),
                    N, C, H_in, W_in, H_out, W_out,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)
            else:
                scale_h, scale_w = _get_scales(
                    [(H_in, H_out), (W_in, W_out)], use_npu=need_npu_scale)
                bicubic_vec_kernel[grid](
                    x, output, N, C, H_in, W_in, H_out, W_out,
                    align_corners=ac, scale_h=scale_h, scale_w=scale_w,
                    BLOCK_W=BLOCK_W, MAX_IN_W=MAX_IN_W, multibuffer=True)

        elif mode == 'area':
            max_kw = (W_in + W_out - 1) // W_out
            max_kw = min(max_kw, 64)
            if max_kw < 1:
                max_kw = 1
            area_vec_kernel[grid](
                x, output, N, C, H_in, W_in, H_out, W_out,
                BLOCK_W=BLOCK_W, MAX_KW=max_kw, MAX_IN_W=MAX_IN_W,
                multibuffer=True)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return output

    def forward(self, x, size=None, scale_factor=None,
                mode='nearest', align_corners=None,
                recompute_scale_factor=None, antialias=False):
        return self._route(x, size, scale_factor, mode, align_corners,
                           recompute_scale_factor, antialias)
```

#### L2.2 辅助函数

```python
def _get_num_cores():
    try:
        device = torch.npu.current_device()
        props = triton.runtime.driver.active.utils.get_device_properties(device)
        num_cores = props.get("num_vectorcore", -1)
        if num_cores <= 0:
            num_cores = props.get("num_aicore", 48)
        return max(num_cores, 1)
    except Exception:
        return 48

def _select_block_w(W_out, cap=512):
    if W_out <= cap:
        return triton.next_power_of_2(W_out)
    return cap

def _select_max_in_w(W_in, cap=4096):
    if W_in <= cap:
        return triton.next_power_of_2(W_in)
    return cap

def _cpu_scale(num, den):
    if den == 0:
        return 0.0
    return float(np.float32(num) / np.float32(den))

_SCALE_CACHE = {}

def _npu_scale(num, den):
    if den == 0:
        return 0.0
    key = (num, den)
    if key not in _SCALE_CACHE:
        a = torch.tensor([num], dtype=torch.float32, device='npu')
        b = torch.tensor([den], dtype=torch.float32, device='npu')
        _SCALE_CACHE[key] = (a / b).cpu().item()
    return _SCALE_CACHE[key]

def _get_scales(pairs, use_npu=False):
    if not use_npu:
        return [_cpu_scale(num, den) for num, den in pairs]

    nums, dens, indices = [], [], []
    for i, (num, den) in enumerate(pairs):
        key = (num, den)
        if key in _SCALE_CACHE:
            continue
        if den == 0:
            _SCALE_CACHE[key] = 0.0
            continue
        nums.append(num)
        dens.append(den)
        indices.append(i)

    if nums:
        a = torch.tensor(nums, dtype=torch.float32, device='npu')
        b = torch.tensor(dens, dtype=torch.float32, device='npu')
        vals = (a / b).cpu().tolist()
        for idx, val in zip(indices, vals):
            _SCALE_CACHE[pairs[idx]] = val

    return [_SCALE_CACHE.get((num, den), _cpu_scale(num, den)) for num, den in pairs]

def _need_npu_scale(dtype, align_corners):
    return align_corners and dtype in (torch.float16, torch.bfloat16)

def _precompute_bicubic_ac_true_all(self, H_in, W_in, H_out, W_out):
    key = ('bicubic_all', H_in, W_in, H_out, W_out)
    if key not in self._coord_cache:
        scale_y = np.float32(H_in - 1) / np.float32(max(H_out - 1, 1))
        scale_x = np.float32(W_in - 1) / np.float32(max(W_out - 1, 1))
        A = -0.75

        def _cubic(t):
            t = abs(float(t))
            if t <= 1.0:
                return float(((A + 2.0) * t - (A + 3.0)) * t * t + 1.0)
            else:
                return float(((A * t - 5.0 * A) * t + 8.0 * A) * t - 4.0 * A)

        def _compute_dim(L_in, L_out, scale):
            coords, idx_m1, idx_0, idx_p1, idx_p2 = [], [], [], [], []
            w_m1, w_0, w_p1, w_p2 = [], [], [], []
            for i in range(L_out):
                s = np.float32(i) * scale
                coords.append(float(s))
                y0 = int(np.float32(np.floor(s)))
                if s < 0:
                    y0 -= 1
                wy = float(s - np.float32(y0))
                idx_m1.append(max(0, min(y0 - 1, L_in - 1)))
                idx_0.append(max(0, min(y0, L_in - 1)))
                idx_p1.append(max(0, min(y0 + 1, L_in - 1)))
                idx_p2.append(max(0, min(y0 + 2, L_in - 1)))
                w_m1.append(_cubic(-1.0 - wy))
                w_0.append(_cubic(0.0 - wy))
                w_p1.append(_cubic(1.0 - wy))
                w_p2.append(_cubic(2.0 - wy))
            return (
                torch.tensor(coords, dtype=torch.float32),
                torch.tensor(idx_m1, dtype=torch.int32),
                torch.tensor(idx_0, dtype=torch.int32),
                torch.tensor(idx_p1, dtype=torch.int32),
                torch.tensor(idx_p2, dtype=torch.int32),
                torch.tensor(w_m1, dtype=torch.float32),
                torch.tensor(w_0, dtype=torch.float32),
                torch.tensor(w_p1, dtype=torch.float32),
                torch.tensor(w_p2, dtype=torch.float32),
            )

        y_all = _compute_dim(H_in, H_out, scale_y)
        x_all = _compute_dim(W_in, W_out, scale_x)
        self._coord_cache[key] = (y_all, x_all)
    return self._coord_cache[key]
```

> Layer 3 关键 kernel 按 mode 拆为三张卡：
> [nearest](tensor_transform__interpolate_nearest.md)、[bilinear](tensor_transform__interpolate_bilinear.md)、[bicubic/area](tensor_transform__interpolate_bicubic_area.md)。
