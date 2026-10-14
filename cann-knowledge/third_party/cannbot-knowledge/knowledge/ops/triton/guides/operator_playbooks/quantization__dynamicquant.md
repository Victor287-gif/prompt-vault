---
type: Guide
title: DynamicQuant 算子（reduction + element-wise 复合） 实现手册
description: '算子类别: reduction + elementwise 复合（perrow max_abs reduction + 量化）。'
tags: [task.implementation, topic.compute.quantization, topic.implementation.masking, topic.implementation.memory_management, topic.tiling, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/sum.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/reduce.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# DynamicQuant 算子（reduction + element-wise 复合） 实现手册

> 类别约束见 [quantization_constraints.md](../constraints/quantization_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 DynamicQuant 算子（reduction + element-wise 复合）

**算子类别**: `reduction` + `element-wise` 复合（per-row max_abs reduction + 量化）
**典型特征**: 输入 x `[M, K]` 或 `[B, M, K]` (fp16/bf16) + 可选 smooth_scales `[K]`，输出 int8 quant + float32 scale `[M]`。算法要求两阶段：pass1 计算 per-row max_abs → scale，pass2 用 scale 量化
**性能基准**: 42/42 verify pass，几何平均加速比 **0.3604x** vs torch（**未达 0.6x 用户目标，未达 1.0x 归档阈值，破例归档**）

### §2.0 破例归档说明

本版本 geomean 0.3604x，远低于归档阈值 1.0x 和用户目标 0.6x。破例归档理由：
1. **架构限制典型案例**：Triton DSL 无法暴露 CCE 级 UB 软件管理，memory-bound 算子大 K 场景必然 2-pass HBM，与 torch 单遍融合 kernel 存在结构性带宽劣势。这是**已知不可优化**的场景。
2. **7 种失败优化方向**：记录已验证无效的优化策略，避免后续任务重复踩坑。
3. **量化类算子复用价值**：后续做 per-group quant / weight quant 时，若大 K 慢于 torch，应首先排查 2-pass HBM。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 大 K (>8192) 场景必然 2-pass HBM，禁止期望超越 torch（架构限制）
- **必须**认识到：DynamicQuant 在 K > 8192 时，整行无法放入 UB (192KB)，必须分块处理，导致 x 被从 HBM 读取两次（pass1 max_abs + pass2 quant）。
- **禁止**在 K > 8192 场景期望通过 tile 优化、循环重排等手段超越 torch。torch 的 CCE 底层算子是**单遍融合 kernel**——在 UB 中缓存分块数据，pass1 计算 max_abs 后直接在 UB 中完成 pass2 量化，全程只从 HBM 读取 x 一次。
- **Why:** 实测带宽：torch 818 GB/s (68% HBM 峰值)，triton 277 GB/s (23% HBM 峰值)。2x HBM 读取 = 2x 带宽劣势，这是架构层面的硬限制。
- **How to apply:** 大 K 场景直接接受 0.2x~0.3x 加速比，不要浪费优化轮次。把精力放在小 K (≤4096) 场景的 single-pass 优化上。

#### L1.2 禁止用 L2 cache 命中作为 2-pass 优化手段（已验证无效，负面经验/失败方向）
- **禁止**在单 kernel 内用两个循环（pass1 max_abs → pass2 quant）期望 L2 cache 保留 pass1 的数据供 pass2 复用。
- **Why:** 实测无改善（0.37x），L2 cache 在大 K 场景未有效保留数据。Ascend L2 cache 容量和替换策略不适合跨 pass 数据复用。
- **How to apply:** 2-pass 就是 2-pass HBM，不要试图用 L2 cache "欺骗" 成 single-pass。

#### L1.3 禁止用非对称 BLOCK_K 优化 2-pass（已验证无效，负面经验/失败方向）
- **禁止**给 pass1 (max_abs) 用大 BLOCK_K (8192/16384)、pass2 (quant) 用小 BLOCK_K (4096)，期望减少 pass1 的中间量。
- **Why:** 实测无显著改善。瓶颈是 HBM 带宽而非中间量寄存器压力。
- **How to apply:** 两个 pass 用相同 BLOCK_K 即可。

#### L1.4 禁止用 bf16 max_abs 避免 fp32 膨胀（已验证无效，负面经验/失败方向）
- **禁止**在 bf16 精度下计算 max_abs，期望避免 fp32 的 4x 内存膨胀。
- **Why:** 实测无改善。max_abs 的精度需求使 bf16 中间结果仍需转 fp32 累积，且膨胀发生在寄存器而非 HBM。
- **How to apply:** 直接用 fp32 累积 max_abs。

#### L1.5 禁止用 inv_scale 乘法替代除法（已验证反而回退，负面经验/失败方向）
- **禁止**用 `x * (1/scale)` 替代 `x / scale` 期望加速。
- **Why:** 实测反而回退。Ascend 的乘法和除法延迟相近，额外的 `1/scale` 计算和 inv_scale reshape 反而增加开销。
- **How to apply:** 直接用 `x / scale` 或在 kernel 内 `inv_scale = 1.0 / scale` 后 broadcast 乘法——两者性能相近，选可读性更好的。

#### L1.6 禁止在大 K 场景用 BLOCK_M > 1（UB 溢出）
- **禁止**在 K > 4096 场景用 `BLOCK_M > 1` 的 2D tile，会导致 UB 容量溢出。
- **Why:** UB 容量 192KB，`BLOCK_M * BLOCK_K * sizeof(fp32)` 易超限。大 K 时必须 `BLOCK_M = 1` 或极小值。
- **How to apply:** `_choose_block_size`: `cols >= 512` 用 `(8, 512)`；否则 `block_m = max(1, min(32, 4096 // block_n))`。

#### L1.7 Triton DSL 无法暴露 CCE 级优化（架构限制，禁止尝试）
- **禁止**尝试以下 CCE 级优化，Triton DSL 不支持：
  - 软件管理 UB 的分块缓存（在 UB 中保留 pass1 数据供 pass2 使用）
  - 跨核同步原语（atomic max + barrier 实现单遍跨块 reduction）
  - DMA 异步传输与计算的 overlap
- **Why:** 这些是 torch CCE 单遍融合 kernel 的核心优化手段，Triton DSL 层面无法表达。
- **How to apply:** 遇到 memory-bound + 跨 pass 数据复用的算子，直接判定为架构限制场景，不要尝试逆向工程 CCE 优化。

#### L1.8 小 K (≤4096) 必须用 single-pass fused kernel（可达 1.0x+）
- **必须**在 K ≤ 4096 时用 single-pass fused kernel——整行一次性加载到 UB，单遍完成 max_abs + 量化。
- **Why:** 小 K 整行可放入 UB，避免 2-pass HBM，可达甚至超越 torch 性能（实测部分 case 1.192x）。
- **How to apply:** `_choose_block_size` 根据 cols 选择 BLOCK_N = pow2(cols)，整行单块处理。

#### L1.9 必须用 tl.cast overflow_mode='saturate' 替代手动 clamp
- **必须**用 `tl.cast(q, tl.int8, overflow_mode='saturate')` 把浮点量化结果转为 int8，禁止手动 `tl.maximum(tl.minimum(q, 127), -128)`。
- **Why:** `overflow_mode='saturate'` 是 Ascend 硬件原生支持的饱和转换，比手动 clamp 更快且精度一致。
- **How to apply:**
  ```python
  q = tl.where(q >= 0, q + 0.5, q - 0.5)   # round to nearest
  q = tl.cast(q, tl.int8, overflow_mode='saturate')   # saturate to [-128, 127]
  ```

#### L1.10 必须用 round-half-away-from-zero 而非 tl.round
- **必须**用 `tl.where(q >= 0, q + 0.5, q - 0.5)` 然后 `tl.cast` 实现 round-half-away-from-zero，禁止用 `tl.round`。
- **Why:** `tl.round` 默认 round-half-to-even（banker's rounding），与 torch `npu_dynamic_quant` 的 round-half-away-from-zero 语义不一致，会导致 verify 失败。
- **How to apply:** 见 L1.9 代码片段。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分支决策树

```python
def forward(self, x, smooth_scales=None, group_index=None, dst_type=None):
    if dst_type is None: dst_type = torch.int8
    if group_index is not None:
        raise NotImplementedError("Group quantization not supported")

    x = x.contiguous()
    orig_shape = x.shape
    dim = x.dim()

    # 1. shape 归一化为 2D
    if dim == 2:
        rows, cols = x.shape
        quant = torch.empty(orig_shape, device=x.device, dtype=dst_type)
        scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    elif dim == 3:
        b, m, cols = x.shape
        rows = b * m
        quant = torch.empty(orig_shape, device=x.device, dtype=dst_type)
        scale = torch.empty((b, m), device=x.device, dtype=torch.float32)
    else:
        x_2d = x.reshape(1, -1)
        rows, cols = x_2d.shape
        # ... 1D 退化处理

    # 2. 自适应 block size (L1.6 / L1.8)
    block_m, block_n = self._choose_block_size(cols)
    grid = (min(triton.cdiv(rows, block_m), self.num_cores),)

    # 3. 按 smooth_scales 分派
    if smooth_scales is None:
        dynamic_quant_fused_2d_kernel_no_smooth[grid](
            x, quant, scale, rows, cols, cols, cols,
            BLOCK_M=block_m, BLOCK_N=block_n,
        )
    else:
        smooth_scales = smooth_scales.contiguous()
        dynamic_quant_fused_2d_kernel[grid](
            x, smooth_scales, quant, scale, rows, cols, cols, cols,
            BLOCK_M=block_m, BLOCK_N=block_n,
        )

    if dim not in (2, 3):
        quant = quant.reshape(orig_shape)
    return quant, scale
```

#### L2.2 _choose_block_size 自适应策略

```python
def _choose_block_size(self, cols):
    if cols >= 512:
        return 8, 512          # 宽行: (8, 512) tile 摊销循环开销
    cols_pow2 = triton.next_power_of_2(cols)
    block_n = cols_pow2
    block_m = max(1, min(32, 4096 // block_n))   # 窄行: tile 约 4K elements
    return block_m, block_n
```

#### L2.3 Kernel 两阶段循环骨架（2D tiling）

```python
@triton.jit
def dynamic_quant_fused_2d_kernel(
    x_ptr, smooth_ptr, quant_ptr, scale_ptr,
    ROWS, COLS, ROW_STRIDE, QUANT_STRIDE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_row_blocks = tl.cdiv(ROWS, BLOCK_M)

    # 1D Grid + 交织循环 (Q2)
    for block_idx in range(pid, num_row_blocks, num_programs):
        row_start = block_idx * BLOCK_M
        row_offsets = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < ROWS

        # === Pass 1: 累积 max_abs ===
        max_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for col_start in range(0, COLS, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < COLS
            mask2d = row_mask[:, None] & col_mask[None, :]

            x_tile = tl.load(
                x_ptr + row_offsets[:, None] * ROW_STRIDE + col_offsets[None, :],
                mask=mask2d, other=0.0
            ).to(tl.float32)
            smooth_tile = tl.load(smooth_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
            x_tile = x_tile * smooth_tile[None, :]   # smooth broadcast 到行
            max_acc = tl.maximum(max_acc, tl.max(tl.abs(x_tile), axis=1))

        # === 计算 scale ===
        scale = max_acc / 127.0
        scale = tl.maximum(scale, 1e-10)
        tl.store(scale_ptr + row_offsets, scale, mask=row_mask)

        # === Pass 2: 量化 ===
        inv_scale = tl.reshape(1.0 / scale, (BLOCK_M, 1))   # broadcast 准备
        for col_start in range(0, COLS, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < COLS
            mask2d = row_mask[:, None] & col_mask[None, :]

            x_tile = tl.load(
                x_ptr + row_offsets[:, None] * ROW_STRIDE + col_offsets[None, :],
                mask=mask2d, other=0.0
            ).to(tl.float32)
            smooth_tile = tl.load(smooth_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
            q = x_tile * smooth_tile[None, :] * inv_scale
            q = tl.where(q >= 0, q + 0.5, q - 0.5)           # round-half-away-from-zero (L1.10)
            q = tl.cast(q, tl.int8, overflow_mode='saturate')  # 饱和转换 (L1.9)
            tl.store(
                quant_ptr + row_offsets[:, None] * QUANT_STRIDE + col_offsets[None, :],
                q, mask=mask2d
            )
```

### §2.3 Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

#### L3.1 tl.cast overflow_mode='saturate' 替代手动 clamp

**问题**: 量化结果需 clamp 到 `[-128, 127]`，手动 `tl.maximum(tl.minimum(q, 127), -128)` 生成多条指令。

**解决**: `tl.cast(q, tl.int8, overflow_mode='saturate')` 是 Ascend 硬件原生饱和转换，单条指令完成。

```python
q = tl.where(q >= 0, q + 0.5, q - 0.5)   # round
q = tl.cast(q, tl.int8, overflow_mode='saturate')   # saturate
```

**可替代方向**: 若 `overflow_mode` 参数不可用（旧版 triton-ascend），回退到手动 clamp，但性能略差。

#### L3.2 inv_scale reshape broadcast 替代逐元素除法

**问题**: `x_tile / scale` 中 scale 是 `[BLOCK_M]`，x_tile 是 `[BLOCK_M, BLOCK_N]`，需要 broadcast 除法。

**解决**: `inv_scale = tl.reshape(1.0 / scale, (BLOCK_M, 1))`，然后 `x_tile * inv_scale` 利用 broadcast 乘法。预先计算 `1/scale` 避免逐元素除法。

**注意**: L1.5 标注此优化"反而回退"，但在某些 case 仍可能有效，需实测验证。

#### L3.3 smooth_scales broadcast 到行

**问题**: smooth_scales 是 `[K]`（per-column），需应用到 `[BLOCK_M, BLOCK_N]` tile 的每一行。

**解决**: `x_tile * smooth_tile[None, :]`，`smooth_tile` 加载为 `[BLOCK_N]`，`[None, :]` broadcast 到 `[BLOCK_M, BLOCK_N]`。

**可替代方向**: 若 smooth_scales 全 1（无 smooth），用 `_no_smooth` kernel 避免额外 load。

#### L3.4 _choose_block_size 自适应 tile 选择

**问题**: 不同 K 需要不同 BLOCK_N，固定 tile 会在小 K 浪费、大 K 溢出。

**解决**:
```python
if cols >= 512:
    return 8, 512                          # 宽行: 摊销循环开销
cols_pow2 = triton.next_power_of_2(cols)
block_n = cols_pow2
block_m = max(1, min(32, 4096 // block_n))  # 窄行: tile 约 4K elements
```

**关键点**: `4096 // block_n` 保证 tile 总量约 4K elements，避免 UB 溢出。

**可替代方向**: 若 K 固定已知，可离线调优最优 (BLOCK_M, BLOCK_N) 组合。

#### L3.5 3D 输入 reshape 为 2D 处理

**问题**: 3D 输入 `[B, M, K]` 需要按 `[B, M]` 每行计算 scale。

**解决**: 不 reshape x（保持 stride），直接用 `rows = b * m` 把 3D 当 2D 处理，scale 输出为 `[B, M]` shape。

**可替代方向**: 若 3D 不连续，需先 `.contiguous()` 再处理。

#### L3.6 将 `COLS` 声明为 `tl.constexpr` 以启用编译期地址推导

**问题**: `ROW_STRIDE` / `QUANT_STRIDE` 作为运行时 kernel 参数传入时，编译器无法静态确定每行起始地址的 stride，导致地址计算无法完全向量化/常量化。

**解决**: 在 host 侧保证输入 `contiguous` 并把最后一维展平后，`ROW_STRIDE == QUANT_STRIDE == COLS`。将 `COLS` 声明为 `tl.constexpr`，kernel 内直接用 `row_offsets[:, None] * COLS + col_offsets[None, :]`，编译器可静态推导地址模式。

**正确示例**:
```python
@triton.jit
def dynamic_quant_kernel(
    x_ptr, quant_ptr, scale_ptr,
    ROWS, COLS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # ...
    x_tile = tl.load(
        x_ptr + row_offsets[:, None] * COLS + col_offsets[None, :],
        mask=mask2d, other=0.0
    ).to(tl.float32)
```

**效果**: 在 ascend910b2 / 42 cases 评测中，该优化使几何平均加速比从 0.795x 提升至 0.891x（相对基线 0.696x）。

**可替代方向**: 若输入无法保证 stride==cols，仍需保留 `ROW_STRIDE` 参数，但优先在 host 侧通过 `.view`/`.contiguous()` 规整 stride。

#### L3.7 小 K (≤4096) single-block 特化路径

**问题**: 通用 2-pass kernel 使用固定 BLOCK_N=512，小 K（如 128、256、1024）会被拆成多个 col 循环，增加 mask 计算和循环开销。

**解决**: 当 `COLS ≤ 4096` 时，启用 single-block 特化 kernel：`BLOCK_N = next_power_of_2(COLS)`，整行一次性加载，单 kernel 内完成 max + quant，避免 col 循环。

**正确示例**:
```python
# host 侧分派
if cols <= 4096:
    block_n = triton.next_power_of_2(cols)
    block_m = max(1, min(32, 4096 // block_n))
    dynamic_quant_single_block_kernel[grid](...)
else:
    block_m, block_n = 8, 512
    dynamic_quant_fused_2d_kernel[grid](...)
```

**效果**: 与 `COLS` constexpr 优化配合，小 K case 加速比普遍 >1.0x，部分达到 2.8x+。

**可替代方向**: 对中等 K（2048-4096）可尝试 BLOCK_N=1024/2048 而非整行，以平衡 register 压力和循环次数。

### §2.4 DynamicQuant 性能基准

| 版本 | K 范围 | cases | geomean | 区间 | 说明 |
|------|--------|-------|---------|------|------|
| 历史基线 | K ≤ 4096 | 20 | 0.590x | [0.219, 1.192] | 接近目标，小张量部分超 1.0x |
| 历史基线 | 4096 < K ≤ 8192 | 4 | 0.311x | [0.141, 0.475] | 中等表现 |
| 历史基线 | K > 8192 | 14 | 0.186x | [0.158, 0.236] | 瓶颈所在，2-pass HBM 限制 |
| 历史基线 | 全量 42 cases | 42 | 0.3604x | — | 未达 0.6x 目标，破例归档 |
| **v2 (2026-07-02)** | 全量 42 cases | 42 | **0.8910x** | [0.2886, 3.5556] | 加入 `COLS` constexpr + 小 K single-block 特化后，在 ascend910b2 环境达成目标 0.8x |

**带宽实测对比**：

| 实现 | HBM 读取量 | 延迟 | 带宽 | HBM 峰值利用率 |
|------|-----------|------|------|---------------|
| torch 单遍 | 90 MB | 0.11 ms | 818 GB/s | 68% |
| triton 双遍 | 180 MB | 0.65 ms | 277 GB/s | 23% |

**关键结论**:
1. DynamicQuant 是 Triton DSL 架构限制的边界案例——memory-bound + 大 K + 跨 pass 数据复用
2. 7 种优化策略全部失败，证明瓶颈在架构层而非实现层
3. 小 K (≤4096) 可达 1.0x+，是可用区间；大 K (>8192) 0.19x 是架构硬限制
4. **新经验**: 将 `COLS` 声明为 `tl.constexpr` 并配合小 K single-block 特化，可在 ascend910b2 上把全量 42 cases 几何平均加速比从 ~0.7x 提升至 0.89x，超过 0.8x 目标
5. 后续量化类算子若大 K 慢于 torch，应首先排查 2-pass HBM

---
