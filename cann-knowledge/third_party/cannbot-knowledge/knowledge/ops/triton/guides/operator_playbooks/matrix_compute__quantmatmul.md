---
type: Guide
title: QuantMatmul 算子（quantization-matmul） 实现手册
description: '算子类别: quantizationmatmul。'
tags: [task.implementation, topic.compute.composite, topic.implementation.masking, topic.compute.quantization, topic.tiling, topic.implementation.memory_management]
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
# QuantMatmul 算子（quantization-matmul） 实现手册

> 类别约束见 [matrix_compute_constraints.md](../constraints/matrix_compute_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 QuantMatmul 算子（quantization-matmul）

**算子类别**: `quantization-matmul`
**典型特征**: int8×int8 矩阵乘，后接反量化（scale/pertoken_scale）、bias、offset、cast 到 fp16/bf16/int8/int32
**性能基准**: 57 cases 精度全过；Triton 实现几何平均 **0.9282x** vs `torch_npu.npu_quant_matmul`（ascend910_9362）

> **评判口径**: framework 参考实现为 `torch_npu.npu_quant_matmul`（高度优化原生算子），`speedup_vs_torch < 1` 属于正常现象；优化目标为 **达到 0.8x**。本次 Phase 3 基线即达到 **0.9282x**，超过目标。

> **⚠️ 复现注意**: 该算子单次 benchmark 易受并发 profiler 干扰出现虚高/虚低，关键结论应基于**隔离多次 benchmark**（repeats=20，至少 3 次取稳定值），不要采信含异常 case（如某个 case speedup 突增至数百倍）的单次结果。

### §2.1 算子语义

**输入**:
- `x1`: int8, shape 2-6D, 最后两维 `[M, K]`
- `x2`: int8, shape 2-6D, 最后两维 `[K, N]`
- `scale`: fp32/bf16, shape `[N]`, 列方向反量化 scale
- `pertoken_scale`: fp32, shape `[B, M]` or `[M]`, 行方向 per-token scale（可选）
- `bias`: fp32/bf16/fp16/int32, shape `[N]` or `[B, 1, N]`（可选）
- `offset`: fp32/bf16, shape `[N]`（可选）
- `output_dtype`: 输出 dtype（fp16/bf16/int8/int32）

**输出**:
- `output`: 按 `output_dtype`, shape `[B, M, N]`（B 为展平后的 batch）

**计算流程**:
1. int8×int8 matmul，int32 accumulator
2. 转 fp32
3. 乘 scale
4. （如有 pertoken_scale）再乘 pertoken_scale
5. （如有 bias）加 bias
6. （如有 offset）加 offset
7. cast 到 output_dtype

### §2.2 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须使用 int32 accumulator 做 int8 matmul

- **必须**使用 `tl.dot(x1_tile, x2_tile, acc)`，accumulator dtype 为 `tl.int32`。
- **禁止**使用 fp32/fp16 accumulator 或直接转浮点后做 matmul。
- **Why:**
  - `torch_npu.npu_quant_matmul` 在 int8 量化语义下，matmul 阶段使用 int32 累加，后处理阶段再转 fp32 反量化。
  - 使用浮点累加器会改变舍入行为，导致 fp16 输出 case 精度不过。
- **How to apply:** 所有 QuantMatmul kernel 内初始化 `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)`，并在 K 维循环中 `acc = tl.dot(a, b, acc)`。

#### L1.2 必须根据 shape 动态选择 tile size

- **禁止**对所有 shape 使用单一固定大 tile。
- **必须**根据 `(B, M, N, K)` 在运行时分派 tile，至少区分小 shape（64×64 / 64×128）与中大 shape（128×128）。
- `BLOCK_K` 根据 K 选 256 或 512； ascend910_9362 上实测 `BLOCK_K=256` 更稳（`BLOCK_K=512` 时 multi-buffer 下 a+b 缓冲达 256KB 超出 UB 192KB）。

#### L1.3 小 shape 必须限制 grid 大小

- 当 `TOTAL_TILES = B*ceil(M/BLOCK_M)*ceil(N/BLOCK_N)` 小于 cube core 数时，必须使用 `grid = min(num_cores, TOTAL_TILES)`。
- **禁止**对所有 shape 固定满核 grid。
- **Why:** 小 shape 下固定满核 grid 会造成大量空 program，调度 overhead 远大于计算收益。

#### L1.4 fp16 输出的 scale 截断方式

- 当 `output_dtype==float16` 且 `scale.dtype==fp32` 且 `pertoken_scale is None` 时，host 端对 scale **向零截断到 fp16 再扩回 fp32** 传入；禁止四舍五入截断。
- 当存在 `pertoken_scale` 时**禁止**对 scale 做 fp16 截断，保持 fp32。
- **注意**: 若任务 `get_input_groups` 已统一对 fp16 输出 / fp32 scale / 无 pertoken 的 case 做 `_crop_float32_to_19bit` 预处理（与 `forward_native` 文档一致），且该 scale 同时喂给参考与 Triton，则 kernel 内可直接 fp32 乘 scale，不再叠加旧版 fp16 向零截断。

#### L1.5 后处理顺序必须为 acc * scale * pertoken (+ bias + offset)

- 必须先乘 scale，再乘 pertoken_scale，最后加 bias/offset。
- **禁止**改变顺序（int32 bias 且无 pertoken 除外，见 L1.6）。

#### L1.6 int32 bias 且无 pertoken_scale 时，顺序为 (acc + bias) * scale

- bias dtype 为 int32 且 `pertoken_scale` 为 None 时，必须在 int32 accumulator 上先加 bias，再转 fp32 乘 scale。
- **Why:** 与 `torch_npu.npu_quant_matmul` 的 int32 bias 路径语义一致。

#### L1.7 禁止在 forward() 中引入大量 host 端辅助算子

- scale 截断、pertoken_scale expand、bias broadcast 等逻辑必须尽量下沉到 Triton kernel 内部。
- **禁止**在 forward() 保留 `nonzero` / `index_put` / `nextafter` 等 host 辅助算子链。
- **Why:** host 辅助算子会引入同步点和额外 kernel，抵消矩阵乘本身的收益。

#### L1.8 必须 flatten batch 维度到 3D 后再启动 kernel

- 必须将 x1/x2 的 leading dims 展平为 B，reshape 为 `[B,M,K]` / `[B,K,N]`。
- **禁止**在 kernel 内处理 4D/5D/6D 索引。

### §2.3 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧调度流程

```python
# 1. 解析输出 dtype
# 2. flatten batch -> 3D (B, M, K) / (B, K, N)
# 3. scale -> fp32 1D；pertoken_scale -> [M] fp32；bias/offset -> 1D 或按 batch 处理
# 4. 选择 tile size (BLOCK_M, BLOCK_N, BLOCK_K)
# 5. grid = (min(CUBE_CORE_NUM, TOTAL_TILES),)
# 6. launch single quant_matmul_kernel
# 7. reshape 输出回原始 leading dims + (M, N)
```

#### L2.2 动态 tile 选择策略

```python
def _select_tile(M, N, K, simple, do_clamp):
    block_k = 256
    total_mn = M * N
    # 无需 clamp 的大 shape 可用 128x128
    if (not do_clamp) and M >= 128 and N >= 128 and total_mn > 32 * 1024:
        return 128, 128, block_k
    if total_mn <= 32 * 1024:
        return 64, 64, block_k
    elif M <= 64:
        return 64, 128 if N > 128 else 64, block_k
    elif N <= 64:
        return 128 if M > 128 else 64, 64, block_k
    elif M <= 128 and N <= 128:
        return 64, 64, block_k
    else:
        return 64, 128, block_k
```

- `simple = (not has_pertoken) and (not has_bias) and (not has_offset)`：无后处理时 UB 占用小，可用 128×128 大 tile。
- `do_clamp` 为 True 时（fp16/int8 输出），clamp 会物化 2 个全尺寸临时缓冲，128×128 易超出 UB（实测 258KB > 192KB），应回退 64×128 / 64×64。

### §2.4 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 int32 accumulator 与 K 维循环

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
for k in range(0, K, BLOCK_K):
    a_off = x1_base + (m0 + rm)[:, None] * K + (k + rk)[None, :]
    a_mask = ((m0 + rm)[:, None] < M) & ((k + rk)[None, :] < K)
    a = tl.load(x1_ptr + a_off, mask=a_mask, other=0)

    b_off = x2_base + (k + rk)[:, None] * N + (n0 + rn)[None, :]
    b_mask = ((k + rk)[:, None] < K) & ((n0 + rn)[None, :] < N)
    b = tl.load(x2_ptr + b_off, mask=b_mask, other=0)

    acc = tl.dot(a, b, acc)
```

#### L3.2 后处理分支（scale / pertoken / bias / offset / clamp）

```python
n_mask = (n0 + rn) < N
scale_t = tl.load(scale_ptr + (n0 + rn), mask=n_mask, other=0.0)

if BIAS_INT32 and (not HAS_PERTOKEN):
    # (acc + bias) * scale  (L1.6)
    bias_off = bb * N + (n0 + rn) if BIAS_BATCHED else n0 + rn
    bias_i = tl.load(bias_ptr + bias_off, mask=n_mask, other=0)
    acc = acc + bias_i[None, :]
    acc_f = acc.to(tl.float32) * scale_t[None, :]
else:
    acc_f = acc.to(tl.float32) * scale_t[None, :]
    if HAS_PERTOKEN:
        m_mask = (m0 + rm) < M
        pt = tl.load(pertoken_ptr + (m0 + rm), mask=m_mask, other=0.0)
        acc_f = acc_f * pt[:, None]
    if HAS_BIAS:
        bias_off2 = bb * N + (n0 + rn) if BIAS_BATCHED else n0 + rn
        bias_t = tl.load(bias_ptr + bias_off2, mask=n_mask, other=0.0)
        acc_f = acc_f + bias_t.to(tl.float32)[None, :]
    if HAS_OFFSET:
        off_t = tl.load(offset_ptr + (n0 + rn), mask=n_mask, other=0.0)
        acc_f = acc_f + off_t[None, :]

if DO_CLAMP:
    acc_f = tl.minimum(tl.maximum(acc_f, out_min), out_max)

out = acc_f.to(out_ptr.dtype.element_ty)
```

#### L3.3 1D grid 索引解码（禁止 %）

```python
pid = tl.program_id(0)
NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
TILES_PER_BATCH = NUM_BLOCKS_M * NUM_BLOCKS_N
NUM_BLOCKS = B * TILES_PER_BATCH

for tile in range(pid, NUM_BLOCKS, num_cores):
    bb = tile // TILES_PER_BATCH
    rem = tile - bb * TILES_PER_BATCH
    bm = rem // NUM_BLOCKS_N
    bn = rem - bm * NUM_BLOCKS_N
    m0 = bm * BLOCK_M
    n0 = bn * BLOCK_N
```

### §2.5 性能基准

| 版本 | 日期 | geomean | 精度 | 关键改动 |
|------|------|---------|------|---------|
| Phase 3 基线 | 2026-07-13 | **0.9282x** | 57/57 | int32 acc + 动态 tile + 19-bit crop scale 对齐 |

- **优化目标 0.8x 已达成**；Phase 4 尝试调整 tile 启发式未带来进一步提升，回退到基线。

---
