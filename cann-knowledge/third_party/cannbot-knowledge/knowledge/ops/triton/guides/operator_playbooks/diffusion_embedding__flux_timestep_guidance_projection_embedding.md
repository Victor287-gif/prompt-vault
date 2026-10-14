---
type: Guide
title: FluxTimestepGuidanceProjectionEmbedding 算子（diffusion-embedding） 实现手册
description: '典型特征: timestep[B]×1000 → 外积 freqs[384] → sin/cos concat 得 [B,768]；Linear(768→3072)+SiLU+Linear(3072→768)
  全程 fp32；pooled[B,3072]@Wt^T；两路相加 cast 回输入 dtype。B∈[1,384]，60 case。'
tags: [task.implementation, topic.compute.composite, topic.compute.elementwise.activation, topic.precision, topic.programming_model.execution_unit.vector, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-31T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# FluxTimestepGuidanceProjectionEmbedding 算子（diffusion-embedding） 实现手册

> 类别约束见 [diffusion_embedding_constraints.md](../constraints/diffusion_embedding_constraints.md)（DE1~DE10），
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 10_FluxTimestepGuidanceProjectionEmbedding 算子（本类首个实例）

**算子类别**: `diffusion-embedding`（F1-F6 全命中）
**典型特征**: timestep[B] ×1000 → 外积 freqs[384] → sin/cos concat 得 [B,768]；Linear(768→3072)+SiLU+Linear(3072→768) 全程 fp32；pooled[B,3072]@Wt^T；两路相加 cast 回输入 dtype。B∈[1,384]，60 case，dtype ∈ {fp16, bf16, fp32} × bias {fp32, lowp}
**性能基准**: 60/60 pass，几何平均 **1.4083x** vs torch（target 5.0 未达；7 轮迭代后确认平台期，~±0.2 点为同代码 run-to-run 噪声带）

> 本节的常量（HALF=384 / TD=768 / INNER=3072 / 档位边界 128、17、96）是**本例的标定值**。新算子入类时：宽度替换为该算子的模型超参；档位边界按其 B 分布重新 probe 标定（DE5 的 law 形式不变）。

### §2.0 首次生成必读：为什么必须把段级融合框架写对

首次生成最大的坑是 **GEMM 实现路径**：按大 M 直觉写 `tl.dot` 得到 300-1600x 灾难退化（DE1）。其次是**中间舍入**：漏掉参考实现的 `.to(bias.dtype)` 回转，verify 位级失败且难定位（DE3）。骨架必须在 Phase 2 就定为：**段级融合 + 向量外积 + 舍入 constexpr 复现**；结构错了后续 round 在 tile 上救不回。

### §2.1 Layer 1: 设计约束（引用 §1 + 本例具体化）

#### L1.1 tl.dot 禁用 → 向量外积（同 DE1，本例 probe 结论：全族禁用）
#### L1.2 `F.linear` 语义 `y = x @ Wᵀ`：kernel 传 `w.stride(1), w.stride(0)`（交换 stride），行主 `[N,K]` 直接 load，无 Host 侧 transpose
- 权重从 forward 入参传入（无 RNG 复刻）；方阵下算成 `x @ W` shape 检查不报错、只有数值错

#### L1.3 3-kernel 结构（同 DE2，本例阶段划分）
- K1 `_timestep_embed_kernel`（纯 Vector：scale/outer/sin/cos/cat 单 kernel，中间不落 HBM）
- K2 `_vec_linear_silu_kernel`（linear1 + bias + SiLU）
- K3+4 `_vec_dual_linear_add_cast_kernel`（DE4 双路 pass-merge + per-branch round + add + cast）

#### L1.4 tile law（同 DE5/DE7，本例标定值；bm32 档下界 32→17 为 round-6 两次 run 认证）
```python
bm  = 64 if B >= 128 else 32 if B >= 17 else 16
# 边界演进：初始 32 → round-4 调至 128/32 → round-6 把 bm32 下界 32→17（B∈[17,31]
# 从 bm16/NUM_M=2 升为 bm32/NUM_M=1，少付一整趟 W 面板重读，5 case 各 +33~38%）
bn2 = 256 if (bm == 32 and B >= 96) else 128   # 仅 K2（N=3072 段），NUM_M>=3 保核饱和
BK = 128                                        # 缩小全线回归，不试
```

#### L1.5 精度契约（同 DE3）：linear2 / textproj 输出各自 round 到自己 bias dtype 后再相加，最终 cast 到 `result_type` 推导的 out_dtype
#### L1.6 动态核数 + 1D 扁平 grid（同 tensor-transform G1/G4）：`get_device_properties` 取核数禁硬编码；GEMM kernel grid = `N_tiles * NUM_M` 一维展开，`pid_n = pid // NUM_M`；K1 用 `grid = min(NVEC, B)` + 核内步长循环
#### L1.7 硬顶清单（同 DE7）：BM≤64（1M numel cap）；禁 BM=4/128、fp16 3D 中间量（`ConvertLinalgRToBi`）；`multibuffer=True`
#### L1.8 grid 禁共享（同 DE8）：K2 用 `INNER//bn2`、K3+4 用 `TDIM//BN`，独立计算

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分派决策树

```python
# 宽度常量来自算子定义（本例 384/768/3072），泛化时替换
bm = 64 if B >= 128 else 32 if B >= 32 else 16        # 档位边界按 B 分布实测标定
bn2 = 256 if (bm == 32 and B >= 96) else 128
NUM_M = (B + bm - 1) // bm
grid2  = (D_inner // bn2) * NUM_M                     # K2: N = D_inner
grid34 = (D_out // BN) * NUM_M                        # K3+4: N = D_out，禁与 grid2 共享
# out_dtype 按 torch.result_type 语义在 host 推导
# _MID = {fp16:0, bf16:1, fp32:2} → 每路 MID*_DTYPE constexpr（DE3）
```

#### L2.2 kernel 启动模式（1D 扁平 grid + constexpr 特化）

```python
_vec_linear_silu_kernel[(grid2,)](temb, w1, b1, h, B, NUM_M,
    temb.stride(0), w1.stride(0), w1.stride(1), h.stride(0),
    K_C=D_emb, BN_C=bn2, BK_C=128, BM_C=bm, SILU=True,
    MID_ROUND=False, MID_DTYPE=2, multibuffer=True, A_IS_F32=True)
```

### §2.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `_timestep_embed_kernel`（F1 嵌入段：broadcast 外积 + 三角函数 + concat 单 kernel）

```python
@triton.jit
def _timestep_embed_kernel(ts_ptr, f_ptr, out_ptr, B, s_ts, s_f,
                           HALF_C: tl.constexpr, BLK: tl.constexpr, NUM_PROG: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLK)          # BLK = next_pow2(HALF_C)
    fmask = offs < HALF_C
    f = tl.load(f_ptr + offs * s_f, mask=fmask, other=0.0).to(tl.float32)
    for b in range(pid, B, NUM_PROG):  # 核内步长循环，每 task 一行
        t = tl.load(ts_ptr + b * s_ts).to(tl.float32) * 1000.0   # scale 来自算子定义
        arg = t * f                    # fp32 契约
        tl.store(out_ptr + b * (2*HALF_C) + offs, tl.sin(arg), mask=fmask)
        tl.store(out_ptr + b * (2*HALF_C) + HALF_C + offs, tl.cos(arg), mask=fmask)
```

#### L3.2 `_vec_linear_silu_kernel`（DE1 向量外积 GEMM + bias + 激活 + 可选中间舍入）

```python
@triton.jit
def _vec_linear_silu_kernel(a_ptr, w_ptr, bias_ptr, o_ptr, M, NUM_M,
                            s_a, s_wo, s_wi, s_o,
                            K_C: tl.constexpr, BN_C: tl.constexpr, BK_C: tl.constexpr,
                            BM_C: tl.constexpr, SILU: tl.constexpr,
                            MID_ROUND: tl.constexpr, MID_DTYPE: tl.constexpr,
                            A_IS_F32: tl.constexpr):
    pid = tl.program_id(0)
    pid_n = pid // NUM_M
    pid_m = pid - pid_n * NUM_M
    offs_n = pid_n * BN_C + tl.arange(0, BN_C)
    offs_k = tl.arange(0, BK_C)
    offs_m = pid_m * BM_C + tl.arange(0, BM_C)
    m_mask = offs_m[:, None] < M
    acc = tl.zeros((BM_C, BN_C), dtype=tl.float32)
    wbase = w_ptr + offs_k[:, None] * s_wi + offs_n[None, :] * s_wo   # Wᵀ: 交换 stride
    for k in range(0, K_C, BK_C):
        a_raw = tl.load(a_ptr + offs_m[:, None]*s_a + (k+offs_k)[None, :], mask=m_mask, other=0.0)
        a = a_raw if A_IS_F32 else a_raw.to(tl.float32)               # DE6 剪除 no-op cast
        w = tl.load(wbase + k * s_wi).to(tl.float32)
        acc += tl.sum(a[:, :, None] * w[None, :, :], axis=1)          # DE1 向量外积
    acc += tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
    if SILU:
        acc = acc * tl.sigmoid(acc)
    if MID_ROUND:                                                     # DE3 中间舍入复现
        if MID_DTYPE == 0:   acc = acc.to(tl.float16).to(tl.float32)
        elif MID_DTYPE == 1: acc = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + offs_m[:, None]*s_o + offs_n[None, :], acc, mask=m_mask)
```

#### L3.3 `_vec_dual_linear_add_cast_kernel`（DE4 pass-merge：单 K-loop 双累加器）

```python
@triton.jit
def _vec_dual_linear_add_cast_kernel(a1_ptr, w1_ptr, bias1_ptr,   # 路 A（MLP 尾层）
                                     a2_ptr, w2_ptr, bias2_ptr,   # 路 B（并行投影）
                                     o_ptr, M, NUM_M,
                                     s_a1, s_a2, s_w1o, s_w1i, s_w2o, s_w2i, s_o,
                                     K_C: tl.constexpr, BN_C: tl.constexpr, BK_C: tl.constexpr,
                                     BM_C: tl.constexpr, OUT_F32: tl.constexpr,
                                     MID2: tl.constexpr, MIDT: tl.constexpr,
                                     MID2_DTYPE: tl.constexpr, MIDT_DTYPE: tl.constexpr):
    # 同 L3.2 的 pid 分解与 offs 计算…
    acc2 = tl.zeros((BM_C, BN_C), dtype=tl.float32)
    acct = tl.zeros((BM_C, BN_C), dtype=tl.float32)
    for k in range(0, K_C, BK_C):            # 一个 K-loop 同时推进两个累加器
        a1 = tl.load(...).to(tl.float32); w1 = tl.load(...).to(tl.float32)
        acc2 += tl.sum(a1[:, :, None] * w1[None, :, :], axis=1)
        a2 = tl.load(...).to(tl.float32); w2 = tl.load(...).to(tl.float32)
        acct += tl.sum(a2[:, :, None] * w2[None, :, :], axis=1)
    acc2 += bias1[None, :]; acct += bias2[None, :]
    # 各自 round 到自己中间 dtype 之后才 add（DE3 位级契约）
    if MID2:  acc2 = round_through(acc2, MID2_DTYPE)
    if MIDT:  acct = round_through(acct, MIDT_DTYPE)
    acc = acc2 + acct
    tl.store(..., acc if OUT_F32 else acc.to(o_ptr.dtype.element_ty), mask=m_mask)
```

### §2.4 性能基准（本例实测，round-4 champion 按 B 分段）

| B 段 | cases | 加速比 geomean | 区间 | 备注 |
|------|-------|---------------|------|------|
| 1-12 | 26 | **1.953x** | [1.86, 2.10] | 融合压 launch 的主收益段 |
| 13-31 | 9 | 1.284x | [1.08, 1.49] | bm16 消 masked-lane |
| 32-95 | 10 | 1.139x | [0.99, 1.44] | bm32 段 |
| 96-127 | 3 | 1.067x | [1.00, 1.18] | BN256 gate 生效段 |
| 128-255 | 5 | 1.217x | [1.04, 1.57] | bm64 gate 生效段 |
| 256+ | 7 | **0.730x** | [0.62, 0.86] | 回归段：vector 外积 < aclnn Cube |
| 全量 | 60 | **1.3939x** | — | target 5.0 未达，平台期 |

**关键结论**（对本类的启示，非本例专属）:
1. 收益全部来自**结构**（段级融合 + pass-merge + tier law）；tile 族参数微调全线证伪。
2. 最大 B 段可能结构性回归（向量外积吞吐 < aclnn Cube 路径 + tf32 降级）；若新算子的 B 上界更大，突破需另寻非 tl.dot 的 Cube 路径，不要在 tile 上消耗 round。
3. 迭代史（7 轮）：R1 融合框架 1.2063 → R3 pass-merge 1.3579 → R4 tier law 1.3939 → R6 边界 32→17 **1.4083**（新 champion，两次 run 认证）；R5/R7 为 recipe 复推 + 结构 probe（嵌入段融合进 MLP 段被证伪回退，见 DE2）——**最后一个有效 lever 往往是 tier 边界而非新结构**。
4. 同代码不同轮 geomean 差 ~±0.2 点属 profiler 方差（DE10），平台期判定前先排除。
5. 1.39-1.41x 是本例 recipe 在 910B1 的实际天花板；新算子的天花板由其权重流量下限决定，勿直接沿用该数值作预期。
