---
type: Guide
title: CoTAttention 算子（cotattention） 实现手册
description: '典型特征: 空间-通道双 attention——Coordinate Attention：grouped conv key embed
  产空间注意权重，value_embed conv1x1 产通道权重，经 2 次轻量 attention（att1/att2）+ bias/mean 汇聚。'
tags: [task.implementation, topic.attention, topic.compute.neighborhood.convolution, topic.compute.normalization, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:27:14Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# CoTAttention 算子（cotattention） 实现手册

> 类别约束见 [../cv_attn_agg_constraints.md](../constraints/cv_attn_agg_constraints.md)，
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §4.1 CoTAttention 算子（cotattention）

**算子类别**: `cv_attn_agg`　**典型特征**: 空间-通道双 attention——Coordinate Attention：grouped conv key embed → 空间维与通道维各做一次加权聚合（att1/att2）+ bias/mean 汇聚（§0.2-(b)）。
**性能基准**: 主线 49/49 精度通过，几何平均 **1.5147x**（NPU15，2026-08-19，fusion-search 7 轮，框架 0.222ms / impl 最差 case 0.289ms）；并行纯 Triton 线 **1.1417x**（NPU5，2026-08-25，无 torch 退化）

### §4.1.1 Layer 1 设计约束（算子专属）

本算子逐条适用 [../cv_attn_agg_constraints.md](../constraints/cv_attn_agg_constraints.md) 的 C1~C8，并追加并行线专属约束：

- **C1 ★★ RNE 复刻 + fp16 域加法强制逐 conv 舍入**（并行线 R3）：单 fp32 累加 + 最终一次 fp16 舍入 FAILS MERE（3.7e-2 vs 阈 9.8e-4）；k1、v 各自舍入 fp16 后 **fp16 域相加**（`k1_16+v_16`，Triton 中 fp16+fp16 保持 fp16 不提升重融合）→ MERE=0.0。
- **约束禁 torch 退化时 CANN 路由不可用**（AST Type3 拦截）——小 case 用**融合 kernel 减 launch**，不是路由 torch（并行线 R2）。
- **n==1 特判仅当 hw==1 数学精确**：`softmax_over_hw(att)` 当 hw=1 恒等于 1；hw>1 时 softmax 非恒等不可套用（并行线 R1 边界）。
- **tile 可靠性以独立 forward 验证为准**（并行线 R5）：大 tile（BM=512/BK=256 weight block 388KB>192KB）有时 auto-split 编译成功、隔离/全量 forward 又失败——sweep 碰巧编译过不可信，可靠档 BM=128×BK=256。
- **最大整除 M-tile 只在 B 操作数巨大（[c,n]，如 value_embed/att1）时有效**（并行线 R17/R18）：小 K 小操作数 GEMM（att2_gemm）应保持小 BM 保证 grid 并行度（R19，-3.9%/-4.1% 证伪）。
- **simulator SCALAR 信号在小 shape 不转移真实大 shape**——采集后落地前先估收益量级（并行线 r16 教训）。

### §4.1.2 Layer 2 算法骨架

```
主线（T1~T4，CANN conv 路由 + Triton attention 融合链）：
  grouped_conv（k1 分组卷积）   → CANN F.conv2d（64 TFLOP/s，最优后端）
  value_embed（conv1x1）        → CANN F.conv2d
  att1（cat-conv）              → CANN F.conv2d（case 38 特例除外，见 T4）
  att2（w2_avg 预归约 kk 维）   → Triton：att_w2 [C*kk,mid]→[C,mid] 预缩放 1/kk（FLOPs 少 9x）
  softmax_fused（单 kernel online softmax + residual）→ Triton
  小 case 整算子路由 torch（T2）：Triton 6-kernel 启动 ~0.1ms/kernel 不可摊销

并行纯 Triton 线（r11→r15，与主线并列记录，不做覆盖）：
  n1_fused_kernel（n==1 退化整段跳过 + 3 launch→1 launch 融合）
  att_mean / softmax（Triton ieee GEMM）
  w2sum 用 compute-dtype（fp32→dtype，att2_gemm 每 K 迭代 .to cast 变 no-op）
```

### §4.1.3 Layer 3 关键技巧（可参考不可照抄）

- **T1 卷积路由 CANN**（最大杠杆，0.69→1.51x）：k1 分组卷积（Triton fused_singlerow 4.4 vs CANN 64 TFLOP/s）、value_embed conv1x1（8.99→2.87ms）、att1 cat-conv（3.9-7.9→1.3-3.2ms）全部换 `F.conv2d`。参考本身就是 torch conv，CANN 是其最优后端。
- **T2 小 case 整算子路由 torch**：判别规则 `n==1 ∨ b*c*n<1.2M ∨ (bf16∧b*c*n<6M)`，不误伤 fp32 大 case（case 9/10 保留 1.6x）。
- **T3 Triton attention 融合链**：w2_avg 预归约 kk 维 + softmax_fused 单 kernel online softmax + residual。
- **T4 case 38 特例**：bf16 640×223×191 的 att1 用 CANN 反而慢（4.73 vs Triton 3.98ms）→ 保留 Triton att1。
- **并行线 R1 n==1 退化整段跳过**（case47 0.088→1.2266x，最大杠杆）：`out=k1+v` 数学上**精确**（实测 max_abs_diff=0.0 / MERE=0.0），k1（grouped conv3x3）与 v（conv1x1）降为 **batch 作 dot-N 的 batched matvec**（权重 `[C,CIN_PG]`/`[C,C]` × 输入 `[CIN_PG,B]`）。
- **并行线 R2 退化 case 必须融合单 kernel**（3 launch→1 launch）：launch overhead ~0.1ms/kernel，3 分离 kernel（k1+value+add）=490us（sp 0.80x 反亏）；融合 `n1_fused_kernel` 单 launch=330us（sp 1.20x）。
- **并行线 R4 w2sum fp32→compute-dtype**：`w2sum_buf` 改 `dtype`，att2_gemm 每 K 迭代的 `.to` cast 变 no-op。
- **R6 grid 元组陷阱**：`_grid_for_tiles` 已返回 tuple，`kernel[(grid,)]` 再包层 → "tuple object cannot be interpreted as integer"；早期 sweep 的"大 tile 快"数字是此 bug + 机器噪声的伪影。

### §4.1.4 性能基准 / 优化记录表

| 方案 | 几何平均 | 说明 |
|---|---|---|
| **主线：CANN conv 路由 + Triton attention 融合链** | **1.5147x** | 49/49，NPU15，2026-08-19，fusion-search 7 轮 |
| **并行线：纯 Triton 约束线** | **1.1775x** | 49/49，NPU5，2026-08-25，`verify_r11/perf_result_r17.json`（R17+R18，att2_gemm 已回退）；r16 补 msprof op simulator 全 kernel 采集（覆盖门禁），R17/R18 落地「非 pow2 最大整除 M-tile」后 1.1417 → **1.1775x**。轨迹 r12d=1.0556 → r13 w2sum compute-dtype → r14 n==1 融合 fast path → r15=1.1417x → **r16 采集 → R17/R18 → 1.1775x**。case47（`[8,2048,1,1]` 退化 n==1）0.088→**1.2266x**，case46（`[1,1,1,1]`）2.541x |

主线剩余瓶颈：Triton ieee GEMM（att_mean/softmax）算力上限；CANN matmul 实测不更快（已验证）。2.2x 目标因 CANN conv 已极优而不可达。

#### 并行线续：simulator 诊断 + 最大整除 M-tile（r16→r18b，2026-08-25，NPU5）— 已验证

> r16 补 msprof op simulator 全 kernel 采集（覆盖门禁），R17-R18 落地「非 pow2 最大整除 M-tile」，R19 证伪其误用并回退。**1.1417x → 1.1775x**（新纪录）。磁盘代码 = R17+R18（att2_gemm 已回退）实测 R18b=1.1753x，与 R17 同噪声带。

**r16 simulator 全 kernel 采集（3 瓶颈 kernel 均非 MMAD bound）**：

- `grouped_conv_implicit_kernel`：Cube MTE2 40.9% / SCALAR 27.2% / CUBE 9.6% → **MTE2 访存 staging bound**。ghost-pad 隐式 GEMM 对 kk=9 tap 各做一次 x tile 的 L1 ND2NZ 转换 staging（`MOV_OUT_TO_L1_MULTI_ND2NZ` ~8200 cyc×2 ×9）每次伴 SET_FLAG/WAIT_FLAG 同步；Cube 为关键路径（65887 cyc ≫ Vector 19026）。
- `att1_kernel`：Cube SCALAR 55.6% / CUBE 9.9% → **SCALAR 标量降级 bound**（dual-dot 每轮 4×block_ptr load 地址算术标量化）。小 shape 信号，真实大 shape 不转移（见 §4.1.1 约束 6）。
- `value_embed_kernel`：Cube MTE2 25.2% / SCALAR 25.0% / CUBE 10.8% → **均衡混合**（最接近 compute-balanced，优先级最低）。
- **证伪**：「fp32 dot 不走 Cube / dot 是硬件瓶颈」——CUBE 实测仅 9.6-10.8%，dot 极便宜，开销在 dot 周围的**访存 staging + 标量地址算术**。

**R17 att1 非 pow2 最大整除 M-tile（1.1417→1.1775x，+3.1%，本线最大杠杆）**：机理 = att1 dual-dot（wk1@k1 + wx@x）的 m_block 扫描每次重读 `[c,n]` k1/x 操作数。mid%128!=0 的 19/48 shape 原 BM=8/16/32/64（pow2 规则），M-block 扫描 3-7 次；改用 mid 的**最大整除 BM**（mid=112→BM=112、mid=224→BM=112），扫描降到 1-4 次。成本代理 `cost = m_blocks + 2.0*mid/bn`，遍历 (BN,BK)∈(128,64,32)×(128) 选最小档；UB 预算（fp16 dual-dot）4·BM·BN + 4·BM·BK + 4·BK·BN ≤ 196KB → BN=128 时 BM cap≈122。受益 shape 集中 mid=112/224（case35 +35% / case20 +27% / case16 +20% / case17 +18% / case32 +16%）。

**R18 value_embed M-tile（保留，case35 叠加 +5.5%）**：同规则扩展到 value_embed 的 c%128!=0（case2 c=96 → BM=96 mblk 3→1；case35 c=448 → BM=224/BN=128 mblk 7→2）。geomean 持平（1.1772），但 case35 净叠加 +5.5%，保留。

**R19 att2_gemm M-tile（已回退，关键负样本）**：同规则误用于 att2_gemm 反而有害（case2 -3.9% / case35 -4.1%）。根因：att2_gemm 两个操作数都小（A=[c,mid] mid=24/112、B=[mid,n]），加大 BM 省不了流量却减并行程序数。

### §4.1.5 陷阱 / 证伪方向

| 方向 | 结果 | 来源 |
|------|------|------|
| 纯 Triton fused 卷积 | 4.4 vs CANN 64 TFLOP/s，慢 | T1 证伪 |
| CANN att_mean matmul | 实测不更快 | 主线证伪 |
| n1 分离 3 kernel（k1+value+add） | 490us sp 0.80x 反亏（launch 不可摊销） | R2 |
| 单 fp32 累加 + 一次 fp16 舍入 | MERE=3.7e-2 超阈（逐 conv 舍入未复刻） | R3 |
| 大 tile BM=512/BK=256 | UB overflow 388KB>192KB + 编译非确定 | R5 |
| 未采集就断言「dot 是硬件瓶颈 / fp32 dot 不走 Cube」 | **证伪**（CUBE 实测 9.6-10.8%，开销在 dot 周围访存 staging + 标量算术） | 并行线 r16 |
| 最大整除 M-tile 扩展到小操作数 GEMM（att2_gemm） | **-3.9%/-4.1%**（大 BM 省不了流量却减 grid 并行；M-tile 只在 B 操作数巨大 [c,n] 时有效） | 并行线 R19 |

泛化启示：softmax-over-hw 类算子遇空间维=1 的退化 shape，可整段跳过 attention（数学精确无损）；纯 Triton 约束下超小 case 的救法是**融合 kernel 减 launch 数**，不是路由 torch；多 kernel 算子先过「全 kernel 采集覆盖门禁」再定瓶颈（三个瓶颈 kernel 全部非 MMAD bound，白省「换 bf16 / 加大 tile」臆测）；非 pow2 整除 M 维的 GEMM，把 pow2 BM 换成最大整除 BM 是通用低成本杠杆，但只对 B 操作数巨大的形态有效——先看操作数尺寸再套规则。
