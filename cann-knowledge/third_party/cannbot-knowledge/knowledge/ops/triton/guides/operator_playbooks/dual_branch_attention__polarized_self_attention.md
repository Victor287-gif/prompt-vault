---
type: Guide
title: Polarized Self-Attention 算子（polarized_self_attention） 实现手册
description: '典型特征: PSA 通道 softmax-over-N 支与空间 GAP/softmax-over-C2 支门控特征图；并行加性 op57
  out=x·(S+T) / 串行乘性 op61 out=x·T·S，7-kernel GEMV 链。'
tags: [task.implementation, topic.attention, topic.compute.neighborhood.convolution, topic.compute.reduction, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:23:23Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Polarized Self-Attention 算子（polarized_self_attention） 实现手册

> 类别约束见 [../dual_branch_attention_constraints.md](../constraints/dual_branch_attention_constraints.md)，
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §1 Polarized Self-Attention 算子（polarized_self_attention）

**算子类别**: `dual_branch_attention`　**典型特征**: PSA 通道 softmax-over-N 支与空间 GAP/softmax-over-C2 支门控特征图；并行加性 op57 `out = x·(S+T)` / 串行乘性 op61 `out = x·T·S`，7-kernel GEMV 链。　**性能基准**: 各 50/50 verify，几何平均 **2.3237**（op57）/ **2.4783**（op61）（ascend910_93）

### Layer 1 设计约束（算子专属）

全部继承类别卡 L1.1~L1.12（见 [../dual_branch_attention_constraints.md](../constraints/dual_branch_attention_constraints.md)）。本算子特判：

- **L1.2 → 全 vector 路径**：双支全为 M=1 / 深度 <16 的归约，**禁 `tl.dot`**；
- 小 C 双路径融合 5-kernel（C≤128，§3.2）；F7 舍入点契约：op57 两处 / op61 四处。

### Layer 2 算法骨架（7-kernel GEMV 链）

**并行加性版（op57，out = x·(S+T)）**，数学重排与 kernel 划分：

```text
通道支: q_c[b,n] = W_wq·x[:,n]+b_q → a = softmax_N(q_c) → xtil[c] = Σ_n a·x
        → z_c = W_wv·xtil → y_c = W_wz·z_c → T[c] = sigmoid(LN_C(y_c))
空间支: xbar[c] = mean_N(x) → q_s = W_sq·xbar → a_s = softmax_C2(q_s)
        → whead[c] = Σ_i a_s·W_sv[i,c] → S[n] = sigmoid(whead·x[:,n] + ζ)
K1 流过 x（q_c GEMV + xsum partials）→ K2 softmax_N + xbar 归并 → K3 q_s
→ K4 a_s+ζ+whead+xtil（同 grid 三合一）→ K5 z_c → K6 y_c → K7 终融合（LN 冗余 + S + out）
```

**串行乘性版（op61，out = x·T·S）差异**：

1. 空间支吃 channel_out：`g0 = T ⊙ xmean`——**xmean 与通道支无依赖，前置到 K2 并行算**，消一处串行等待；
2. T 折进空间支权重：`w2[c] = whei[c]·T[c]`，K7 的 S 归约只做一遍 GEMV（`S = sigmoid(w2·x + ζb)`）；
3. 依赖链 K1→K2→K3→K4→K5→K6→K7 **严格串行**（6 个全局同步点）；
4. 双路径特化：C≤128 时 K3+K4、K5+K6 各融合 → 5-kernel（§3.2）；
5. Split-K：N≥2048 且 B·GC≤16 时 K2 拆三段（§3.1）。

### Layer 3 关键技巧（可参考不可照抄）

#### §3.1 ★ Split-K 拆三段（op61 K2，+5.68%）

```python
# host 门控：任务少、单任务长才拆（launch 串行主导的小 case 不拆）
split_k = (N >= 2048) and (B * GC <= 16)
nseg = min(next_pow2(cdiv(N * C, B * GC)), cdiv(N, 256))
# K2a softmax_a  grid=(B,)            a 原地覆写 qc
# K2b pool_part  grid=(B*GC*nseg,)    partial 写 [t*CB] 布局，段间无竞争
# K2c pool_merge grid=(B*GC,)         [NSEGP, CB] tl.sum(axis=0) 树形归并
```

- **门控**：`N ≥ 2048 且 B·GC ≤ 16`（10/50 case 命中；B·GC=15 的 2 个 case 收益 < 3 kernel launch 开销，
  门控应收紧到 ≤12——门控阈值本身要按实测修剪）。
- 跨段树形归并 vs 顺序累加的 fp32 差异由下游 u 的 dtype 舍入吸收（精度无损）。
- 命中 case 8/10 提升（最高 **-51.3%** 时延）。

#### §3.2 双路径融合 5-kernel（op57 +2.6%，op61 +0.41%）

```python
# host 双路径分派：小 C 用 5-kernel（省 2 次 launch），大 C 保持 7-kernel
if C <= 128:
    _k14_fused[grid](...)   # K3 并入 K4：任务内冗余算 qs 全量 GEMV
    _k56_fused[grid](...)   # K5/K6 合并：冗余算 zc 全量
else:
    _k3[grid](...); _k4[grid](...); _k5[grid](...); _k6[grid](...)
_k7[grid](...)
```

- **判据**：逐 kernel Event 计时显示小 case 端到端 ≈ launch 串行主导
  （op57 `[1,640]` 79.5µs ≈ 7×11µs）；冗余 GEMV 开销可忽略当 `C2×C ≤ 64×128` MAC。
- 大 C 走原 7-kernel 路径不受影响——双路径 host 分派，不是全局改参数。

#### §3.3 任务循环 + constexpr 档位化（op57 +20.7%）

- `grid = min(NTASK, NUM_CORES=40)` + 核内 `for t in range(pid, NTASK, NPROG)`（类别卡 F3）；
- 小 C case 的 grid（582/1158）远超 40 vector cores，program 发射开销累积主导
  （同流量 `[6,32,61,101]` 233µs vs `[1,640,21,69]` 89µs，2.6 倍）。
- 档位扩展（C≤256/CP=256）证伪：~持平（2.3181 < best 放弃），不外推。

### 性能基准 / 优化记录表

| 指标 | op57 ParallelPSA | op61 SequentialPSA |
|------|------------------|--------------------|
| 架构 / NPU | ascend910_93 / NPU13 | ascend910_93 / NPU2 |
| geomean | **2.3237** | **2.4783** |
| 关键路径 | 任务循环+档位化 +20.7% → 双路径融合 +2.6% | K2 Split-K +5.68%；IR/simulator 终局干净 |
| verify | **50/50** | **50/50** |
| 轮次 | Phase 3×2 + Phase 4×6 | Phase 3×2 + Phase 4×5 |

### 陷阱 / 证伪方向

- **pass-merge（K4 顺带产出 zs_part，x 遍数 4→3）** **-33%**：双方向归约破坏循环流水/寄存器分配——
  pass-merge 必须与原归约**同方向**。
- **softmax_a 并入 pool_part**：**+61% GM 读 vs 省 1 launch**，流量核算证伪——融合前先算流量账。
- g-major 任务重排 -0.25%：噪声级放弃。
- **小 case 的 launch 墙**：每 kernel ≈ 11-13µs，7-kernel 小 case 下限 ≈ 80µs——省 1 个 kernel 净赚 ~12µs
  （双路径融合的全部动机）；大 C 是带宽墙（op57 `[2,1280]` 达 ~205GB/s，x 三遍流是算法下限）。
  **两墙修法互斥**：先定位在哪堵。
- 精度：op57 仅 load/store 两处降精度（fp16 atol=9e-2 宽松）；op61 四处 dtype 舍入点
  （`u`/`zc`/`T_d`/`out`）需匹配参考中间 dtype（类别卡 F7/L1.5）。
