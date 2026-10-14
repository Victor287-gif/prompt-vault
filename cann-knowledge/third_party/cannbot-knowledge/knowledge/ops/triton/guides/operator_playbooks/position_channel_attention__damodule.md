---
type: Guide
title: DAModule（DANet 双重注意力模块） 实现手册
description: '典型特征: 无投影位置/通道双分支——Q=K=V 同一 Y（conv 输出），position 分支 (B,N,C) 与 channel
  分支 (B,C,N) 各做一次 softmax(Y·Yᵀ)·Y 三段式，out=out_pos+out_ch 残差相加；4 kernel + host 门控双路径分派，Phase
  3 一轮 2.4313x、50/50。'
tags: [task.implementation, topic.attention, topic.compute.neighborhood.convolution, topic.compute.normalization, topic.precision, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-31T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# DAModule（DANet 双重注意力模块） 实现手册

> 类别约束见 [position_channel_attention_constraints.md](../constraints/position_channel_attention_constraints.md)（L1.1~L1.8、S1~S3、§5 优化点/证伪表），
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 Layer 2: 算法骨架（4 kernel + host 门控）

```
host_prepare():                              # cache miss 时一次，warmup 吸收
    RNG 逐字复刻两分支 conv 权重（L1.4）
    预重排 w9[k²,ci,co] 连续（L1.3）；cache[key] = (w9_pos, b_pos, w9_ch, b_ch)

kernel 1  conv3x3（implicit GEMM，TRANS_OUT: constexpr）:
    acc[BM(n), BN(co)] fp32
    for ci_block in range(0, C, BC):
        for k in range(k²):                  # k² 次移位 dot，全部 Cube
            halo = (0≤p+dh<H) & (0≤q+dw<W)   # padding=k//2 零填充语义
            xt = load(x[ci_block, n+dh·W+dw], mask=ci_ok & halo, other=0)   # [BC,BM]
            wt = load(w9[k, ci_block, co_block])                            # [BC,BN] 连续
            acc += dot(trans(xt), wt)
    y = cast(acc + bias, dtype_in)           # conv 段舍回（L1.7）
    TRANS_OUT=true  → store (B,N,C)；false → store (B,C,N)
    两次 launch：position 分支(true) → y_pos；channel 分支(false) → y_ch

kernel 2  路径 A online softmax（D≤256 or T>128）:
    q_tile[BQ,BD] 常驻；m=-1e30, l=0, acc=0
    for kv in range(0, T, BKV):              # kv_lo=0 不对齐（L1.8）
        s = dot(q, trans(k)) * scale         # 原生操作数 + fp32 累加
        s = where(kv_ok, s, -1e30)           # F1：ghost 列显式排除
        m_new = maximum(m, rowmax(s), propagate_nan=ALL if MULTI_BLK)
        p = exp(s - m_new)；alpha = exp(m - m_new)
        acc = acc*alpha + （p 二段拆分 dot | fp32 原生 dot）（L1.2）
        l = l*alpha + sum(p)；m = m_new
    o = cast(acc / l, dtype_in)
    if ADD_OUT: o += load(out 同位置)        # dtype_in add（L1.7）
    统一 (b,c,n) stride store（L1.1）

kernel 3+4  路径 B P 物化两阶段（D>256 且 T≤128）:
    3a attn_smax:  s = Σ_d dot(a[:,d], trans(a[:,d]))   # Q=K=V 共享同一 load（§4.4）
                   单趟 softmax（rowmax→exp→sum→div，无 online 状态）→ p_buf[B,T,T] fp32
    3b attn_pv:    o = dot(p, v)（p 二段拆分，L1.2）沿 D 分块并行补并行度
                   → cast dtype_in → ADD_OUT → store

host 门控（每分支独立判定，L1.6）:
    for (y, T, D, scale, trans, add) in [(y_pos, N, C, C**-0.5, true,  false),
                                          (y_ch,  C, N, N**-0.5, false, true )]:
        D<=256 or T>128 → 路径 A 一次 launch；否则 → 路径 B 两次 launch
    return out.view(b, c, h, w)
```

### §3.2 双路径判据的出处

路径 A 的 UB 瓶颈在 `acc[BQ, BD]`：D>256 时 96KB 预算装不下有意义的 BQ。但实测 shape 空间里
**T·D 此消彼长**——D>256 只在 T≤128 时出现（position 分支 D=C 大则 N 小；channel 分支 D=N 大则 C 小），
此时 `T×T ≤ 128²` 的 P 可物化（fp32 ≤64KB），把 online softmax 换成"S 全量 + 单趟 softmax + PV GEMM"：
S 内积维分块累加（D 可达数千），阶段 b 再沿 D 分块补并行度。**先查 (T,D) 组合再定路径，禁止对 D>256 硬走 online。**

### §3.3 launch 计数与小 case 画像

conv 每分支 1 次 + 路径 A 每分支 1 次（路径 B 2 次）⇒ 全算子 **4~6 次 launch**。
小 case（B=1, N 仅十几）launch 串行占比高，是本类小 case speedup<1 的结构原因之一，与 FA 小 case
画像一致；**不接受** host 侧合并两分支 conv（权重不同），小 case 回退属结构固有，写入报告画像即可。

## §4 Layer 3: 关键技巧

### §4.1 ★★★ ADD_OUT kernel 内融合残差相加（省一路 kernel + 一遍 out 流量）

```python
if ADD_OUT:
    prev = tl.load(out_ptr + base, mask=msk, other=0.0)
    ov = ov + prev          # dtype_in add，对齐参考 out_pos + out_ch 的语义
tl.store(out_ptr + base, ov, mask=msk)
```

position 分支先写（ADD_OUT=false），channel 分支 kernel 内加（ADD_OUT=true）。
**前提是 L1.1 布局统一**——两分支 stride 算出同一批地址，融合才成立。
顺带把"两分支在 dtype_in 上相加"的契约（L1.7）天然落实。

### §4.2 ★★★ p hi+lo 二段拆分（见约束卡 L1.2 代码块）

p∈[0,1] 先 `p_hi = p.to(dtype)`，再 `p_lo = (p − p_hi.float()).to(dtype)`，两次原生 dtype dot 共享
fp32 累加器。仅非 fp32 输入需要（fp32 case 参考本来就 fp32 算）。⚠️ 与 FA 卡 L1.5"禁降"不冲突：
拆分不是降精度，是用两次原生 dot 还原 fp32 乘积。**不要复制变量名/结构**——按语义重新实现。

### §4.3 ★★ k×k conv implicit GEMM 配方（k² 移位 dot + halo mask）

```python
p_r = offs_n // W;  q_c = offs_n % W            # token n 的 (行,列)
for k in range(KK):                             # k×k = k² 次移位 dot
    dh = k // KS - KS // 2;  dw = k % KS - KS // 2
    halo = (p_r+dh >= 0) & (p_r+dh < H) & (q_c+dw >= 0) & (q_c+dw < W)
    xt = tl.load(x_ptr + bi*C*N + offs_ci[:,None]*N + (offs_n+dh*W+dw)[None,:],
                 mask=ci_ok[:,None] & (n_ok & halo)[None,:], other=0.0)   # [BC,BM]
    wt = tl.load(w9_ptr + k*C*C + offs_ci[:,None]*C + offs_co[None,:],
                 mask=ci_ok[:,None] & co_ok[None,:], other=0.0)           # [BC,BN] 连续
    acc += tl.dot(tl.trans(xt), wt)
```

3×3 conv 的 tile 实测最优 **BM=64(n) / BN=128(co) / BC=64(ci)**——两个方向的收缩都已证伪。
halo 用静态索引 + 边界 where（不动态 gather），other=0 即 padding 零填充语义。

### §4.4 ★ 路径 B 的 Q=K=V 共享 load

无投影自注意力的访存红利：阶段 a 同一 `a` tile 自乘 `tl.dot(a, tl.trans(a))`，K/V 零额外访存；
内积维分块循环累加（D 可达数千，BDC=128/64 分档）。

### §4.5 scale host 预计算

`C**-0.5` / `N**-0.5` 在 host 算成 fp32 标量传入，kernel 内只做一次 s*scale。

## §5 证据基础（DAModule，2026-08，910B2 24 CubeCore）

- 覆盖：B∈[1,4]、C∈[8,1024]、H·W∈[16,5041]、fp16 为主 + 少量 fp32/bf16。
- **Phase 3 一轮通过**（轮内修 3 个 A 类：PV 异构 dot / UB 面积 / 双分支布局），首版即 geomean **2.4313**、
  精度 **50/50**、target 2.0 达标；Phase 4 三轮优化（tile 分档 / conv tile 收缩 / launch 常量 constexpr 化）
  **全部实测无提升**，IR 无新建议，终局以 Phase 3 基线收尾。同码跨日复测 50/50，偏差 −1.1% 在波动带内。
- 本类**没有投影段 GEMM 可吃**，收益全部在结构（双分支统一 kernel + ADD_OUT + 双路径分派）；tile 级微调已全部证伪（约束卡 §5.2）。
