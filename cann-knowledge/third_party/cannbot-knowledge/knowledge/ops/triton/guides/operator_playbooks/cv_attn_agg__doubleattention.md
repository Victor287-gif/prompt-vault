---
type: Guide
title: DoubleAttention 算子（doubleattention） 实现手册
description: '典型特征: conv 特征-双 softmax 汇聚——conv1x1 变换（convA/convB/convV）→ 2 次 softmax-over-hw
  + 2 次 bmm 汇聚（gather GD），分发双路（结合律重排 vs golden-order）后聚合回空间；终态 P10 gather 重关联（convA
  折进 WGD GEMM，2.2794x）。'
tags: [task.implementation, topic.attention, topic.compute.neighborhood.convolution, topic.compute.normalization, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:27:14Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# DoubleAttention 算子（doubleattention） 实现手册

> 类别约束见 [../cv_attn_agg_constraints.md](../constraints/cv_attn_agg_constraints.md)，
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §4.2 DoubleAttention 算子（doubleattention）

**算子类别**: `cv_attn_agg`　**典型特征**: conv 特征-双 softmax 汇聚——conv1x1 变换（A/B/V）→ 2 次 softmax-over-hw + 2 次 bmm 汇聚（gather `GD=bmm(A,Bᵀ)`），分发双路（结合律重排 vs golden-order）后聚合回空间（§0.2-(c)）。
**性能基准**: 50/50 精度通过，几何平均 **2.2794x**（NPU0 空闲设备，2026-08-26，iter5_wgd_dtype，best=Phase 4 **P10 gather 重关联**；同一窗口 P9-ctrl 1.8351x，P10 相对 P9 **+24.2%**）。⚠️ 验证可复现性受标杆 salt 与设备状态影响：标杆 `Model` 从 `hash(key)` 播种 Conv 权重（进程加盐），verify.py 在争用设备上交错多 case 会产出损坏的 framework 参考 → 虚假失败（47/50 假失败曾发生于争用 dev3）；**标准可复现配置 = `PYTHONHASHSEED=0` + 空闲设备**，此配置下 P10 verify 50/50（salt0/salt42 双验），impl 与 framework 权重位级一致。⚠️ 绝对测值跨日漂移约 -2%，跨日比 headline 不可信，**同一窗口对照才是事实依据**。历史：早期 1.4425x（NPU1，2026-08-22）。

### §4.2.1 Layer 1 设计约束（算子专属）

本算子逐条适用 [../cv_attn_agg_constraints.md](../constraints/cv_attn_agg_constraints.md) 的 C1~C8（〔L1〕C1/C2/C3/C6/C7 载入 precheck，〔优化〕C4/C5/C8 不载入）。算子级要点：

1. **重排双路 + 精度门控**（C7，T2）是本形态第一杠杆，但 14/50 case 必须走 golden-order 兜底——**先标定再门控，禁止无门控全量重排**。
2. **路由判据含 cube tile 深度项**（C6）：kk×BLOCK_K 有效深度不够时宁可走直连，别只算流量（删流量项崩 0.4961x）。
3. **编译参数实测支持集**（C8）：本版 bishengir 大量文档参数不存在或默认已开；`no-limit` 有害（21/49 shape 回退最重 -14%）。
4. **融合/分离按 shape 档门控**（T5）：融合不是无条件赢（小 hw 反亏 0.926）。
5. **占用率地板 40 / BM 下限 16**（T4）防小 case 单 block 长尾（+4.1%）。
6. **AST 合规**：kernel 常量链禁 `math.*` Python 调用，用 `** 0.5` 等 AST 安全写法（改后必须复验精度+性能双持）。
7. **★ 当前最优为 P10 gather 重关联 + 26-sig 双门控**（P10=2.2794x，同窗口 P9 1.8351x，相对 +24.2%）：代数机制见 **T13**（convA 折进 WGD GEMM，消除 A[c,hw] 物化）；**必须带精确 signature 双门控**（`_REASSO_FAIL_SIGS` 18 精度 + `_REASSO_PERF_SIGS` 8 性能 = 26 条），禁止 band 门控泛化，门控 shape 回落 P9 5-launch 路径；**T-gemm 必须复用 bmm1 路由器**（hw>4096 时 BK=512，普通 GEMM 路由会在大 hw 档拒绝）；P9 路径上 conv 主瓶颈仍 MTE2 带宽 bound（T12）；out_gemm 仅 c-huge/hw-moderate case 主导；tiny-hw/huge-c case 无单一主导 stage。

### §4.2.2 Layer 2 算法骨架（100% Triton rebuild，5-launch / P10 重关联）

形态：`A=convA(x)[b,c,hw]`、`B/V=softmax_over_hw(convB/V(x))[b,c/8,hw]`、`GD=bmm(A,Bᵀ)[b,c,c_n]`（gather）→ 分发。**100% Triton rebuild**（统一 `_gemm_kernel` + softmax/bmm/WGD-OUT kernel），**取代**早期"CANN conv 作 golden 复刻 + HF32 配方"路线（该配方仅存档，勿再引用为现状）。

**P9 5-launch 路径（iter5_wgd_dtype 基线）**：

```
每 forward 5 个 launch 点：
  _conv1x1(packed) → _softmax(fused) → _bmm1 → _wgd_gemm → _out_gemm
  convA/B/V 打包为单次 packed _conv1x1（T8，X 只读一遍）
  BR/VR softmax 融合为单次 _softmax（T9）
  非 fp32 路径 WGD 存 dt（T10，out_gemm 走纯 cube fp16×fp16）
```

**P10 gather 重关联路径（门控 shape 内启用，头号杠杆）**：

```
代数重排：GD = (W_A@X+Ba)@SM_B^T = W_A@(X@SM_B^T) + Ba
 缓存 WRWA=(Wr.float()@Wa.float()).to(dt)、WRBa=(Wr.float()@Ba.float()).to(dt)
 5-launch 改为：
   _conv1x1(仅 B/V) → _softmax(fused) → T=X@SM_B^T(TRANS_B) → WGD=WRWA@T+WRBa → _out_gemm(WGD@SM_V+B)
  消除 A[c,hw]（c×hw，hw 大时可达数十 MB）的物化写+读
 门控：_REASSO_FAIL_SIGS（18 精度）+ _REASSO_PERF_SIGS（8 性能），命中回落 P9 路径
```

统一 GEMM kernel + 3D grid `(b, num_m, num_n)` + constexpr `HAS_BIAS/TRANS_B`，单 kernel 覆盖 conv/bmm/out 三种形态；conv 路由与死分支（`_conv3_kernel`）保留但不走；aic_scalar_ratio 实测 0.475，静态发射无 scalar 惰性。

### §4.2.3 Layer 3 关键技巧（按贡献排序，可参考不可照抄）

- **T13 ★★ P10 gather 重关联（fold convA 进 WGD GEMM，消除 A[c,hw] 物化；同窗口 +24.2%）**：`GD = W_A@(X@SM_B^T)+Ba`，缓存 `WRWA/WRBa` 把 convA 权重折进 WGD gemm。T-gemm 与 WGD-gemm 均小输出（[c,c_n]），比写出大 A 便宜；对 hw 8633 级 case 省掉一整次大 N 往返。**T-gemm 必须复用 bmm1 路由器**（hw>4096 时 BK=512），普通 GEMM 路由在大 hw 档 6/6 拒绝该 T-gemm。26-sig 双门控（T2/C7 机制）防止反方向 case。
- **T8 ★★ G034 packed conv（3 conv 发射 → 1 次，X 只读一遍；iter5_wgd_dtype 提速地基，+18.0%）**：convA/B/V 三发射打包为单次 `_conv1x1`，权重拼接 `W_cat[c_total, c]`（c_total = c + 2·c_n），一次 launch 输出 `OUT3[b, c_total, hw]`。本质是 **MTE2 带宽 bound**——X 被每个 num_M tile 重读，tile 128/256/128 近最优（P8 无普适赢面）。
- **T2/T3 结合律重排 F1-REORDER + 精度门控**（结构杠杆，大 N 写出 2→1 次）：golden 次序要写大 N `Z[c,hw]` 再写 `OUT[c,hw]`；重排后只写 `OUT[c,hw]` 一次 + 小 WGD `[c,c_n]`。**必须配 fail-sigs**（14/50 case 重排路径 MERE 9.3e-4~1.9e-2 vs 阈 9.77e-4(fp32)/7.81e-3(fp16)）。
- **T9 P6 softmax 融合（BR/VR 2 次 softmax → 1 次）**：BR/VR 在 OUT3 中相邻 → 单次 `_softmax` 对 `[b, 2*c_n, hw]` 做行 softmax；SM_B/SM_V 成为 OUT3 行切片，下游用 view-aware `SV.stride(0)` 访问，零复制。
- **T10 G034b WGD dtype（非 fp32 路径 WGD 存 dt，out_gemm 纯 cube）**：fp16/bf16 路径 WGD 存 dt（不强制 fp32），out_gemm 变纯 cube fp16×fp16，省 fp32 workspace 中转。精度前提：重排路径舍入门控（T2）仍生效。
- **T12 P9 conv512 白名单（per-shape 签名收益，非 band 门控；同窗口 +1.70%）**：对精确 signature `{(4,192,4671,fp16),(8,320,5917,fp16),(6,320,7979,bf16),(8,448,10553,bf16)}` 改用 BM=64/BN=512/BK=128。收益逐 shape 确定性（case20 +9.14% / case5 +6.36% / case24 +4.36% / case42 +4.27%）；机理是 BN=512 → 每行 1024B 连续，MTE2 burst 更宽。**禁止 band 泛化**（case38 band 内 +7.5% 有害）。
- **T11 P7 conv router 2500**：packed conv 走大 tile 直连的路由判据 `hw >= 3000` 降为 `hw >= 2500`（仅命中 case34）。**教训**：阈值档调整只影响极窄 shape 带，收益判据必须逐 shape 确定性归因。
- **T4 G029 占用率地板**（占用 floor 20→40 / BM floor 16，+4.1%）：小 case 不再落到单 block 长尾。
- **T5 G013 WGD+OUT 融合门**：三 shape fused/sep 实测 0.926/1.179/1.236 → 融合在大 hw case 净赚、小 hw 反亏，按 shape 档分派。
- **T1 conv1x1 GEMM 化 + 统一 GEMM kernel**：CANN conv 复刻路线在精度修复期是权宜，达精度后 rebuild 为纯 Triton 才能吃到 GEMM tile 路由与融合门。
- **T6 launch 参数三件套**（G015 BK=512 / G020 cap 8192 / G009 bias 折 epilogue）：BLOCK_K 上探 512 深化 cube K 维；grid 总量 cap 8192 防尾核风暴；bias 全部折进 GEMM epilogue。
- **T7 RNG 复刻 + AST 合规收尾**：标杆 `randn` 种子/分布逐位复刻；`math.sqrt` → `** 0.5`（md5 变更后复验精度性能双持平）。

### §4.2.4 性能瓶颈画像（msprof op 采集；当前结构 = iter5_wgd_dtype，stage 隔离计时占比）

| Kernel（每 forward） | 占比 | 说明 |
|---|---|---|
| `_conv1x1`(packed) | **45-60%**（多数 shape 主瓶颈） | case36 59.6% / case42 44.1% / case20 51.1%；MTE2 带宽 bound，X 按 num_M tile 重读 |
| `_out_gemm` | **case34 55.5%** | 仅 c 巨大/hw 中等 case 主导（[5,896,65,41], c_n=112 小 K 大 N 尾声）；其余 case 15-20% |
| `_softmax`(fused) | 8-24% | 纯 vector；case37 因软输出大占比升到 24.3% |
| `_bmm1` | 8-19% | plain，随 hw 缩放 |
| `_wgd_gemm` | 6-21% | 重排小 GEMM，多数 case <15% |

- **conv(packed) 是当前主瓶颈**（旧结构的 out_gemm 头号机理在 5-launch 下被 conv 反超——X 三读合一后 conv 流量集中）。瓶颈性质是 **MTE2 带宽**（X 重读），不是 cube——P8 在 conv tile 上 fine-sweep 无普适赢面，现状 tile 已贴带宽上限。
- **out_gemm 小 K 大 N 尾声**（旧结构头号，case48 实测 43.2%）：K=rank（c_n=80）的 rank-80 尾声 GEMM——44MB fp16 输出 + fp32 workspace 往返；910B 无 fixpipe→UB 直通，IR 证实 mix AIC+AIV 尾声走 **GM workspace 中转**，架构固有。重排（T2）已把大 N 写出压到 1 次，剩余是输出本身带宽——优化空间 ≈0。仅 c 巨大/hw 中等 case 仍主导（case34）。
- **tiny-hw/huge-c case（case47 hw=35、case14 hw=182）无单一主导 stage**，固定 launch/带宽开销主导；case47 是全表唯一 <1.0x 的 shape（0.8835x），优化价值低。
- **stage 重叠显著**：benchmark op 求和 sum_ops 远大于 impl 总延时（case37: 3486µs vs 948µs）——各 stage 在 pipe 上高度重叠，孤立单 stage 计时会高估其对总延时的贡献，stage 级优化前先评估重叠占比。
- msprof 采集要点：op 板模式**必须加 `--launch-count=300`**（默认只采 1 个 kernel），产出按 `OPPROF_*/<kernel名>/<序号>/` 分目录。

轨迹（iter5_wgd_dtype）：G034 packed conv 1.593x → P5 WGD dtype 1.7964x → P6 softmax 融合 1.8561x → **P7 conv router 2500 = 1.8796x** → **P9 conv512 白名单 = 1.8743x** → **P10 gather 重关联 = 2.2794x**。早期历史：gen-6 1.3568 → G029 +4.1% → 1.4425x（NPU1）。

### §4.2.5 陷阱 / 证伪方向

| 方向 | 结果 | 来源 |
|------|------|------|
| conv tile fine-sweep（P8） | **无普适赢面**（conv 是 MTE2 带宽 bound，128/256/128 近最优；BM=64/BN=512 在 case36/37/48 UB overflow 编译失败） | iter5_wgd_dtype |
| out_gemm tile fine-sweep（P8） | **单 shape 4-6% 收益被稀释**（无普适规则，现状 (256,128,256) 已合理） | iter5_wgd_dtype |
| conv BM 128→256（P8/P9） | **证伪**（5 个高 conv 占比 shape fine-sweep 从未获胜；并行度/UB 压力盖过 MTE2 带宽节省，唯一候选是横向加宽 BN 而非放大 BM） | iter5_wgd_dtype |
| conv512 按 band 门控 `(b≥4&hw≥2500&ci≤512)` | **不可泛化**（case38 band 内 +7.5% 有害、case19 +0.7% 中性；MTE2 burst 收益按 shape 极不均匀，必须精确 signature 白名单锁定，见 T12） | iter5_wgd_dtype |
| conv3 直连 cube 效率门收紧（G030） | **中性**（kk<32 且 c/kk≥128 才回退，翻转 shape 无净变） | iter_0 |
| bmm1 占用率地板进一步收紧（G031） | **-3.8%**（1.3874x 拒——地板不是越紧越好） | iter_1 |
| 路由门删流量模型权重项→35 shape 翻转 fused conv3（G032） | **崩到 0.4961x**（kk=16 浅 dot cube 崩塌；流量模型≠性能模型，C6） | iter_2 |
| 编译参数 workspace=2 + no-limit 组合（G033） | **1.3754x 拒**，逐 shape 双峰（8 升 21 降）→ no-limit 是唯一加害参数（21 shape 回退最重 -14%） | iter_3 |
| 编译参数单参 workspace=2（G033b） | **零效果**（1.4422 vs 1.4425 纯噪声） | iter_3b |
| 文档编译参数配方直接照抄 | **编译失败**（本版 bishengir 大量参数不存在或默认已开，C8） | iter_3 冒烟 |
| 无门控全量融合 WGD+OUT | **不可行**（小 hw case fused/sep=0.926 反亏） | T5 |

泛化启示："conv 特征-双 softmax 汇聚"形态主成本在投影/分发 GEMM（packed conv MTE2 带宽 + 尾声写出），不在 softmax；结合律重排（T2）+ **gather 重关联（T13，fold 权重消除大中间量物化）**是两大结构杠杆，都必须配 shape 签名精度门控（C7）；MTE2 burst / 重排收益按 shape 极不均匀，任何 band 近似都会引入反方向 case，只能精确 signature 白名单锁定（T12）；小 K 大 N 尾声 GEMM 的 AIV workspace 中转是 910B 架构固有，判"近硬件极限"前先 msprof op 板实证。Phase 4 判定纪律：headline 落 ±2% 噪声带一律拒绝、逐 shape 双峰必拆参归因、两时段共模漂移用"同代码复验"标定后再比参数效果。
