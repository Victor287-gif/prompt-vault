---
type: Guide
title: HaloAttention 算子（邻域展开） 实现手册
description: '典型特征: F.unfold/im2col 构造 halo 邻域 KV + 块局部 softmax(QKᵀ/√d)V，常带 nn.Linear
  投影段'
tags: [task.implementation, topic.attention, topic.implementation.masking, topic.compute.neighborhood.convolution, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:23:10Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# HaloAttention 算子（邻域展开） 实现手册

> 类别约束见 [../sparse_unfold_constraints.md](../constraints/sparse_unfold_constraints.md)（L1.1~L1.5 硬约束在此），
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §1 HaloAttention 算子（sparse_unfold）

**算子类别**: `sparse_unfold`　**典型特征**: F.unfold/im2col 构造 halo 邻域 KV + 块局部 softmax(QKᵀ/√d)V，常带 nn.Linear 投影段。
**性能基准**: 50/50 pass，几何平均加速比 **2.0973 / 2.1243**（双跑，910B2C / 24 AI 核）；分 dtype fp16 2.0240 / bf16 2.3424 / fp32 1.9353（target 2.0 达标，fp32 为本类薄弱档）。

> **证据基础**：`50_HaloAttention` 50 case 完整定位轨迹——fp16 24 / bf16 15 / fp32 10
> （block∈{2..8}，halo∈{0..4}，c∈{8..2048}，含 b=0 空输入）。iter_0 **1/50** → 16 个对照实验逐项定位
> → 精度 **50/50**，两次独立 benchmark 几何平均 2.0973 / 2.1243。
> 优化哲学：unfold 物化 `[b, c·S_kv, L]` 巨量拷贝是显性浪费，读放大 `(1+2·halo/blk)²` 倍
> （block=8/halo=4 时 **4 倍**）；**第一优化动作 = 每像素单次 dense 投影 + 按块窗口 gather**，
> 且投影 GEMM 占总算力 >90%，**先攻投影侧**。

### §1.1 Layer 1 设计约束（算子专属，逐条遵守）

类别卡 L1.1~L1.5 全部适用，本算子命中情形：

- **L1.1 ★★★ 舍入契约首项**：q/kv 投影用**自研 Triton GEMM**（K 循环顺序累加，M 行布局不敏感）
  按像素主序每像素投影一次；**attention 输出 → out 投影边界按参考块主序落盘**
  （`bh*SQ + ms` 行序）——像素主序落盘 + 下游 aclnn 语义会 **98.7% 元素 1 ulp 不等**。
- **L1.2 ★**：softmax 按分 dtype 配方——sim 物化在输入 dtype；**仅 fp16** 把 `(sim−m)` 舍回 fp16 再 exp；bf16/fp32 减法直接 fp32；exp/sum/div 恒 fp32；p 落输入 dtype。
- **L1.3 ★**：全部行号解码用 **constexpr 除数**（`NH / nhbnwb / nwb / BLK`）＋纯乘法
  `nbk * SQ + offs_s`；**禁止运行时除数 div/mod**（Nblk=512 越界 qrow=4108>4095、非确定）。
- **L1.4 ★**：`tl.dot` 操作数 next_pow2 封顶（`_pow2(SKV, lo=16, hi=128)`）；ceil16 只用在向量 tile。
- **L1.5 ★**：16 位融合投影取 `(BM, BN=64, BK=64)`；BN>64 的 weight-tile 转置静默错值，扫参结果先过精度再采信。

### §1.2 Layer 2 算法骨架（前向五阶段，kernel 划分是参考不是教条）

数据流按"**连续化 → 投影 → 窗口取数 → attention → 回写**"五阶段组织；每个阶段一个独立 kernel。
**照抄阶段职责与任务分解才有意义，最优 kernel 数量在 4~7 之间随 shape 浮动**。

```
阶段 A（可选） 布局连续化    X[b,C,HW] --纯搬运--> XT[b,HW,C']
    · NCHW channel 大步离散，token 化后 A 操作数行才连续（FA 卡 L1.6：禁 host permute）
    · C' = C 向上对齐（与权重同步 pad）→ 下游 GEMM K 循环免 masked 尾块
    · 纯 Vector kernel：grid = min(VEC 核数, tiles) + 核内步长循环

阶段 B          融合投影      P = XT @ [Wq | Wkv]ᵀ
    · q|kv 拼一张权重（预转置 [in,out]，FA 卡 L1.12），**每个像素只投影一次**
    · 行序任意（自研 Triton GEMM 顺序 K 累加对 M 布局不敏感，L1.1）
    · scale 为 2 的幂时可折进 epilogue；BN 档位受 L1.5 约束

阶段 C          窗口取数      KV[nblk, (blk+2h)², 2*inner]  <- 从 P 按固定几何窗口读
    · unfold 物化在此消除：窗口行号由 (b, hb, wb, ki, kj) 纯乘加推导
    · halo 越界行取 0 → k=v=0 → score=0 保留在 softmax 分母（Q2，勿当 padding 排除）
    · 纯 Vector kernel：grid = min(VEC 核数, NBLK_total)

阶段 D          块局部 attention   O = softmax(Q Kᵀ · scale) V
    · S_kv ≤ 256 → **单趟精确 softmax**（m/l 全列归一，免 online rescale）
    · softmax 按 L1.2 分 dtype 配方；QK/PV 两处 dot 输出均落输入 dtype（FA 卡 L1.16）

阶段 E          out 投影 + 回写    y = O @ Woutᵀ (+ bias) --> NCHW
    · A = O 按**参考 to_out 输入的行序**落盘（块主序，L1.1 边界约束）
    · bias 加在 fp32 累加器再 cast（Q3：nn.Linear 默认 bias=True）
    · 块主序 → NCHW 的 fold 是独立 Vector kernel：2D grid (NBLK_total, c_tiles) 零解码

host
    · 权重复刻按 S1（含 bias 核对）；空输入直接 return（S2）
    · 核数 __init__ 时读 npu_config：Vector 类阶段 grid 上限 40 / Cube 类阶段 20（910B2C）
    · 全部 dot 操作数 2 幂（L1.4）；16 位融合投影 BN 档受 L1.5 限制
```

**阶段合并/省略判据**（Phase 2 草图期逐条回答）：
- **A 阶段**：C 维大步离散且 C 不对齐 → 做（投影是算力大头，A 连续 + K 免尾块值一次全量搬移）；C 已对齐/极小 → 省。
- **B/C 合并**：窗口 gather 直接进 B 的 A 操作数会**重复投影 R 倍**（省 P 物化）；先每像素投影一次再 gather（本类默认）消 R，代价是一份 P 中间量 + 一次 launch。
- **D 单块/分块**：S_kv ≤ 256 恒成立域单趟精确 softmax；超上限分块 online 或两趟法。
- **E fold 融合**：投影 store 直接按 NCHW 地址 scatter（省一次搬移，但短连续段）；独立 fold kernel（默认，store 连续性好）——何者快随 shape 变，Phase 4 实测定。

### §1.3 ★★★ 投影侧 vs attention 侧——收益判据（先算账再动手）

设读放大 `R = (1+2·halo/blk)²`，投影 FLOPs 占比 `φ = F_proj / (F_proj + F_attn)`：

- **判据 A（本类常态）**：`φ > 80%`（c 大、head_dim=64 固定时几乎恒成立；op50 的 c=2048 case：
  投影 ~866 GFLOP vs attention ~68 GFLOP，φ≈93%）⇒ **主攻投影侧**（每像素单次 dense 投影消读放大 R、
  权重预转置、tile 档位），attention 侧只在 SKV 贴 UB 上限时扫一档。
- **判据 B**：`φ < 50%`（c 极小如 c=8）⇒ attention/gather 侧主导，主攻 K3 gather 与 K4 tile。

**每像素单次投影 vs 按窗口重复投影**：窗口直接进 GEMM A 操作数会把 halo 重叠区像素**重复投影 R 倍**
（block=8/halo=4 时 4 倍投影量）；先每像素投影一次再 gather（阶段 B + C）把重复消到 1——
这是本类与"窗口 gather GEMM"写法的分水岭。

### §1.4 Layer 3 关键技巧（可参考，不可照抄）

- **舍入点逐位复刻（精度闸门，先于一切性能优化）**：按六舍入点对 aclnn golden 逐位对齐——
  ①q 投影（含 ×scale=2⁻³ 精确可折 epilogue）②kv 投影 ③QK bmm（fp32 acc → cast 输入 dtype → 升回）
  ④softmax 输出（L1.2 分 dtype 配方）⑤PV bmm ⑥out 投影（**含 bias**）。缺一即 96-99% 元素错 1-2 ulp。
  全 fp32 或全 fp16 内部都差 4.88e-4（8 变体穷举），目标是**对齐 aclnn，不是算得更准**。
- **每像素单次 dense 投影 + kv gather 消读放大**：block=8/halo=4 时消 4× 投影量（§1.3 判据 A 的头号动作）。
- **解码安全化**：constexpr 除数解码（`NH / nhbnwb / nwb / BLK`）+ 纯乘法行号 `q_row0 + i1*W + j1`，
  既过 L1.3 又天然给出 L1.1 要的块主序；**弃用** host 行号表与运行时除数。
- **C-pad 64 倍数 + 权重同步 pad**：XT 通道维补零到 64 倍数、投影权重同步 pad → dense matmul 的 K 循环
  **无 masked 尾块**：累加精确且省 mask 向量开销。归因第 5 位，零风险顺手做。
- **K 转置直读**：`kt` 直接按 `[HD, BSKV]` 转置态 load（d 方向 stride 1）免 `tl.trans`；正常 load + `tl.trans(k)` 同样过。
- **VEC/CUBE 双核数分流**：转置（K1/K6）与 gather（K3）用 `vector_core_num`（40）；GEMM（K2/K5）与 attention（K4）用 `cube_core_num`（20）。混用同一上限会让 Vector kernel 少拿一倍核。

### §1.5 性能基准 / 优化记录表

| 方案 | 几何平均 | 说明 |
| --- | --- | --- |
| **终态六 kernel**（K1 连续化 / K2 融合投影 / K3 gather / K4 块局部 attention / K5 out 投影 / K6 fold） | **2.0973 / 2.1243**（双跑） | 50/50 verify；target 2.0 达标，余量 ~5% < 跨会话噪声地板（~4.6%），**必须双跑判定** |
| 分 dtype | fp16 2.0240 / bf16 2.3424 / **fp32 1.9353** | fp32 本类薄弱档（ieee dot 吞吐天花板） |

- **测量口径**：b=0 空输入 case `PROFILER_COLLECT_FAIL` 属预期（S2，计入 total 不计入 geomean）；
  精度只看 verify 的 passed/total（50/50），perf 的 `passed_cases`（49/50）是采集成功数，语义不同。
- **噪声带**：单 case ±30%、跨会话 ±2~5%——2.0x 达标必须双跑（两跑 2.0973/2.1243）。

### §1.6 陷阱 / 证伪方向（不要重跑）

| 现象 | 根因 | 处理 |
|---|---|---|
| NaN ~10% + 后续 case 级联 `aicore exception` | 运行时除数解码越界行号，级联是设备污染 | 先单跑第一个失败 case → L1.3 |
| ~96-99% 元素差 1-2 ulp、maxd ~0.045 | 舍入契约错位三连：边界布局 / softmax 域 / 漏 bias | 六舍入点逐处核对 → L1.1/L1.2/Q3 |
| fp32 case `should be static` 编译崩溃 | 非 2 幂 dot tile | L1.4 |
| 单 case 恒 0 / 半张图未写 | 空输入分支未复刻 / K6 scatter 漏 tile | S2 + 覆盖率检查 |
| 扫参后更快但 matched_ratio 断崖 | BN>64 weight-tile 转置静默错值 | L1.5 |

证伪方向（详见类别卡 §2.1）：attention 输出像素主序落盘 + aclnn golden ⛔ 98.7% 不等；
softmax 全 fp32/全 fp16 ⛔ 8 变体全差 4.88e-4；运行时除数解码 ⛔ 规模相关非确定越界；
非 2 幂 dot tile ⛔ 编译崩溃；merged BN>64 ⛔ 静默错值；按窗口重复投影 ⛔ R 倍投影量。
