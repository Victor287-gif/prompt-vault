---
type: Guide
title: cv_attn_agg 类算子设计约束（CV-Attn-聚合分发）
description: CV 空间/通道特征经变换聚合（卷积/通道扩展/spatial shift/mean）→ 轻量 attention 分发（逐通道/逐分支/softmax-over-hw）→
  聚合回空间的算子类别：判别特征、聚合-分发结构轴与形态补三问、通用经验 C1~C8（Layer 1 = C1/C2/C3/C6/C7 载入 precheck）与精度闸门、Phase
  4 优化点清单与症状驱动陷阱表。
tags: [task.design, topic.attention, topic.compute.neighborhood.convolution, topic.compute.normalization, topic.precision, topic.compute.composite]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-26T08:27:14Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# cv_attn_agg 类算子设计约束（CV-Attn-聚合分发）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；
> 同属 attention 家族：attention 主链约束（KV 分块、online softmax）继承
> [flash_attention_constraints.md](flash_attention_constraints.md)，投影/GEMM 段约束继承
> [mha_constraints.md](mha_constraints.md)（L1.11/L1.12 权重预转置、§2.5 M1~M4），卷积与归约
> 精度分别继承 [convolution_constraints.md](convolution_constraints.md) / [normalization_constraints.md](normalization_constraints.md)；
> 家族分类路由见 [attention_writing_classification.md](attention_writing_classification.md)。

适用范围：**空间/通道变换聚合**（卷积 / 通道扩展 / spatial shift / mean 归约）→ **轻量 attention
分发**（逐通道 / 逐分支 / softmax-over-hw）→ **聚合回空间**。attention 的 KV 维**极小或极特殊**
（k=3 分支、逐通道权重、softmax-over-hw），**不是**序列 attention——FA 那套 KV 循环握手基本不适用。
**核心优化哲学**：优化杠杆完全取决于**聚合-分发的结构形态**（§0.2），主成本通常落在投影 GEMM 而非
attention；低精度中间量的舍入点（RNE 复刻）与本类最大结构杠杆（结合律重排）是两条生死线。

## 本类别的算子级实现手册

- [`operators/cv_attn_agg__s2attention.md`](../operator_playbooks/cv_attn_agg__s2attention.md) — 59_S2Attention（通道扩展-分支分发）：8 kernel 纯 Triton 流水线 + RNE/Kahan/tanh 三件套，性能基准 2.2702×、50/50。
- [`operators/cv_attn_agg__cotattention.md`](../operator_playbooks/cv_attn_agg__cotattention.md) — 42_CoTAttention（空间-通道双 attention）：CANN conv 路由 + Triton attention 融合链（及纯 Triton 并行线），性能基准 1.5147×、49/49（并行线 1.1775×）。
- [`operators/cv_attn_agg__doubleattention.md`](../operator_playbooks/cv_attn_agg__doubleattention.md) — 47_DoubleAttention（conv 特征-双 softmax 汇聚）：100% Triton rebuild（packed conv + softmax 融合 + P10 gather 重关联 + 26-sig 双门控），性能基准 2.2794×、50/50。

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件（CV-Attn-聚合分发类）：

1. **通道/空间变换聚合**：输入经 `Linear` / `conv1x1` / spatial shift / mean 先聚合成低维中间量（`[B, C]` 或 `[B, C', H, W]`）；
2. **轻量 attention 分发**：基于聚合量做 **softmax 加权**，KV 维**极小**（`k=3` 分支、逐通道、`softmax-over-hw` 空间维）——**不是**序列 attention；
3. **聚合回空间**：attention 输出再经 `Linear`/卷积/加权求和还原到原空间形状；
4. 算子是 CV 特征（`[B,C,H,W]` / `[B,N,C]`）而非 token 序列，attention 用于**特征门控/通道注意力**而非 token 间关系。

不满足 → 序列 attention 走 [flash_attention_constraints.md](flash_attention_constraints.md)；纯卷积走 [convolution_constraints.md](convolution_constraints.md)。

### §0.2 ★ 聚合-分发结构轴（**本类核心路由，Phase 2 第一步必须定位**）

本类的优化杠杆**完全取决于聚合-分发结构的形态**。先定位再选杠杆：

| 形态 | 含义 | 典型 | 核心杠杆 |
|------|------|------|---------|
| **(a) 通道扩展-分支分发** | 通道先扩 3x，拆 3 分支做不同变换，加权合并 | S2Attention (59) | 投影 GEMM（C→3C）+ 分支 kernel 融合 + 低精度 RNE 复刻 |
| **(b) 空间-通道双 attention** | 空间维与通道维各做一次加权聚合 | CoTAttention (42) | grouped conv + 两次轻量 attention + 归约精度 |
| **(c) conv 特征-双 softmax 汇聚** | conv1x1 变换 → 2 次 softmax(hw) + 2 次 bmm 汇聚 | DoubleAttention (47) | conv1x1 GEMM 化（独立三发射）+ 结合律重排双路（精度门控）+ bmm tile 路由 |

### §0.3 形态补充三问（在 FA §0.2 四问之上追加）

| # | 问题 | 影响 |
|---|------|------|
| **C-Q1** | attention 的 KV 维多大？`k=3` 固定分支 / 逐通道 / softmax-over-hw 空间维？ | 决定 attention 是否值得独立 kernel（k=3 点乘不可矩阵化）；大 KV 才走 FA 继承 |
| **C-Q2** | 聚合量是标量（`[B,C]` mean）还是空间（`[B,C',H,W]`）？ | 决定归约 kernel 与后续分发 kernel 的衔接 |
| **C-Q3** | 低精度（fp16/bf16）中间量有多少处舍入点？golden 对中间舍入敏感吗？ | 决定 RNE 位运算复刻的范围（本类最大精度坑） |

## §1 通用经验（跨成员，首次生成必须遵守）

> 本类**特有**、FA/conv/norm 未覆盖。FA F1~F6 / L1.1~L1.14、conv 卷积、norm softmax 不重复。
> **Layer 映射（Phase 2 读卡指引）**：**Layer 1（载入 precheck.json）** = **C1/C2/C3/C6/C7**（标注 〔L1〕）
> + 命中算子的「关键约束清单」（见各 playbook）；**C4/C5/C8** 标注 〔优化〕，是 Phase 4 优化指导、
> **不载入 precheck**（同 FA：L 系载入 precheck、F 系通用经验不载入）。漏载任一条 〔L1〕 可能把该条
> 对应的死路在 Phase 3 重踩一遍。
> **Layer 2（算法骨架，参考方向，输出必须是全新草图）**：§0.2 聚合-分发结构轴三形态 (a)/(b)/(c)
> + 各 playbook 的结构形态；**Layer 3（关键技巧，可参考但变量名/结构必须重新设计）**：各 playbook 的
> 制胜技术 T1-T13 与 kernel 片段。跨成员按收益排序的 Phase 4 落地顺序见 §5.1；症状驱动的陷阱排查见 §7。

- **C1 ★★ 〔L1〕必须 低精度中间量在每个舍入点用 RNE 位运算复刻，禁止 `.to(dtype).to(fp32)` round-trip**
  - **Why（59 实锤）**：`triton-ascend` 编译器把 `.to(dt).to(fp32)` 当 no-op 消掉；fp16 用硬件原生乘 MER 方差放大、bf16 原生乘超阈 3 倍；改 RNE 位运算后 38/50 → 50/50。
  - **How to apply:** bf16：`(u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000`；fp16：13-bit 尾数 RNE + 指数 rebias + FTZ/inf 处理，大 tile 用 4-op 变体避免 i32 比较标量降级。补充（42 纯 Triton 线实锤）：多个已各自舍入的 fp16 中间量**相加**时用 **fp16 域加法**（`a_16 + b_16`，Triton 不提升到 fp32 重融合）强制各中间量独立舍入；单 fp32 累加后一次舍入 FAILS MERE（3.7e-2 vs 阈 9.8e-4）。
  - **判别信号**：低精度 case 单独失败、`MERE` 卡在阈值附近 ⇒ 查 RNE 复刻，不是 tiling。

- **C2 ★ 〔L1〕必须 大 N 归约（mean/sum）用 Kahan 补偿求和，匹配 torch reduce 树**
  - **Why（59 实锤）**：`mean(xa, dims=(1,2))` 顺序累加 fp32 尘埃 ~2e-5，在 fp16 舍入边界产生翻转；Kahan 后 ~3e-8，消除边界失败。
  - **How to apply:** 归约 kernel 内用 Kahan 补偿累加；判别信号：某 case 的 mean 结果差 ~1e-5 量级、fp16 输出边界翻转 ⇒ 上 Kahan。

- **C3 ★ 〔L1〕必须 激活函数近似匹配 NPU 底层实现，禁止用数学精确式**
  - **Why（59 实锤）**：GeLU 用 tanh 近似匹配 NPU `F.gelu`；erf 精确式偏差 4.7e-4，fp16 边界超阈。
  - **How to apply:** `0.5*x*(1+tanh(√(2/π)*(x+0.044715x³)))`；判别信号：GeLU 相关 case 单独失败且偏差 ~5e-4 ⇒ 查激活近似。

- **C4 ★ 〔优化〕禁止 对聚合量小的中间 kernel（`[B,C]`/`[B,3,C]` 级）投入优化轮次**
  - **Why（59 实锤）**：S2Attention 的 mlp3/mlp4/softmax 合计 <80µs、占 impl <6%，**再优化无收益**。
  - **How to apply:** profiling 分解后识别"聚合量级" kernel（只算 `[B,C]` 或 `[B,3,C]`），直接跳过、不写进优化队列。

- **C5 ★ 〔优化〕必须 把投影 GEMM（C→3C / C→C）当主攻对象，禁止把优化重心放 attention**
  - **Why（59 实锤）**：S2Attention 的 MLP1/MLP2 占 56% 主成本；attention 只有 k=3 分支，算力极小。
  - **How to apply:** profiling 分解后按 kernel 占比排优先级，投影 GEMM 优先做 tile 分档（按 (M,N,K)）。

- **C6 ★★ 〔L1〕禁止 kernel 路由决策只用纯流量模型，必须含 cube tile 深度项（47 实锤）**
  - **Why（47 实锤）**：删掉路由门里的 cube 效率项后 35/49 shape 翻转到 fused conv 路由，几何平均**崩到 0.4961x**——机制：kk=16 的浅 dot 在 cube 侧塌陷（K 有效深度远小于 MMA 指令满载需求），省下的搬运远补不上 cube 崩塌。
  - **How to apply:** 任何"选 kernel A 还是 B"的路由门（如直连 conv vs im2col+GEMM 化）在决策函数里同时评估：①搬运流量 ②**cube 侧 K 深度**（dot 的 K 维 tile 展开深度 kk×BLOCK_K）。判别信号：路由翻转后大 shape 集体回退、回退幅度远超噪声带（-50% 级）⇒ 判据缺算力项，不是 kernel 本身变慢。泛化：**流量模型 ≠ 性能模型**，910B 上"少搬数据"只在 cube 不空转时才成立。

- **C7 ★★ 〔L1〕必须 结合律重排配逐 shape 精度门控，禁止无门控全量重排（47 实锤）**
  - **Why（47 实锤）**：`OUT = WR @ (GD @ SM_V) + b`（golden 次序，大 N 中间量 Z 先舍入 dt）重排为 `OUT = (WR @ GD) @ SM_V + b` 使中间量从 `[c, hw]`（大 N）缩为 `[c, c_n]`（rank 级）、大 N 写出 **2→1 次**；但 14/50 case 重排路径 MERE 9.3e-4~1.9e-2 vs 阈 9.77e-4(fp32)/7.81e-3(fp16)——**不门控必挂**，门控后 50/50。
  - **How to apply:** 离线全量标定重排路径 MERE，超阈 shape 用**签名集合**（shape+dtype 指纹）锁死走 golden-order 兜底，不得运行时试探。判别信号：重排/融合后大部分 case 提升但固定一小撮卡阈 ⇒ 上 shape 签名门控，别放弃重排也别全局回退。

- **C8 ★ 〔优化〕必须 编译参数先实测本机工具链支持集，文档配方不可直接抄（47 实锤）**
  - **Why（47 实锤）**：文档（07-compile-params.md）配方参数在本版 bishengir 大量不存在（`enable_flatten`/`enable_mixed_cv` 等）或默认已开（`multibuffer`/`sync_solver`）；实测增量仅 `set_workspace_multibuffer=2`（单参零效果，1.4422 vs 1.4425 纯噪声）与 `limit_auto_multi_buffer_of_local_buffer="no-limit"`（**有害**：UB 多缓冲不设限 → 大 tile gemm 的 vector 尾声过度碎化，21/49 shape 回退最重 -14%）。
  - **How to apply:** launch 时 kwargs 先用本机编译器 `--help`/冒烟实测支持集，文档配方里不存在的参数会让**编译直接失败**。判别信号：加参后逐 shape **双峰分布**（部分升部分降）⇒ 拆参重试定位加害参数；headline 落噪声带（±2%）⇒ 判零效果拒绝，别拿共模漂移当收益。

## §2 精度闸门（CV-Attn 特有）

> RNE 复刻（C1）/ Kahan 归约（C2）/ tanh 激活近似（C3）三件套见 §1，此处不复述。本节只收**本类额外**的精度闸门。

- **标杆种子确定性**：若标杆用 `torch.manual_seed(hash((channels, device, dtype)))`，`hash(torch.float16)` 是**对象地址派生**，跨进程随 ASLR 漂移 → 权重每进程重抽 → fp16 边际 case MERE 随机翻转（"抽签"）。**必须**锁步改为 `(zlib.crc32(repr(key).encode()) + k) & 0xFFFFFFFF`，k 经预扫选定使边际 case 有余量。判别：同一代码跨 verify 进程结果随机变化 ⇒ 查种子。
- **fp32 锁定判据**（继承 FA L1.4）：参考全程 fp32 + 量级超 dot 安全区时，全 fp32 中间 buffer 是唯一正确解。

## §N 证伪方向 / 常见陷阱（跨成员，持续积累）

> 本类证伪结论的**唯一全量入口**见 §5.3（跨成员持续积累）；§7 是按症状反查的排查表。

## §5 Phase 4 优化点清单（跨成员，按收益排序）

> C 编号见 §1（〔L1〕设计约束 / 〔优化〕优化指导），T/R 编号见各算子 playbook。Phase 4 先扫本表定位高收益点，再进对应章节取细节。

### §5.1 优化点按收益排序（★ = 高收益）

| # | 方向 | 实测增益 | 适用条件 |
|---|------|---------|----------|
| 1 | ★★ 低精度中间量 RNE 位运算复刻（C1） | fp16 38/50→50/50；bf16 原生乘超阈 3 倍 | fp16/bf16 且对中间舍入敏感 |
| 2 | ★★ 结合律重排 + 精确签名精度门控（C7，47 T2/T13） | 47 P10 相对 P9 **+24.2%**；大 N 写出 2→1 | 聚合-分发链有 rank 级中间量；**必须**配 fail-sigs |
| 3 | ★ 投影 GEMM 当主攻对象，别攻 attention（C5） | 59 投影占 56% | 通道扩展-分支分发形态 |
| 4 | ★ 路由决策含 cube tile 深度项（C6） | 只算流量崩到 **0.4961x** | 任何"选 A 还是 B"的路由门 |
| 5 | ★ 归约用 Kahan（C2） | fp32 尘埃 2e-5→3e-8 | 大 N mean/sum |
| 6 | ★ 激活近似匹配 NPU（C3） | erf 偏差 4.7e-4 | GeLU 相关 |
| 7 | ★ 标杆种子确定性修复（§2） | 抽签→可复现 | 标杆用 `hash()` 播种 |
| 8 | 编译参数实测本机支持集（C8） | 文档配方全失败；`no-limit` 有害（21/49 回退） | 想用编译参数时 |
| 9 | 跳过聚合量小的 kernel（C4） | 合计 <6% 无空间 | 小聚合量 kernel |
| 10 | 最大整除 M-tile（42 R17/R18） | 42 **+3.1%** | 多 kernel 流水 + 非 pow2 M，**仅 B 操作数巨大** |

### §5.2 结构性下限（小 case 别踩）

- **多 kernel 流水在小 case 上受 launch 开销 ~0.1ms/kernel 约束**（42 R2 实锤：3 分离 kernel=490us 反亏，融合单 kernel=330us 净赚）。小 case 先融合 kernel 削 launch 数，不路由 torch（纯 Triton 约束下）。
- 本类 attention 极轻（C-Q1），**别为 attention 拆 kernel**。

### §5.3 证伪方向全表（跨成员，持续积累）

> 继承 FA / conv / norm 全表；以下为本类**新增**。本表是证伪结论的**唯一全量入口**——各 playbook「已证伪」表只保留未入全表的行（重复行已并入本表）。

| 方向 | 结果 | 来源 |
|------|------|------|
| 低精度中间量 `.to(dtype).to(fp32)` round-trip | **编译器消除为 no-op**，精度 fail | C1（59） |
| 依赖硬件原生乘复刻低精度 | fp16 MER 方差放大 / bf16 超阈 3 倍 | C1（59） |
| mean 普通顺序累加 | **精度崩**（Kahan 才对齐 reduce 树） | C2（59） |
| GeLU 用 erf 精确式 | **偏差 4.7e-4**（NPU 是 tanh 近似） | C3（59） |
| 优化聚合量小的 kernel | **no-op**（合计 <6%） | C4（59） |
| 标杆种子用 `hash()` | **抽签**（dtype hash 地址派生漂移） | §2（59） |
| kernel 路由门只按搬运流量决策 | **0.4961x 崩盘**（kk=16 浅 dot cube 崩塌） | C6（47） |
| 结合律重排不做精度门控直接全量 | **14/50 case 超阈**（MERE 至阈 2.4 倍） | C7（47） |
| 文档编译参数配方直接照抄 | **编译失败**（参数不存在/默认已开） | C8（47） |
| `set_workspace_multibuffer=2` | **零效果**（1.4422 vs 1.4425 纯噪声） | C8（47） |
| UB 多缓冲 `no-limit` | **有害**（vector 尾声碎化，21/49 shape 回退最重 -14%） | C8（47） |
| 占用率地板越紧越好 | **-3.8%**（地板 40 再收紧反伤 bmm1） | DoubleAttention（47） |
| WGD+OUT 无条件融合 | **小 hw 反亏**（fused/sep=0.926） | DoubleAttention（47） |
| conv tile fine-sweep（BM 放大 / out_gemm 扫档） | **无普适赢面**（conv 是 MTE2 带宽 bound，128/256/128 近最优） | DoubleAttention（47, P8） |
| conv512 按 band 门控 | **不可泛化**（case38 band 内 +7.5% 有害） | DoubleAttention（47, T12） |
| 大 tile 靠 sweep 碰巧编译过即采用 | **UB 编译非确定**（388KB>192KB，须独立 forward 验证 tile） | CoTAttention 并行线（42, R5） |
| grid helper 已返回 tuple 再套一层 | **运行时错误** | CoTAttention 并行线（42, R6） |
| 未采集就断言「dot 是硬件瓶颈」 | **证伪**（CUBE 实测 9.6-10.8%） | CoTAttention 并行线（42, r16） |
| 最大整除 M-tile 扩展到小操作数 GEMM | **-3.9%/-4.1%** | CoTAttention 并行线（42, R19） |
| `num_stages` 一味加三 | **拒**（1.8478x < 1.8796x） | DoubleAttention（47, P8） |

## §7 常见陷阱与避免方法

> 症状驱动的排查表（学 FA）。按**你看到的症状**反查原因与治法；交叉引用见 C（§1）/ T / R（各 playbook）/ §2 / §5。

| 陷阱（症状） | 原因 | 避免方法 |
|------|------|---------|
| 低精度 case 单独失败、`MERE` 卡在阈值附近 | 中间量靠 `.to(dtype).to(fp32)` round-trip，编译器消除为 no-op；或依赖硬件原生乘（非完全 RNE） | **C1**：每处舍入点用位运算复刻 RNE |
| 某 case 的 mean 结果差 ~1e-5 量级、fp16 输出边界翻转 | 顺序累加 fp32 尘埃 ~2e-5，在舍入边界触发翻转 | **C2**：Kahan 补偿对齐 torch reduce 树 |
| GeLU 相关 case 单独失败且偏差 ~5e-4 | 用了 erf 数学精确式，golden 是 NPU 的 tanh 近似 | **C3**：改用 `0.5*x*(1+tanh(...))` 匹配 `F.gelu` |
| 同一代码跨 verify 进程结果随机变化（"抽签"） | 标杆 `torch.manual_seed(hash(...))` 的 `hash(dtype)` 是对象地址派生，随 ASLR 漂移 | **§2**：锁步改 `(zlib.crc32(repr(key).encode()) + k) & 0xFFFFFFFF` |
| 路由翻转后大 shape 集体回退、幅度远超噪声带（-50% 级） | 路由门只算搬运流量、缺 cube tile 深度项——省搬运却喂了空转 cube | **C6**：决策函数同时评估流量 + kk×BLOCK_K 有效深度 |
| 重排/融合后大部分 case 提升但固定一小撮 case 精度卡阈 | 结合律重排改变了中间量舍入次序，超阈 shape 是确定性的 | **C7 / T2 / T13**：离线标定 + 精确 signature 门控锁死走 golden-order 兜底；**禁止 band 门控** |
| 加编译参数后逐 shape **双峰分布**（部分升部分降） | 组合参数里某单个参数对部分 shape 有害 | **C8**：拆参逐一重试定位加害参数（`no-limit` 使 21/49 回退最重 -14%） |
| 小 case 多 kernel 反而变慢 | launch 开销 ~0.1ms/kernel 在小 case 不可摊销 | **§5.2 / 42 R2**：融合 kernel 削 launch 数，不路由 torch |
| 大 tile 隔离能编译、全量 forward 又失败 | UB overflow 的编译非确定性：auto-split 有时过、隔离/全量不一致 | **42 R5**：tile 可靠性以**独立 forward 验证**为准 |
| 单 fp32 累加 + 最终一次 fp16 舍入，逐 conv 舍入未复刻 | 多个已各自舍入的 fp16 中间量相加被提升到 fp32 重融合 | **42 R3**：fp16 域加法强制各中间量独立舍入（MERE 3.7e-2 → 0.0） |
| 把最大整除 M-tile 用到小操作数 GEMM 反而变慢 | M-tile 只在 B 操作数巨大（`[c,n]`）时省重读；小操作数 GEMM 加大 BM 只减 grid 并行 | **42 R19**：仅 value_embed/att1 这类 B 巨大形态用 |
| 拿噪声带 headline 当收益、事后复验无差异 | 两时段跑间共模漂移可达 +2%（同代码 1.4505→1.4336） | **测量口径节**：参数判定须"同代码复验基线"或逐 shape 确定性归因；跨日绝对测值漂移 ~-2%，同窗口对照才是事实依据 |
| 签名级收益想用 shape band 门控泛化 | MTE2 burst / 重排收益按 shape 极不均匀，band 内必混入反方向 case | **T12**：用精确 signature 白名单锁定受益 shape（case20 +9.14% / case38 band 内 +7.5% 有害） |
| 未采集就断言「dot 是硬件瓶颈 / 投影 GEMM 走 Cube 满载」 | 开销常在 dot 周围的访存 staging + 标量地址算术，不在 dot 本身 | **42 r16**：先过 msprof op 板全 kernel 采集（`--launch-count=300`）再定瓶颈 |

## §N 测量口径与时效（跨成员）

- **几何平均**：每 case 权重相同（FA §7.0）。本类主成本在投影 GEMM，优化要盯最慢的 shape 档。
- **groupby 口径陷阱**：benchmark 旧口径按 kernel 名 groupby（同名多次发射只计一次）——S2Attention 的 `_gemm_bias_kernel` 发射 2 次（MLP1+MLP2）旧口径漏计 1 次；47 号统一 `_gemm_kernel` **每 forward 发射 5 次**（iter5_wgd_dtype P7：packed conv/softmax/bmm1/wgd/out_gemm；早期 6-7 次结构见 doubleattention playbook 历史），groupby 口径必须按发射序号拆开归因，不能按 kernel 名聚合。跨口径比较须注明。
- **off-by-one**：perf_result.json 的 case_idx 为 **1 基**（= JSON 行号），归因脚本若 0 基需对齐后再比。
- **噪声地板与共模漂移**：本机 NPU 实测两时段跑间 headline 漂移可达 +2%（同代码 1.4505→1.4336）；headline 差 <±2% 判噪声拒绝，参数效果判定须有"同代码复验基线"或逐 shape 归因支撑。
- **msprof op 板采集**：必须加 `--launch-count=300`（默认只采 1 个 kernel）；产出按 `OPPROF_*/<kernel名>/<序号>/` 分目录，case_idx 对齐见上条。
- **时效**：RNE 位运算语义随编译器版本可能变，历史数字标注来源算子与日期，判据/机理优先于绝对数字，重测前重新标定。
