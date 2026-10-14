---
type: Guide
title: dual_branch_attention 类算子设计约束（CV 双分支特征图门控）
description: CV 双分支特征图门控类算子（BAM / Polarized Self-Attention 并行与串行 / TripletAttention）的判别特征、通用经验与
  Layer 1 设计约束；判别核心是特征图门控而非 softmax(QKᵀ) 主链。
tags: [task.design, topic.attention, topic.compute.neighborhood.convolution, topic.compute.reduction, topic.precision, topic.tiling]
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

# dual_branch_attention 类算子设计约束（CV 双分支特征图门控）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；
> 家族内写法类路由见 [attention_writing_classification.md](attention_writing_classification.md)。
> ⚠️ 与 FA 家族**无交叉**——本类无 softmax(QKᵀ) 主链，
> [flash_attention_constraints.md](flash_attention_constraints.md) 的 online softmax / KV 分块 / tile 放大摊薄经验**整卡不适用**。

本卡覆盖「五 专用类·特征图门控」细分：**通道支 `T[b,c]` 与空间支 `S[b,n]` 两路门控特征图**、
无 `softmax(QKᵀ/√d)` 主链的 CV 注意力算子（BAM / Parallel & Sequential Polarized Self-Attention / TripletAttention）。
凡满足 §0.1 判别特征的算子一律使用本卡；其中 k×k conv 的基础经验（G1-G13 通用约束）回 `convolution_constraints.md` 取。

> **核心优化哲学**：头号结构陷阱是 **[B,C2,N] 规模中间量物化**（O(B·C2·N) 流量 × 多遍），
> 第一动作永远是**结合律重排把它消掉**（F1，架构级、Phase 2 就要做）。
> 消掉后主链退化为 GEMV 链 / 小 conv 链，瓶颈二分：
> **小 case = kernel launch 串行主导（~11-13µs/个）→ 融合 kernel / 双路径分派**；
> **大 case = 权重流 / 带宽 → 任务循环档位化 / Split-K / conv 平面化 / store 行内仿射化**。
> 生成时**禁止混用** `flash_attention_constraints.md` 的经验。

## 本类别的算子级实现手册

- [`operators/dual_branch_attention__bam.md`](../operator_playbooks/dual_branch_attention__bam.md) — BAM：通道 GAP→MLP 支与空间 k×k 膨胀 conv 支加性门控 `x·(1+sigmoid(ch+sp))`；conv 帧平面化三件套 2.17→2.94。
- [`operators/dual_branch_attention__polarized_self_attention.md`](../operator_playbooks/dual_branch_attention__polarized_self_attention.md) — PSA：并行加性 op57 / 串行乘性 op61 两形态，7-kernel GEMV 链 + Split-K，2.3237 / 2.4783。
- [`operators/dual_branch_attention__triplet_attention.md`](../operator_playbooks/dual_branch_attention__triplet_attention.md) — TripletAttention：CH/CW/HW 三分支 `(o1+o2+o3)/3`，permute 消解为索引映射，store 行内仿射化，2.6433。

## §0 适用范围与算子分类

架构常量（ascend910_93 / Ascend910_9372，四算子共用）：UB 192KB、Cube:Vector = 1:2（20 cube / 40 vector cores）。
全 vector 路径 `NUM_CORES = 40`；含 Cube dot 的路径 `NUM_CORES = 20`。

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. forward 主链是 `out = x ⊙ 门控`，门控量来自**通道支 `T[b,c]` 与空间支 `S[b,n]` 两路**——
   加性 `x·(S+T)`（op57）、乘性 `x·T·S`（op61）、`x·(1+sigmoid(ch+sp))`（op36）、三分支 `(o1+o2+o3)/3`（op64）；
2. softmax 存在但作用在 **N 维（空间 softmax）或 C2 维（通道 softmax）**，**不存在** `softmax(QKᵀ/√d)` 主链；
3. 参考实现含 `[B,C2,N]` / `[B,C2,H,W]` 规模中间张量（V_c/V_s/channel_out 物化）——头号消除对象；
4. BAM 形态：`GAP → MLP(C→C/r→C) → sigmoid` 通道注意力 + `k×k dilated conv → pool` 空间注意力；
5. Triplet 形态：三分支 permute → zpool(mean/max 拼两平面) → 小 conv → BN → sigmoid → 相加平均。

不满足 → attention 家族走 `attention_writing_classification.md` 索引；纯卷积无门控走 `convolution_constraints.md`；
有 softmax(QKᵀ) 主链（哪怕带 CV 特征图）**不是本类**，回索引重判。

> **证据基础**：四个算子各 50 case 的完整优化轨迹——`36_BAM`（conv 平面化四步 2.1697 → 2.9430）、
> `57_ParallelPolarizedSelfAttention`（任务循环+档位化 1.9255 → 2.3237）、
> `61_SequentialPolarizedSelfAttention`（Split-K 2.3450 → 2.4783）、
> `64_TripletAttention`（IR store 行内仿射化 1.0580 → 2.6433）。
> 四者门控结构、分支内算子、kernel 数各不相同，却独立收敛到同一套瓶颈结论，这是本卡可外推的根据。

### §0.2 ★ 形态识别五问（Phase 2 第一步必须回答）

| # | 问题 | 影响 |
|---|------|------|
| **Q0** | 分支结构：并行加性 / 串行乘性 / 三分支平均？ | 决定依赖链与 kernel 数：并行两支可交错、串行有硬依赖（T 先行）、三分支互不依赖可乱序 launch |
| **Q1** | 分支内有没有 k×k conv（含 dilated）？ | 有真实 `[co,ci]×K²` GEMM → Cube dot + 平面化路径；纯 1×1/Conv2d(C→C2) GEMV → 全 vector 路径。判据见 L1.2 |
| **Q2** | 归约维最大值 N_max / C2_max / C_max 是多少？ | 决定定宽 softmax/LN lane 档位——**覆盖不足 = 静默精度失败**（L1.4，op57 case48 实锤） |
| **Q3** | 参考实现的中间张量是什么 dtype？ | 决定舍入点数量与位置：op57 仅 load/store 两处降精度，op61 四处（L1.5） |
| **Q4** | 评测口径（geomean）？ | 小 case launch 主导时，省 1 个 kernel ≈ 净赚 12µs，直接进几何平均 |

## §1 通用经验（跨形态，首次生成必须遵守）

### F1 ★★★ 结合律消除 [B,C2,N] 物化（本类头号杠杆，架构级）

参考实现把两个分支都写成大 matmul，中间量是 `[B,C2,N]` 规模。**必须**在 Phase 2 就用结合律重排消掉：

```text
通道支: matmul([B,C2,N], [B,N,1])  ==  W_wv @ (Σ_n a[n]·x[:,n])   ← 先 a 加权池化，再 GEMV
空间支: matmul([B,1,C2], [B,C2,N]) ==  (a_s @ W_sv)·x[:,n] + a_s·b_sv   ← 先权重头重组，再逐 n 点积
```

- 消除后中间量只剩 `[B,N] + [B,C]` 级；`channel_out = x·T` 同理不物化（串行版把 T 折进空间支权重 `w2 = whei ⊙ T`）。
- 50 case 中 C2×N 最大 ~1M×B，物化即 O(数十 MB) 流量 ×(写+读) 多遍，消掉是数量级差异。
- **Phase 4 再补做等于重写**——sketch 阶段就要写进架构决策。

### F2 权重复刻必须逐 draw 对齐（含「不使用但仍消耗 RNG」的权重）

**可直接套用的配方（四环节缺一即全错，op61 漏第 ③ 环全量失败）**：

1. **CPU 上 fp32 采样**——禁止直接以 fp16/bf16 采样；
2. 按 `kaiming_uniform_(a=√5)` / `uniform_` 的**同一 RNG 序列顺序**逐个 draw，顺序与次数和参考完全一致；
3. **不参与计算但仍消耗 RNG 的权重也要画**（op61 `ch_wv`：漏画一次 draw 后续全部权重错位）；
4. 采样完再 `.to(device, dtype)`，不要采样时降精度。

- **判别**：失败形态是**全量错**而非部分错 → 先把实现的权重与参考逐个 `allclose` 单独验掉，再查 kernel。

### F3 任务循环 `grid = min(NTASK, NUM_CORES)` + 核内段循环（全部 kernel 统一）

- op57 实测 **+20.7%**：小 C case 的 grid（582/1158）远超 40 vector cores，program 发射开销累积主导
  （同流量 `[6,32,61,101]` 233µs vs `[1,640,21,69]` 89µs，2.6 倍差距）。
- kernel 内 `for t in range(pid, NTASK, NPROG)`，每 program 连续处理若干任务。

### F4 定宽 lane 档位表 + host 侧覆盖断言

- softmax_N / softmax_C2 / LN_C 的 lane 宽度取**定宽档位**（编译次数最少），档位表示例（op61）：
  `NB = 256|2048|16384 by N`、`C2P = 128|1024`、`CP = 128|2048`、`CB = 32|64`，全 2 的幂。
- **必须在 host 断言 `NB ≥ N / C2P ≥ C2 / CP ≥ C`**——op57 iter_0 用 NB=8192 跑到 N=9797 的 case，
  softmax 尾部**静默丢质量**，50 case 只挂 1 个，排查代价极高；换 16384 全过。
- 小 case 按实际尺寸收缩档位（定宽 16384 在 N=64 时纯空转），这是同一条规则的两面。

### F5 小向量的跨 program 同步点用「冗余重算」吸收，不加 kernel

- softmax/LN 这类 `[B,N]`/`[B,C]` 小向量归约：让**每个 program 冗余算全量**（定宽 lane 一次 load），
  而不是为它单开归约 kernel 或 atomic。
- 共享小标量用**单写者直存**：仅 `blk==0` 的 program 写（op61 的 `zb`/`T_d`），其余 program 冗余持有。

### F6 无原子归约三件套（按数据规模选一）

| 形态 | 方案 | 实例 |
|------|------|------|
| 大向量跨 program 归约 | partials 落盘（`[B,T_N,C]` 布局）+ 下游 kernel 归并 | op57 的 xsum_part |
| 每 (b, blk) 一个输出 | 单写者直存，天然无竞争 | op61 K2 的 pool/xmean |
| 全量标量统计 | `grid=(1,)` 单程序双趟确定性归约 | op64 的 bn_stats |

### F7 dtype 舍入点契约：load 升 fp32 → 全程 fp32 → 仅在「参考会舍入的位置」落 dtype

- op61 四处：`u`(K3 出)、`zc`(K4 出)、`T_d`(K5 出)、`out`(K7 出)——匹配参考 conv+matmul 链的中间 dtype 舍入；
- op57 两处：仅 load 后升 fp32 与最终 store 降精度——其容差宽松（fp16 atol=9e-2，sigmoid+LN 阻尼放大）。
- 判别方法：打开参考实现看**中间张量的 dtype**，不是猜；失败集合严格按 dtype 划分 ⇒ 先查舍入点。

### F8 空/退化 case 早退（host 侧逐字匹配参考）

- `B==0 / C==0 / H==0 / W==0` → `return x`；verify 按 0 元素通过，benchmark `PROFILER_COLLECT_FAIL`
  是**预期行为**（框架无法 profile 空 shape），不计入 geomean，不要当代码 bug 修。
- `[1,1,1,1]` 退化 case：`C2 = max(1, C//2) = 1`，softmax 单元 = 1，LN 单元素 var=0 → 除 `sqrt(eps)`；
  各档位取最小档 + mask 全保护即可，无需特判分支。

## Layer 1 设计约束

> **本节共 12 条（L1.1 ~ L1.12）+ 3 条跨算子冲突判据，全部是 Phase 2 的硬性边界。**
> Phase 2 Step 1 产出 `precheck.json` 时**必须逐条载入全部 12 条**，不得只取前 N 条或按篇幅截断——
> 实测 op57 iter_0 漏掉 L1.4 覆盖断言，NB=8192 撞上 N=9797 的 case，softmax 尾部**静默丢质量**、
> 50 case 只挂 1 个。带 ★ 的（L1.1 / L1.4 / L1.8 / L1.9 / L1.10）是实测大收益或精度红线，尤其不能漏。
> **约束强度分级**：物理红线（违反必失败）——L1.4 覆盖断言、L1.8/L1.9 地址非负仿射、L1.11 `other=-inf`、
> F2 draw 序列、L1.12 逐字核对动作本身；架构目标——L1.1（消 [B,C2,N] 物化）、L1.10（store 行内连续），手段不限；
> 写法基线——L1.3、F5、F6；门控判据（数字来自本类 4×50 case，跨算子前先按新分布重标定）——F4 档位表、
> Split-K 门控、`C≤128`、冲突判据 3 的定界方法。
> 本卡约束的是**失败模式与量级目标，不是唯一实现路径**——新路线按 Phase 7 多样性保护并列归档，不覆盖旧经验。

### L1.1 ★★★ 禁止物化 `[B,C2,N]` / `[B,C2,H,W]` 任何中间张量

公式见 F1。**首次生成就必须做对，不是 Phase 4 的可选优化**——它是 sketch 的架构检查项（3.2b），
回退即架构违约；Phase 4 再补做等于重写。不是性能问题，是**不许写出来**。

- **How to apply:** 执行检查（3.2b 逐项核对用）：列出 sketch 中 kernel 间传递的全部中间张量及元素数——
  凡单个中间量达到 `B·C2·N` 量级（≈输入本身或更大）即违约；合法中间量为 `[B,N]` / `[B,C]` /
  `[B,T,C]` partial 级（T 为分核段数）。

### L1.2 ★ 路径判定：GEMV(M=1) 禁 `tl.dot`；真实 `[co,ci]` GEMM 才用 dot

- PSA 双支 / Triplet 门控 conv（C_in=2、C_out=1）：全是 M=1 或深度 <16 的归约 → **全 vector 路径**，
  禁 `tl.dot`（Triplet 的 conv 用 Cube 16×16 阵列浪费 15/16）。
- BAM 的 k×k conv（cr→cr，真实跨通道 GEMM）→ shifted-1×1 GEMM + `tl.dot`（Cube 路径）。
- **判别信号**：simulator 采集 `MMAD = 0%` ⇒ 纯 vector 定局；矩阵深度（min(co,ci)）< 16 ⇒ vector。

### L1.3 1D grid + int32 解码，禁 `%`

- `pid` 解码一律 `a - (a // b) * b`；禁 3D grid、`tl.swizzle2d`、kernel 内嵌套函数定义。

### L1.4 ★ 定宽 lane 覆盖断言（精度红线）

- host 侧断言 `NB ≥ N`、`C2P ≥ C2`、`CP ≥ C`（见 F4）；
- softmax 定宽 load 的 mask `other = -1e38`（参与 max/exp 后归零）；求和类 `other = 0`。
- **判别信号**：失败集合 = 恰好 1 个 case 且其 N（或 C2）为全组最大 ⇒ 直接查 lane 覆盖，
  不用扫参（op57 case48：NB=8192 撞 N=9797，50 case 只挂这 1 个）。

### L1.5 dtype 舍入点写进 sketch 的「精度匹配点」小节

- 逐个标出落 dtype 的位置（参考会舍入的位置才落），并写验证失败时的**回退序**
  （op61 实例：`qs 落盘舍入 → ζ 舍入 → g0 逐元素舍入`，逐级尝试）。

### L1.6 权重 / workspace 必须 host 缓存（shape 键控；zeros 的 halo 环只清一次）

- `_W_CACHE[(C, C2, dev, dtype)]`：miss 时按 draw 序列复刻（F2），一次构造终身复用。
- `_B_CACHE` / zeros workspace 按 shape 键缓存：`torch.empty` 部分全量覆写免清零；
  **zeros 的 halo/pad 环只清一次**，每轮只写 interior（op64 实测 +4.5%）。

### L1.7 host 侧只允许：`torch.empty/zeros` 分配、`.to(device)`、tuple 解包、kernel launch

禁止任何 torch 计算算子（matmul/softmax/conv/sigmoid/mean/max）与 `nn.*` 模块构造——退化检测会拦截。

### L1.8 ★★ conv 平面化：load/store 地址恒非负仿射

- **负地址**（`qi = pn + t_off - off0 < 0` 的 masked lane）→ `aicore exception 507015`。
  masked 正向越界实测安全，**负地址不行**。
- **禁止用 `tl.where` 钳位负地址**：破坏地址仿射性 → 编译器无法跨 tap 复用 UB buffer
  → `216KB > 192KB` UB 溢出 MLIR 编译失败（op36 实锤）。
- **正解（等价改写）**：把 off0 偏移从 load 侧挪到 store 侧——
  `block 平面位置 pn 的值 = Σ w_t·x[pn+t_off]`，写到帧位置 `q = pn + off0`
  （q 的正确 tap 输入恰为 `q + t_off - off0 = pn + t_off`）。两侧地址恒非负仿射。

```python
# ❌ load 侧带负偏移：pn + t_off - OFF0 < 0 的 lane 触发 507015 / tl.where 钳位触发 UB 溢出
x_t = tl.load(x_ptr + (pn + t_off - OFF0), mask=in_frame, other=0.0)
tl.store(out_ptr + pn, acc)
# ✅ off0 移到 store 侧：load/store 地址都是恒非负仿射
x_t = tl.load(x_ptr + (pn + t_off), mask=in_frame, other=0.0)
tl.store(out_ptr + (pn + OFF0), acc)
```

- **判别信号**：报 `aicore exception 507015` 或 MLIR `requires N bits ... UB` 编译失败
  ⇒ 直接查负地址 / `tl.where` 钳位，不要去查数值路径。
- **与 L1.9 耦合**：k≥2 平面化用本条 off0 改写；k=1 直接走恒等映射（L1.9），不要硬套偏移。

### L1.9 ★ flat 偏移两陷阱（k=1 恒等平面的前置知识）

1. **行宽不等 → 行漂移**：`q = pn + OFF0` 仅在读写行宽相同时保行。
   "carry-free"（OFF0 不跨界）是必要非充分条件——op36 首版读宽 ws1_w ≠ 写帧宽 ws2_w，直接行错位。
2. **OFF0 > 0 → padding lane 射入帧内**：`pn ∈ [FP, padded)` 的 `qn = pn+OFF0` 落在 `< FP_OUT` 区间，
   store mask 不拒绝 → relu(bias) garbage 污染。
- **正解**：恒等映射（OFF0=0，双侧 `pn < FP` mask 天然安全）+ 偏移由独立 `_expand_kernel` 承担。
- **判别信号**：输出整行错位 ⇒ 陷阱 1（读写行宽不等）；帧内特定位置出现 relu(bias) garbage ⇒ 陷阱 2（padding lane 射入）。

### L1.10 ★★ store 目标地址必须行内仿射（IR DiscreteMemAccess 红线）

- **症状**（IR 证据，op64）：`dst = (rows//H + PAD)*D2P + (rows%H + PAD)` 含整除/取模**跨行跳跃非仿射**
  → 编译器判离散地址 → 展开成 `scf.for` 标量循环逐元素 store
  （每迭代 1 float store + 4 次 wait/set_flag），比向量 store 慢一个数量级。
- **正解**：重构 program 粒度让 store 行内连续——`program=(n, c-block)` static_range 逐 c 归约，
  `dst = (c+PAD)*D2P + hv + PAD` 行内仿射 → 1D 向量 store。op64 两处（rowreduce/creduce）合计 **+13.0%**。
- **判别**：dump `last_pass.mlir` 搜 `DiscreteMemAccess`；或某 kernel 耗时与向量宽度严重不成比例。

### L1.11 max 归约的 mask `other` 必须 `-inf`（mean 归约贡献必须 0）

zpool/空间 max 归约中 `other=0` 会在**负值输入**时取错 max（op64 K1 明确标注）。
**判别信号**：仅含负值输入的 case 失败、非负 case 全过 ⇒ 直接改 `other=-inf`，不用扫参。

### L1.12 BN / LN 语义逐条钉死

- LN：**有偏 var（÷C 非 ÷(C-1)）**、eps=1e-5、γ=1 β=0（ones/zeros 无 RNG 消耗，但 draw 序列要留位）。
- BN（training batch 统计）：单通道每分支标量 μ/σ²、**无偏方差 ×n/(n-1)**、eps=1e-5、γ/β 取自权重元组；
  running stats 更新不影响输出，**不实现**。
- 三分支 z 平面顺序：**以参考实现的 cat 顺序逐字为准**（op64 实例是 `plane0 = mean`、`plane1 = max`，
  与 conv 权重 c2 维一一对应）——约束的是"逐字核对"这个动作，不是 mean/max 的先后；顺序错则分支输出互换。

### 跨算子结论冲突的统一判据（三条，照搬会互相矛盾）

| # | 冲突 | 甲方结论 | 乙方结论 | **统一判据** |
|---|------|---------|---------|------------|
| 1 | 小向量归约：冗余重算 vs 拆 kernel | op57 K2 每 program 冗余算 softmax（不加 kernel） | op61 K2 拆三段 Split-K（+2 launch，+5.68%） | 看**每 program 工作量**：N ≥ 2048 且 B·GC ≤ 16（任务少、单任务长）→ 拆；launch 串行主导的小 case → 冗余。两判据都写进 host 门控 |
| 2 | 门控 conv 走 dot 还是 vector | op36 BAM conv 用 `tl.dot`（Cube） | op64 Triplet conv 禁 dot（vector） | 矩阵深度：真实 `[co,ci]` 且 min(co,ci) ≥ 16 → dot；C_in=2/C_out=1 → 标量权重 × 向量 tile |
| 3 | 档位边界外推 | op57 档位化 +20.7% | op57 档位扩展（C≤256/CP=256）~持平 | 档位表按**实测 shape 分布**扫描定界，命中分布后停止；新增档位前先用单 case 验证，不外推 |

## §2 证伪方向 / 常见陷阱

### §2.1 ⛔ 关键证伪结论（绑定当时的架构形态，结构改动后需单 case 重测）

| 方向 | 结果 |
|------|------|
| **pass-merge：K4 顺带产出 zs_part（x 遍数 4→3）** | **-33%**（op57）。同一 tile 上 axis=0 + axis=1 **双方向归约**破坏循环流水/寄存器分配；pass-merge 必须保证顺带计算与原归约**同方向** |
| block_ptr 全量替换 masked load | **-24~-29%**（op36）。小 tile 下 block_ptr 固定开销超过 gather 节省 |
| g-major 任务重排 | -0.25%（op61），噪声级放弃 |
| BPR=128（行块翻倍） | -1.4%（op64） |
| masked load 形态改造 | -2.1%（op64） |
| tap 阈值 40（放宽分组门控） | -1.7%（op64） |
| rowreduce BLOCK_H 自适应 | 噪声（op64，后由 IR 行内仿射化整体取代） |
| creduce BHW2 分档 64/32/16 | +4~5% 有效（op64），但被 store 行内仿射化**整体取代**——同瓶颈的参数级与结构级解法，先试结构级 |
| C≤256/CP=256 档位扩展 | ~持平（op57），取 2.3181 < best 放弃 |
| softmax_a 并入 pool_part | **流量核算证伪**：+61% GM 读 vs 省 1 launch（op61）——融合前先算流量账 |
| #30 FA 专用（tile 放大 / online softmax） | 本类**不命中**——无 softmax(QKᵀ) 主链，整卡不适用 |

### §2.2 常见陷阱表

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 直接翻译参考实现，物化 [B,C2,N] | 参考实现用大 matmul 表达 | F1/L1.1 结合律重排 |
| 单个大 N case 精度失败 | 定宽 lane NB < N_max，softmax 尾部静默丢质量 | L1.4 覆盖断言 |
| 权重全量错 | draw 序列漏位（含不用仍耗 RNG 的权重） | F2（op61 ch_wv 实锤） |
| `aicore exception 507015` | masked lane 负地址 | L1.8 off0 移 store 侧 |
| MLIR 编译失败 UB 溢出（216KB>192KB） | `tl.where` 钳位负地址破坏仿射，跨 tap 无法复用 buffer | L1.8 等价改写 |
| flat 输出行错位 | 读写行宽不等，`q=pn+OFF0` 不保行 | L1.9 陷阱 1 |
| relu(bias) garbage 污染输出 | OFF0>0 时 padding lane 射入帧内 | L1.9 陷阱 2（恒等+expand） |
| 某 kernel 比同规模慢 10 倍 | store dst 含 div/mod 非仿射 → DiscreteMemAccess 标量循环 | L1.10 |
| max 归约结果错（负值输入） | mask `other=0` 参与 max | L1.11 `other=-inf` |
| 标量 bool 与向量 mask 组合再 507015 | 谓词组合触发编译缺陷（op36 conv2b `kmask & row_ok & col_ok`） | 行判定改标量 `if row_in:` 直接跳过（顺带省 tap load） |
| 平面化后下游读入 garbage | 输出 halo 列未清 | halo_zero 三段式 / IN_CORE constexpr 屏蔽 |
| 清零 kernel 极慢（百 µs 级） | 行循环微型 store / 混合 mask scatter 慢路 | 三段式纯连续（halo_zero） |
| B=0 case benchmark 假失败 | profiler 无法采集空 shape | F8 预期口径，不计 geomean |
| 三分支输出互换/颠倒 | z 平面 cat 顺序与 conv 权重 c2 维不对应 | L1.12 plane0=mean / plane1=max |
| 融合 kernel 越融越慢 | 双方向归约破坏流水 / 融合引入额外 GM 读 | 同方向才 merge，融合前算流量账 |

### §2.3 精度闸门与测量口径要点

- **容差**（verify.py，op57/op61 实测口径）：fp16 `atol=9e-2 / rtol=2^-10`、bf16 `1e-1 / 2^-7`、fp32 `1e-3 / 2^-13`。
- **诊断顺序**：① **全量错** ⇒ 权重 draw 序列（F2）；② 失败集合**按 dtype 严格划分** ⇒ 舍入点（L1.5）；
  ③ **恰好 1 个 case 且其 N/C 全组最大** ⇒ 定宽 lane 覆盖（L1.4）。
- **GAP(conv) == conv(GAP) 线性等价**（op57 空间支 `q_s = W_sq·xbar`）——注意只在**线性**层前成立，
  参考 GAP 之后有 ReLU/BN 的不可交换。
- max 归约 `other=-inf`、softmax `other=-1e38`、求和 `other=0`（L1.4/L1.11）。
- **官方 geomean 每 case 等权**：op57 opt_iter_0 把 `[6,32,61,101]` 从 0.974x 拉到 2.519x（一个 case），
  几何平均 1.9255 → 2.2650——**修最慢的 case 与把快的 case 翻倍贡献相同**。
- 定位手段：逐 kernel Event 计时（launch 主导）、`kernel_details.csv` Duration 占比、
  simulator 终局采集（MMAD/SCALAR/MTE + 热源码行）——下「硬件极限 / 无可优化」结论前**必须**采集；
  本类四算子终局画像一致：全 vector、MMAD=0%、SCALAR 为固定开销非降级。
