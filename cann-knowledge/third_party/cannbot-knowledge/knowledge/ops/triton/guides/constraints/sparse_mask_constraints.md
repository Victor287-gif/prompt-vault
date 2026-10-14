---
type: Guide
title: sparse_mask 类算子设计约束（三·稀疏类·掩码型）
description: Sparse-掩码型注意力（窗口局部 / 块对角 / 元素周期 / 复合 window∪global / 轴向分解 / 加性偏置）经验：以「掩码生效层
  + 掩码几何形态」为核心路由轴，含通用经验 M1~M12、多算子落地与证伪。继承 flash_attention_constraints.md 只写掩码 delta。
tags: [task.design, topic.attention, topic.implementation.masking, topic.compute.composite, topic.tiling, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-20T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# sparse_mask 类算子设计约束（三·稀疏类·掩码型）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)；底层机制（online softmax / UB 预算 / fp32 契约 / 投影 GEMM）继承
> [flash_attention_constraints.md](flash_attention_constraints.md)；家族内写法类路由见 [attention_writing_classification.md](attention_writing_classification.md)。

> **本类算子**：attention 的 score 矩阵**不是**朴素稠密 `Q@Kᵀ`，而是被**结构性调制**——掩码（causal / padding / 窗口带 / 块对角 / 元素周期 / 复合 / topk 的 `-inf` 门控）、**结构化加性偏置**（相对位置偏置 / ALiBi），或**结构局部化**（窗口分区、轴向分解）。共同点：score 模式有**可利用的结构**。
>
> ⚔️ **与 [flash_attention_constraints.md](flash_attention_constraints.md) 的分工**：本类算子**底层仍是 online softmax + KV 分块**，FA 卡的 §1（F1 ghost 列 / F2 有限极小值 / F3 ceil16 / F4 UB 预算 / F5 改结构后重扫 / F6 大中小三档）、Layer 1（L1.1 tl.dot 契约 / L1.3 online softmax 同步缩放 / L1.4 精度契约 / L1.13 BLOCK 面积预算 / L1.14 propagate_nan 分档 / L1.16 逐步舍入回输入 dtype）、§4（BLOCK 推 UB / 权重重排 / 投影分块+并行度保护 / CV 重排）、§5.2 证伪表**全部继承，不在此重复**。本文件**只写 delta**：掩码 / 稀疏 / 结构化偏置带来的**额外**约束、杠杆与陷阱。

---

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

满足以下任意一条 → 用本文件（Sparse-掩码型注意力优化经验）：

1. forward 含 `mask` / `attn_mask` 参数，或 score 注入 `-inf` / 有限极小值门控（causal / padding / 自定义布尔掩码）；
2. score 叠加**结构化加性偏置**（相对位置偏置 `rel_bias`、ALiBi、可学习 bias），形状/索引由几何或学习参数决定；
3. attention **结构局部**：token 分进窗口/块，只窗口/块内 attend（Swin 窗口、局部滑窗），或 **Axial 轴向分解**（2D 局部化被消费为两次 1D 稠密 attention）；
4. KV 由 `indices` / `topk` / block-sparse pattern 选择性参与（只算被选中的 key/value）；
5. score 掩码为**元素级周期**（`j%stride == i%stride` 残基类掩码）；
6. **复合掩码**：多种成分叠加（窗口带 ∪ 全局行/列 ∪ 随机 gather，Longformer / BigBird）。

### §0.2 ★ 掩码/稀疏的「生效层」三分（**本类核心路由，设计第一步必须定位**）

掩码/稀疏的**优化杠杆完全取决于它在哪一层生效**。先定位，再选杠杆——**攻错层 = no-op 或回归**。

| 生效层 | 含义 | 典型算子 | 可用杠杆 | 不可用杠杆 |
|--------|------|----------|---------|-----------|
| **(a) host 侧结构稀疏** | 稀疏在算子边界外已被消费（窗口预切分 / 轴向 reshape），kernel 看到稠密**小 N** | 单窗口 WindowAttention、AxialAttention | 投影 GEMM + attention tile 推 UB + 小 N launch 开销意识 | ❌ 块跳过 / mask 特化 / KV 区间收缩 |
| **(b) kernel 内 score 掩码/偏置** | 掩码（`-inf`）或加性偏置在 kernel 内、softmax 前注入 score | causal / padding / rel-bias / ALiBi / 真·shifted-window；窗口带、元素周期、复合 window∪global | mask 注入精度对齐、mask 分支 constexpr 特化、**带区间收缩**（kernel 内块跳过）、加性偏置 gather 路径、**复合成分分解 OR 合并** | ❌ 把加性偏置当掩码"跳块" |
| **(c) 块级跳过** | 稀疏 pattern 决定**哪些 QK 块根本不算** | 块对角小块、topk-KV | program=掩码块粒度、**单 tile 一次性 softmax**、块边界对齐 tile、索引 gather | ❌ 朴素稠密 tile（白算被掩块） |

> **op14 = (a) + (b) 的加性偏置子集**：窗口稀疏 host-batched（N=Wh·Ww），rel_bias 是 kernel 内加性注入。故制胜靠 (a)/(b) 杠杆（投影 + tile 推 UB），**不是 (c) 块跳过**。

**已落地成员速查**：

| 算子 | 生效层 | 掩码几何 | 对角保底 | 成绩（geomean，50/50） |
|------|--------|----------|:---:|------|
| op14 WindowAttention | (a)+(b) 加性 | host 窗口 + rel_bias | 免（加性无 `-inf`） | 2.1968（旧跑）→ 0817 复跑 3.3972→**4.0054** |
| op23 LocalAttention | (b) 门控 | 窗口带 \|i−j\|<w | ✅ 含对角 | 首版 **3.2810**；复跑 2.7557 |
| op24 BlockSparseAttention | (c) | 块对角，block≤31 | ✅ 块内全 attend | 首版 **3.6387**；复跑 2.6048 |
| op25 StridedAttention | (b) 元素 | j%s==i%s | ✅ 含对角 | 首版 2.4029；复跑 2.4683→**3.4325** |
| op51 LongformerAttention | (b) 复合 | 带 ∪ 全局行列 | ✅ 自窗口保底 | 0.4114→**2.7685**（6 轮） |
| op35 AxialAttention | (a) | 轴向分解稠密小 N | 免（无掩码） | 0818 重跑 iter_2 首过 **2.4269**，#12 分档后 **2.4388** |
| op37 BigBirdAttention | (b) 复合+随机 | 带 ∪ 全局 ∪ random gather | ✅ 自窗口保底 | iter_2 首过即 **67.0198**（host mask cache 通吃，Phase 4 零优化） |

### §0.3 形态补充四问（在 FA 四问之上追加）

| # | 问题 | 影响 |
|---|------|------|
| **M-Q1** | 掩码/稀疏在哪层生效（§0.2 a/b/c）？ | 决定整篇适用章节；答错攻 no-op |
| **M-Q2** | 掩码固定还是运行时可变？ | 可变 → mask 分支 constexpr 特化 +（causal/带）KV 区间收缩是大杠杆；固定 → 这两笔拿不到 |
| **M-Q3** | score 调制是**门控**（`-inf`，改 softmax 分母）还是**加性偏置**（不改分母）？ | 门控须处理 ghost 列/整行被掩；加性偏置不需要，但精度仍须对齐 softmax 域 |
| **M-Q4** | 掩码是否**含对角/自窗口保底**？ | 是（窗口带 / strided / 块对角 / longformer 自窗口）→ 每行至少一个有限 score，**豁免整行被掩 fallback**；否（纯 padding 全掩行等）→ 必须 fallback |

---

## §1 通用经验（跨成员，首次生成必须遵守）

> 本类**特有**、FA 未覆盖。

### M1 ★ 先定位「生效层」，再选杠杆

- **必须** 写 kernel 前按 §0.2 确定 a/b/c。
- **禁止** 对 (a) host-batched 稀疏算子套 (c) 块跳过/mask 特化——稀疏已不在 kernel 内，no-op；去动偏置反而回归（op14 连续两轮回归）。
- **Why**：op14 窗口稀疏 host-batched 后 kernel 内是稠密小 N；连续两轮"再榨掩码/偏置"全回归。攻错层是本类最大浪费。

### M2 掩码/偏置注入精度必须与 softmax 归约域对齐

- **必须** 掩码值（有限极小值）与加性偏置都在 **softmax 累加域**（通常 fp32）注入，再进 online softmax。
- **禁止** 低精度注入 `-inf`/偏置后升精度；禁止偏置降到与累加域不同 dtype 再相加。
- **Why**：op14 参考全程 fp32，rel_bias 须 fp32 域相加才 50/50；降 fp16 相加则 dtype 转换+行对齐劣化抵消流量。门控掩码同理：低精度 `-inf` 经 cast 可能变 NaN。
- **fp16 特例（op37 实测）**：参考链本身在 fp16 域（NPU `F.softmax(fp16)` 全链）时，注入与归约**必须在 fp16 域逐位复刻**，而非笼统升 fp32——完整三连见 M11。

### M3 门控掩码的"幽灵列/整行被掩"必须处理（**加性偏置不触发**）

- **门控型**（`-inf`）必守：① 被掩 key 在分母贡献为 0（继承 FA F1：`sum` 前 `tl.where(kv_valid, p, 0.0)`，或利用 `exp(NEG−m_new)` 精确下溢为 0）；② **整行被掩** → `softmax(全 -inf)=NaN`，须 fallback（该行取 `mean(V)` 或零）——**M-Q4 含对角保底者豁免**（M9）。
- **幽灵列**：masked load `other=0.0` 的行/列 score=0，**会混过窗口带/残基类判定**（op23 实测），keep 掩码必须显式 `& (offs_kv < S)`。
- **加性偏置例外**：rel_bias 加性、无 `-inf`，每 key 都 attend，**不触发** M3。
- **验证集**：门控算子主动构造"整行被掩"case；加性偏置算子构造偏置量级远大于 score 的 case。

### M4 tile 策略必须与掩码/稀疏结构协同

- **(a)/(b) 稠密窗口内**：直接把 attention tile 推到 UB 上限，按 D 分档——op14 夺冠靠这条（+23%）。
- **(b) 带区间收缩**：BLOCK_KV 不必对齐窗口边界，带内逐块扫即可；BLOCK_Q 对齐无要求但决定收缩粒度。
- **(c) 块稀疏**：tile 边界**必须对齐稀疏块边界**，否则每 tile 内 mask 分支爆炸、UB 翻倍编译失败。block_size ≤ 31 时直接 program=单块、BLOCK_M=32 覆盖。
- **禁止** 朴素稠密 tile 算块稀疏——白算的被掩块吃掉稀疏红利。

### M5 小块 / 小 N 的 launch 开销地板

- 含 softmax 的 kernel 在 triton-ascend 下必编成 `MIX_AIC`，每次发射 ~4.5µs 固定成本（FA §5.3）。
- 本类常是**小 N**（窗口 N、轴向 N、小 S）或**多小块**（块稀疏），最弱 case 易被 launch 钉死。op14 N∈[49,361]，最小 case 仅 1.27×；op51 S=13~43 的 case 4~5× 而 S≥128 且 D≥256 的 fp32 case 掉到 1.0~1.7×（计算主导）。**小 case 上不要拆 kernel**。

### M6 ★ 门控掩码 kernel 内实现形态四分（按掩码几何选）

| 掩码几何 | 实现形态 | 代表 | 关键 |
|---------|---------|------|------|
| **带/窗口型** | KV 区间收缩：`kv_lo=max(0, q_start−(w−1))`、`kv_hi=min(S, q_start+BLOCK_Q+(w−1))`，带外整块不 load | op23 | 块级跳过的 kernel 内形态，稀疏红利直接 |
| **块对角+小块（≤32）** | program=单掩码块，只 load 本块行，**单 tile 一次性 softmax** | op24 | KV 循环整个消失 |
| **元素周期型** | dense KV 扫描 + 元素算术掩码（BLOCK≥stride 时每 KV 块含全部残基类，**无法块跳过**） | op25 | 掩码用 `a−(a//b)*b`，禁 `%` |
| **复合型** | 各成分**分别算术生成后 OR 合并**为单个 keep，一次 `tl.where` | op51 | 单 vsel 约束（M10） |

- **禁止** 对元素周期型套带区间收缩（残基类散布全行，收缩后仍须逐元素判定，白忙）。

### M7 ★ 掩码算术的编译器毒性四律（BiShengIR，codegen 层硬约束）

1. **i32 2D LT/GT/LE/GE 比较标量降级 10~100×** → 2D 窗口/带比较**转 fp32 比较**（op51，+6.6× 最大单笔）。1D mask 与 EQ 比较保持整型。
2. **1D mask 的 sitofp+fcmp IR 模式确定性触发 root-alloc 编译失败**（22/50 同指纹）→ pad/区间指示改用 **vmin/vmax 纯算术 clamp** 生成，排除动作改为 `s −= pad_f*1e9`。
3. **mask 一旦被用作张量**（`.to(f32)` / 第二次 `tl.where` 消费）→ 物化 i64 标量比较循环；**仅作 load/store mask 时走 subview 边界零开销**。设计时让每 1D mask 只有一种消费方式。
4. **模运算禁 `%`**，用 `a−(a//b)*b`；**strided 行索引向量禁参与掩码/寻址算术**（aicore 507015）——用 dense `qi` + `qi < sr` 等价判定规避。

### M8 单 tile 一次性 softmax 退化（块稀疏/小块算子的最大红利）

- 当每个 program 的 KV 全集能被**单个 tile** 装下（块对角 block_size≤31 < BLOCK_M=32；轴向 N=H/W 小），**online softmax 退化为单 `tl.dot` + 一次 max/exp/sum**：无 m/l/acc 滚动状态、无 alpha 重缩放、无 KV 循环。
- **机制**：online softmax 的全部复杂度来自 KV 分块；单 tile 时它纯亏。op24 首版 3.64× 的核心即"块对角 → program=块 → 单 tile softmax"三连。
- **判据**：`max_kv_per_program ≤ BLOCK_KV 上限`（UB 预算内）即触发；否则保持 online softmax。

### M9 对角保底豁免整行被掩 fallback（M-Q4 的推论）

- 窗口带、strided（`i%stride==i%stride` 恒含自身）、块对角（块内全 attend）**均含对角** → 每行至少一个有限 score → 整行被掩 fallback **不触发**，省掉 fallback 分支。
- **但仍须**：`m_new = tl.maximum(m_new, MCLAMP)`（MCLAMP=-1e30）防 `-inf − (-inf) = NaN`；`l_safe = tl.where(l==0, 1, l)` 防 0/0。
- 纯 causal 无上界、纯 padding 全掩行**不享受**本豁免。

### M10 复合掩码的单 vsel 约束与成分分解

- **s 上只允许一个 `tl.where`**：多 vsel 共存触发 vexp root-alloc 编译错。复合掩码（win | grow | gcol）**必须**先 OR 成单个布尔 keep，再一次 where。
- keep 的 `& n_mask` 写法在部分参数特化下同样触发 vexp root-alloc——pad 排除改走 M7-2 的算术减（`s −= pad_f*1e9`，有限 −2e9，exp 精确下溢 0）。
- 每行 self-window 保证至少一个有限 s > −1e9 > −2e9，max 归约不受 pad 污染。

### M11 fp16 参考链位级复刻三连（条件：参考链本身在 fp16 域，如 NPU `F.softmax(fp16)` 全链）

- **触发判据**：参考是 fp16 sdpa 链（`matmul→*scale→masked_fill→F.softmax→matmul` 全程 fp16）且精度排查 MERE 超阈值时。以下三处必须**逐位复刻**，任一偏差 MERE 1e-3~6e-3（fp16 rel_threshold=9.77e-4）：
  1. **round16 标量乘**：NPU 上 `fp16_tensor × python_float` 实为 round16(scale) 后 **fp16 域**乘；Triton 标量乘会提升 fp32 丢 post-scale 舍入 ⇒ `s = tl.dot(q,k).to(tl.float16); s = s * scale.to(tl.float16)`；
  2. **softmax d16 域减法**：NPU `F.softmax(fp16)` 的 `d=(s−m)` 在 **fp16 域**减法舍入，exp/Σ/除在 fp32 ⇒ `d = s16 - m[:, None].to(tl.float16); e = tl.exp(d.to(tl.float32))`；
  3. **p 归一化后舍入**：参考输出 p = round16(e/Σe)，**归一化后**才舍入 ⇒ online softmax 单边结构不可用，必须**三遍 KV + HBM scratch**：loop1 s16+滚动 max 暂存 scratch；loop2 重读累 `l=Σexp(d16)`；loop3 重读重算 e，`p16=round16(e/l)` 再 PV dot。
- **探针先行（强约束）**：改 kernel 前先用 ~20 行 torch 探针在 NPU 上逐位比对候选语义路径，拿 0-mismatch 证据再动 kernel；禁止"改 kernel→跑 verify→猜"循环。
- **Why**：op37 iter_0 9/50 → 三连逐个补齐 → iter_2 50/50。位级锁定后 Phase 4 零优化空间。

### M12 确定性掩码的物化分流：几何成分 kernel 算术化 / RNG·查表成分 host 物化 + cache

- **可算术推导成分**（带 / 块对角 / 元素周期 / 全局行列）→ kernel 内算术生成后 OR 合并单 vsel，零额外流量。
- **RNG / 查表成分**（per-row randperm 等）→ **host 物化**为 int8 稠密 `[SQ,SKV]` + 按 shape 参数 cache + kernel 单次 load + 单 `tl.where` 注入；随机列**直接 -inf 掩码，不用 gather**（MTE2 代价不划算）。**禁止**把 RNG 搬进 kernel（流不可复刻）；**禁止**每 forward 重建确定性掩码。
- **cache 语义保持依据**：seed 硬编码在参考 forward 内 ⇒ mask 是 shape 参数的确定函数，重建纯浪费。op37 参考每次 forward 重建 mask 占其 94~97.5% 时间，host cache 后 geomean **67.02×**。
- **RNG 三守**：seed(42) 在任何 randperm 前（对齐点在 forward 首行）；randperm 必须在 **NPU device**（CPU/NPU RNG 流不同）；零 RNG 分支（全局行、SKV≤NRAND）不调 randperm。

---

## §2 精度闸门（掩码/结构化偏置特有）

- **`-inf` vs 有限极小值**：门控掩码用 `-3.0e38`（fp32）+ `tl.constexpr`，不用 `-inf`（整行被掩 → `exp(-inf-(-inf))=NaN`）。本类 mask 更密，更易整行被掩。
- **整行被掩 fallback**：门控算子必须显式复刻参考的退化路径（如 `softmax→mean(V)`），不可简化为"全零行"。M-Q4 对角保底者豁免（M9）。
- **fp32 锁定判据**：参考全程 `.float()` **且** q/k 量级超 `tl.dot` 安全区 [-5,5]（op14 q/k~290）时，**全 fp32 中间 buffer 是唯一正确解**，不可降 native（op14 实测 fp16 QK 21/50 NaN）。fp16/bf16 输入时 QK 可原生 dtype 操作数+fp32 累加，输出逐步舍入回输入 dtype。
- **NaN/Inf 输入 sanitize**：参考含 `nan_to_num` 时**必须拆独立 elementwise kernel**——nan2num 的 vsel 与 flash kernel 内 mask vsel 共存触发 root-alloc 编译错。
- **怪异 scale 逐位对齐**：参考的缩放写法可能反常（如除以 `sqrt_d` 而非 `1/sqrt(d)`）——按参考语义逐位复刻，禁止"改正"。标杆更新会使例子失效，逐位复刻前必须重读当前参考源码。
- **RNG 复刻**：参考 forward 内动态建 `nn.Linear` 的，用 `manual_seed + uniform_(-1/√D, 1/√D)` 逐位复刻 kaiming_uniform_(a=√5) 的 RNG 时序，host 缓存 + 预转置。

---

## §3 各算子章节

### §3.1 14_WindowAttention（单窗口 MSA + 投影 + 加性相对位置偏置）

**生效层**：(a) host-batched 窗口稀疏 + (b) 加性 rel_bias
**形态**：`[B, N=Wh·Ww, C]`（窗口已切，N∈[49,361]）；2 投影 GEMM（qkv + out）；score 全稠密（无 `-inf`）；加性 rel_bias；fp32 锁定。
**结果**：0.7433× → **2.1968×**（50/50，target 2.2× 噪声内），impl 1.115ms → 0.3786ms（−66%）。0817 复跑：Phase3 3.3972 → opt_iter_0 **4.0054**，同架构复现。

**优化历程**：

| 轮 | speedup | 关键改动 | 结论 |
|----|--------:|---------|------|
| 基线 | 0.7433 | 修复投影 `tl.dot` 维度/dtype 双错 | 起跑 |
| opt0 | 1.6625 | **T1 投影 GEMV→GEMM** | **+124%** 最大单笔 |
| opt1 | 1.7801 | attn BLOCK_KV 放大(D≥128) | +7% |
| **opt2 ★** | **2.1968** | **T2 attn tile 推 UB(D-档) + T3 Cube 提前 + T4 并行度保护 + 投影 tile 放大** | **+23%** 夺冠 |
| opt3 | 2.0492 | 投影 BLOCK_M 64→128 | ❌ −6.7% R1 |
| opt4 | 2.0696 | rel_bias fp32→fp16 | ❌ −5.8% R2 |

**制胜技术（通用原理见 FA，此处只给实例）**：

**T1 投影 GEMV→标准 2D GEMM + 跨 batch M 展平（+124%）**：生成码把投影写成 GEMV（`BLOCK_M=1`），C=512 时 task 数 ≈35k，每个 task 只做 1×K 的 dot → **Cube 几乎空转**。改 2D GEMM（`BLOCK_M=64`）后 task 数砍 ~64×。**跨 batch 展平**：把 `B·N` 个 token 行拍平成 M 维，一个 GEMM kernel 吃完所有 batch×token，靠整数除还原 batch。**权重 host 预转置**（FA L1.12）。判据：`aic_mac_ratio` 极低 + 投影逐行循环 → 立即 GEMM 化。

**T2 attention tile 按 D 分层推 fp32 UB 上限（+23%）**（FA L1.13）：fp32 元素 4B，比 fp16 大一倍，同 BLOCK 溢出 UB，必须按 D 分档：
```python
BLOCK_D = _ceil16(D)
if   BLOCK_D <= 32:  BLOCK_Q, BLOCK_KV = 64, 128
elif BLOCK_D <= 64:  BLOCK_Q, BLOCK_KV = 32, 128
elif BLOCK_D <= 128: BLOCK_Q, BLOCK_KV = 32, 64
else:                BLOCK_Q, BLOCK_KV = 16, 32    # D=256
BLOCK_KV = min(BLOCK_KV, _ceil16(N))               # N 小时不超过 N
```

**T3 Cube 提前发射**（FA §4.7）：`pv = tl.dot(p, v_tile)`（Cube）提到 `p_sum = tl.sum(p)`（Vec）之前，Cube 与 Vec 重叠。

**T4 并行度保护（if 级联，禁 while）**：grid 过小（tile 数 < `NUM_CORES//2`）→ 核吃不饱 → 缩 BLOCK_Q 增 task。op14 把 FA 的并行度保护**扩展到 attention 侧 BLOCK_Q**——本类小 N 窗口 attention 极易 tile 数塌陷。

**两个陷阱（首次生成必避）**：

**P1 投影 BLOCK_K = `_ceil16(C)` → UB 溢出**：C=384/512 → x tile 128KB+ 溢出 UB，报 `MLIRCompilationError: multi-buffer extra local buffer`。**修复**：`BLOCK_K` 固定小值（64/128），`for c0 in range(0, C, BLOCK_K)` 自分块。

**P2 5D permute → `memref.collapse_shape` 编译错**：qkv 若 `view(B,N,3,H,D).permute(...).contiguous()` 再按 5D stride 访问，D=64/128 时报 `collapse_shape` 错。**修复**：扁平 `[B,N,3C]` 连续 buffer + stride 算术拆 q/k/v，零 permute。

**两个回归（证伪，机理必读）**：

**R1 投影 BLOCK_M 64→128（耦合 BLOCK_K 128→64）→ −6.7%**：放大 BLOCK_M 意在复用 w，但为不超 UB 须把 BLOCK_K 减半 → **K-loop 翻倍** → 循环/同步开销翻倍，**超过** w 复用收益。教训：BLOCK_M 与 BLOCK_K 被 UB 预算**耦合**，单维放大必然逼另一维回缩。

**R2 rel_bias fp32→输入 dtype → −5.8%**：rel_bias 体积小（~5K 元素），降 dtype 省的流量极少；但 cast 到 fp16 **额外 dtype 转换 + 512B 行对齐劣化**反而拖慢。教训：结构化加性偏置体积小，**降精度得不偿失**；保持在 softmax 累加域（M2）。

**关键约束清单（首次生成必守）**：① 投影禁 GEMV，标准 2D GEMM，跨 batch M 展平；② 投影 BLOCK_K 固定小值；③ qkv 拆分用扁平 buffer + stride，禁 5D permute；④ attention tile 按 D 分档推 UB 上限 + attention 侧并行度保护；⑤ fp32 锁定；⑥ ghost-column mask：`tl.where(mask_kv, scores, NEG_INF)` + `p = tl.where(mask_kv, p, 0.0)`。

---

### §3.2 23_LocalAttention（窗口带门控掩码 + 4 投影，带区间收缩）

**生效层**：(b) kernel 内门控掩码 `|i−j| < window`，带区间收缩实现 kernel 内块跳过
**形态**：`[B, S, C]`；4×`nn.Linear(bias=False)`（RNG 复刻+host 缓存+预转置）；参考 `masked_fill(带外, -inf)`；fp32 契约。
**结果**：首版即 **3.2810×**（50/50）；复跑 2.7557（芯片间 framework 波动）。

**制胜技术**：

**L1 window 带 KV 区间收缩（稀疏红利主来源）**：对每个 query 块，KV 扫描区间收缩为 `|i−j| < w` 的并集，带外整块**不 load 不算**：
```python
kv_lo = max(q_start - wm1, 0)
kv_hi = min(q_start + BLOCK_Q + wm1, S)
for kv_start in range(kv_lo, kv_hi, BLOCK_KV):   # 带内逐块 online softmax
```
S=4K、w=128 时 KV 扫描量砍 ~30×。

**L2 keep 掩码 fp32 比较 + 幽灵列显式排除**（M7-1 + M3）：`keep = (col_f >= row_f - wm1f) & (col_f <= row_f + wm1f) & mask_q[:, None] & (offs_kv < S)[None, :]`。⚠️ masked load `other=0.0` 的幽灵列 score=0 **会通过带判定**（0 落在 [−w, w] 内），必须 `& (offs_kv < S)` 显式排除。

**L3 w≥S 掩码失效特化**：`window_size ≥ S` 时带覆盖全行，keep 恒真——可 constexpr 分支省掉全部掩码算术。

**陷阱/弱项**：大 D fp32 case 0.84~1.03×（ieee dot 吞吐上限）——fp32 + D≥256 是结构性天花板。投影部分与 op14 同套。

---

### §3.3 24_BlockSparseAttention（块对角门控 + 4 投影，单 tile 一次性 softmax）

**生效层**：(c) 块级结构——块对角掩码（`block_ids[i]==block_ids[j]` 才 attend）
**形态**：`block_size∈[2,31]` 极小；4×`nn.Linear(bias=False)`；参考 `masked_fill(块外, -inf)`；fp32 契约（QK 原生 dtype 操作数+fp32 累加）。
**结果**：首版即 **3.6387×**（50/50）；复跑 Phase3 2.6048。

**制胜技术**：

**B1 program = 单掩码块（稀疏结构 = 并行粒度）**：grid 维 `(b, blk, h)`，每 program **只 load 本块的 Q/K/V 行**——跨块计算整块消除，不需要任何"掩码表"或索引 gather，块号即地址。参考语义 `masked_fill(块外, -inf)` 被**结构等价**为"只算块内"。

**B2 单 tile 一次性 softmax（M8 的首个实证）**：`BLOCK_M=32 ≥ max(block_size)=31` → KV 全集一个 tile 装下 → **取消 KV 循环**：
```python
s = tl.dot(q, tl.trans(k)) * scale          # [32, 32] 单 dot
s = tl.where(pair_mask, s, NEG)
m_new = tl.maximum(tl.max(s, 1), MCLAMP)    # 一次性 max，无滚动 m_i
p = tl.exp(s - m_new[:, None])
o = tl.dot(p, v.to(tl.float32)) / tl.maximum(tl.sum(p, 1), 1e-30)[:, None]
```

**证伪**：大 D 投影 GEMM tile autotune **−5.2%**——投影 4 连发已贴近 Cube 上限，扫参无油水。

---

### §3.4 25_StridedAttention（元素级周期掩码，无投影，dense 扫描+算术掩码）

**生效层**：(b) 元素级门控 `j%stride == i%stride`（含对角）——**无法块跳过**：BLOCK≥stride 时每个 KV 块内必含全部残基类
**形态**：纯 attention **无投影**；参考 `masked_fill(-1e9)`；fp32 契约。
**结果**：首版 2.4029（50/50）；复跑 2.4683 → opt_iter_0 **3.4325**（dtype 白名单化）。

**制胜技术**：

**S1 dense KV 扫描 + 元素算术掩码**：`mq = offs_q - (offs_q // stride_step) * stride_step`（禁 `%`，M7-4）；`keep = (mq[:, None] == mk[None, :]) & mask_q[:, None] & mask_kv[None, :]`。EQ 比较保持整型；掩码不物化稠密 [S,S]、不进 host。

**S2 p 不二次清零（省每 KV 迭代一个 vsel）**：掩码列 `scores=NEG(-3e38)`，`m_new ≥ MCLAMP` 时 `exp(NEG − m_new)` **精确下溢为 0** → 不需要 `p = tl.where(keep, p, 0.0)`。

**S3 dtype 白名单（+39% 主来源）**：fp16/bf16 输入走原生 dtype 操作数 + fp32 累加 → `BLOCK_KV=128, bq_budget=8192`；fp32 输入全 fp32 → `BLOCK_KV=64, bq_budget=4096`，另加 UB 估算守卫循环收缩 tile。

**陷阱**：**aicore 507015**：strided 行索引向量（`r + i*ST`）仅出现在掩码算术即触发 507015，shape 相关。规避 = dense `qi` + `qi < sr` 等价判定。**残基类分解两版证伪 −6%**：元素周期掩码的"稀疏红利"在 Ascend 上**不抵重组开销**，dense 扫描就是终态。

---

### §3.5 51_LongformerAttention（复合掩码 = 窗口带 ∪ 全局行列 + sanitize + out 投影）

**生效层**：(b) 复合门控 `win | grow | gcol`，`-1e9` 有限掩码；对角保底（M9 豁免 fallback）
**形态**：`[B,H,S,D]` BNSD 直入无转置；`nan_to_num` 预处理；out 投影；S∈[13,263] 小。
**结果**：0.4114 → **2.7685×**（50/50，6 轮含 1 IR 轮）。小 S case 4~5×，大 S·D fp32 case 1.0~1.7×。

**制胜技术**：

**G1 复合掩码成分分解 + 单 vsel**（M10）：
```python
offs_nf = offs_n.to(tl.float32)
win  = (offs_nf[None,:] >= (offs_m-w2).to(tl.float32)[:,None]) & \
       (offs_nf[None,:] <= (offs_m+w2).to(tl.float32)[:,None])   # 仅 2D 比较转 fp32（G2）
grow = ((offs_m == 0) | (offs_m == g1))[:, None]                 # 1D EQ 保持整型
gcol = ((offs_n == 0) | (offs_n == g1))[None, :]
keep = win | grow | gcol
s = tl.where(keep, s, -1e9)      # s 上唯一一个 vsel
```

**G2 i32 2D 窗口比较标量降级 → fp32 比较**（M7-1，+6.6×）：`.to(tl.float32)` 后走向量 fcmp。**全量转换（含 1D）反而触发 M7-2 编译失败**——只转 2D 比较项。

**G3 pad 列算术排除**（M7-2/M7-3）：`pad_f = tl.minimum(tl.maximum(offs_nf - S + 1.0, 0.0), 1.0)`；`s = s - pad_f[None, :] * 1e9`。相比 `tl.where(n_mask, p, 0)` 同时消除 n_mask 张量物化的 i64 标量循环 + 临时量 + 第二个 vsel。

**G4 sanitize 独立 elementwise kernel**（§2）：`nan_to_num` 与 flash kernel 内 mask vsel 共存 → root-alloc 编译错，必须拆出单 kernel。

**flaky 判据**：长 verify 进程内偶发单 case root-alloc/NaN 为 flaky（重跑可过）；**22/50 级别大面积同指纹失败 = 确定性毒性，立即放弃该方向**。

---

### §3.6 37_BigBirdAttention（复合掩码+随机 gather，host 物化 cache，fp16 位级锁定三遍 KV）

**生效层**：(b) 复合门控 = window 带 ∪ 首/末全局行列 ∪ per-row randperm 随机列；对角保底（M9 豁免 fallback）
**形态**：BNSD 直入纯 attention（无投影）；fp16 参考链。
**结果**：Phase 3 三轮收敛（iter_0 9/50 fp32 中间域 → iter_1 40/50 补 round16(scale) → 47/50 补 softmax d16 → iter_2 50/50 补三遍 KV），iter_2 首过即 **67.0198×**（50/50）；Phase 4 数值锁定零优化。

**制胜技术**：

**BB1 整 mask host 物化 + 参数化 cache（67× 主来源，M12 命名案例）**：window/global/random 三成分全部 host 侧构建为 int8 `[SQ,SKV]`，按 `(SQ,SKV,W,NRAND,device)` cache；kernel 内一次 load + 单 `tl.where` 注入。**随机列直接 -inf 掩码，不用 gather**。67× 构成（诚实口径）：framework 每次 forward 重建 mask 占其 94~97.5%；剔除掩码重建后的纯 attention 对比约 13~33×。同型算子直接复制本架构，勿在 kernel 内找优化空间。

**BB2 RNG 复刻三守**（M12）：seed(42) 在 forward 首行；randperm 在 NPU device；全局行与 SKV≤NRAND 分支零 RNG 消耗。

**BB3 fp16 位级锁定三连 → 三遍 KV + HBM scratch**（M11 命名案例）：loop1 算 s16 + 滚动行 max，暂存 HBM scratch；loop2 重读累 `l=Σexp(d16)`；loop3 重读重算 e，`p16=(e/l).to(fp16)` 再 PV dot。探针先行方法论锁定语义后才动 kernel。

**陷阱/证伪**：Phase 4 零优化（位级锁定后 31 优化点无一可动；simulator 实测 MMAD 仅 5.7%、热点行全为精度必需 fp16 链——**结构极限，勿硬攻**）。

---

### §3.7 105_FlexAttentionBwd（**本卡首个 backward 成员**：lse 重算 p 免 delta kernel，4 kernel 双路径 + WS 物化 fp32 cast）

**生效层**：(b) causal 门控（运行时属性 → constexpr 特化）；参考仅 causal + GQA
**路由注记**：按写法命中路由表行 7（FA 分块流式反向主链）＋ 行 4（GQA 交叉），实际由 `flash_attention`＋`gqa` 双卡生成；经验回填本卡是因为 **FlexAttention 家族（mask_mod 通用掩码 API）前向属行 6b**，其 backward 骨架对本卡全族成员的反向写法复用。
**形态**：纯 attention backward（无投影）；**lse 已知 → 免 delta kernel、免 online 递推**（`p=exp(s-lse)` 直接重算）；精度契约 = 全 fp32 ieee（fp16/bf16 case 同）。
**结果**：首过 0.7168（50/50）→ 两轮 tile 放大 0.9419（+31.4%）/ 0.9772 → IR 轮 WS 物化 **1.0705**（+9.5%）；target 未达，结构性瓶颈 = fp32 ieee cube 吞吐 vs torch 参考走 CANN fp32 优化库。

**制胜技术**：

**FB1 反向 4 kernel 双路径（本卡 backward 骨架基线）**：K1 dq（q 主序 persistent grid-stride，delta 内联，`WRITE_DS` constexpr 可选把 ds **转置**写 WS）；K2 dk（3D grid 纯 GEMM，直读 WS 转置布局）；K3 dv（3D grid 重算 p，转置态一步算出，免 trans(p) 物化）；K4 dkdv 融合（Sk<256 小 S 路径）。host 门控 `USE_WS=(Sk>=256)`。

**FB2 tile 放大到 UB 上限 + 并行度保护（+31.4% / +3.7%）**：BKV 放大后 3D grid 任务数 `< ncores//2` 时回退——大 tile 挤掉并行度比 tile 效率更伤。

**FB3 ★ WS 物化 fp32 cast（IR 证据链驱动，+9.5% 最大单轮增量）**：IR 实锤 mix kernel 里 `tl.load(fp16).to(tl.float32)` 的 vcast 经 GM 回退喂 cube——每个 cube 操作数的 cast 都是一次 UB→GM→L1 全程往返；且 K2/K3 是 kv 主序 3D grid，同一 q tile 被 `cdiv(Sk,BKV)` 个 program 各自重复 load+cast。改法：K1 顺手把 `q_t`/`do_t` 的 fp32 cast 结果物化到 WS fp32，K2/K3 改直读 fp32。**数值 bit 级一致**。

**FB4 warmup-only IR 提取法（debug 二进制崩溃的规避）**：debug 编译标志下本算子 K2 二进制运行期崩 507015（生产正常）→ monkeypatch `JITFunction.run` 强制 `warmup=True`（只编译不发射）。提取前清理 `/tmp` 下他人遗留的陈旧 IR dump。

**陷阱/证伪**：causal WS `torch.zeros→empty + 掩码 load` **证伪回退**——"+23~25ms"源自非紧邻自然实验被分段机器漂移污染，时间紧邻复测 memset≈0。**判定铁律：A/B/A' 时间交错**。

---

## §4 待后续算子填充（骨架占位）

### §4.1 35_AxialAttention（轴向分解，无掩码）✅ 0818 已落地

- **生效层**：(a)——2D 轴向稀疏被 host reshape 消费为**两次稠密小 N attention**（沿 H 轴 + 沿 W 轴，输出相加）。kernel 内**无任何掩码**。
- **终态**（iter_2 首过 2.4269 → 分档 2.4388，50/50）：**4 kernel 6 发射**——K0 layout_copy（X→X_PERM 纯搬运）→ K1 qkv_proj_gemm **单发射吃两轴**（两轴投影对同一 token 集算两遍值相同）→ K2 axial_attn ×2（AXIS constexpr，BLOCK_S=32 ≥ max(H,W)=31 单 tile softmax，M8）→ K3 out_proj_gemm ×2（ACCUMULATE 0 覆盖写 / 1 加写，逐字复刻两次 F.linear 舍入时序）。
- **权重胶水**：host 预拼接 `w_qkv_t[C,3I]` + `w_out_t` 预转置，按 data_ptr 缓存（FA L1.12）。
- **dtype 契约**（低精度逐步舍入链）：s16 = round(dot) → 低精度域乘 scale → 幽灵列 vsel → fp16 **必须 fp32 减 max + 显式 `.to(f16)` 舍入**（隐式舍入被编译器消除，M11-2）；p 归一化后舍入再 PV dot。
- **有效杠杆**：仅大 M（≥1024）BM 64→128（K1/K3 联动，+0.49%）。弱 case 全为大 M 大 C GEMM 主导（接近硬件极限）。
- **simulator 终证**：K1/K2/K3 MMAD 仅 10-19%，MTE2 装载 43-50% 主导 = 小 M 权重流带宽墙，非 Cube 瓶颈；tile 平衡点 128/128 实证。

### §4.2 causal / padding `-inf` 掩码注意力（TODO）

- **生效层**：(b)；运行时可变 → 大杠杆在 KV 区间收缩 + mask 分支 constexpr 特化。
- **必守**：M2（有限极小值）、M3（整行被掩 fallback——causal 首行/纯 padding 全掩行**不享受** M9 豁免）。

### §4.3 topk-selected KV 注意力（TODO）

- **生效层**：(c)；杠杆在索引 gather（只 load 被 topk 选中的 KV 行）。

### §4.4 真·shifted-window（带 `attn_mask` 参数的 SW-MSA）（TODO）

- **生效层**：(b)；与 op14 区别——掩码是 kernel 内可见运行时 mask，可用 mask 特化。

---

## §5 证伪方向（跨成员，持续积累）

> 继承 FA 证伪表全表；以下为本类**新增**。

| 方向 | 结果 | 来源 |
|------|------|------|
| 对 host-batched 稀疏算子套块跳过 / mask 特化 | **no-op**（稀疏已不在 kernel 内） | op14（M1） |
| 加性 rel_bias 降到输入 dtype | **−5.8%**（转换+对齐劣化抵消流量；偏置体积小不值得） | op14 R2 |
| 投影 BLOCK_M 放大（耦合 BLOCK_K 回缩） | **−6.7%**（K-loop 翻倍主导；M/K 被 UB 耦合，单维放大可净负） | op14 R1 |
| 投影 BLOCK_K = `_ceil16(C)` | **编译失败**（x tile 128KB+ 溢出 UB） | op14 P1 |
| qkv 5D permute 拆 q/k/v | **编译失败**（`collapse_shape`） | op14 P2 |
| 把加性偏置当门控掩码"跳块" | 概念错（加性不改分母，无块可跳） | M-Q3 |
| strided 残基类分解（packed+gather / 自然布局 store） | **−6% 两版**（重组开销对冲稀疏红利） | op25 |
| 大 D 投影 GEMM tile autotune（4 连发已贴 Cube 上限） | **−5.2%** | op24 |
| 优化点 6 全量 fp32 转换 / 1D-mask-only 转换 | **22/50 确定性编译失败**（sitofp+fcmp root-alloc 毒性） | op51（M7-2） |
| nan_to_num 与 flash kernel 内 mask vsel 共存 | **root-alloc 编译错** → 拆独立 sanitize kernel | op51（G4） |
| s 上多个 tl.where（复合掩码逐项注入） | **vexp root-alloc 编译错** → OR 合并单 vsel + pad 算术减 | op51（M10） |
| strided 行索引向量参与掩码算术 | **aicore 507015**（shape 相关）→ dense qi + `qi < sr` 等价 | op25 |
| 大 D fp32 case 硬攻（ieee dot 吞吐上限） | 0.84~1.03× 结构性天花板 | op23/op51 |
| 对 per-row 随机掩码强行块跳过 / 带区间收缩 | **不适用**（逐行列集不同，无共享块网格；dense 扫描即终态） | op37 |
| fp16 位级锁定链上做 Phase 4 kernel 优化 | **0/31 命中**（MMAD 5.7%、热点全为精度必需——结构极限） | op37 |
| K^T 转置直载消 vtranspose（小 tile 32×128） | **−1.6%**（跨步转置 load 的 MTE2 代价 > AIV 小 transpose 代价） | op35 |
| softmax p 归一化 divf→倒数乘 | **−1.95% 无收益**（预期 ≤1% 被会话噪声淹没） | op35 |
| MTE2_high 下 fp16/bf16 big 档 BM=256/BN=64 | **−3.4%**（BN=64 ⟹ N_tiles 翻倍；128/128 是实测平衡点） | op35 |
| msprof op simulator 带 `TRITON_DEBUG/ALWAYS_COMPILE` | **app signal 11 崩溃** → 采集前 unset 这两个 IR 提取遗留变量 | op35 |
| backward causal WS `torch.zeros→empty + 掩码 load` | **证伪回退**（memset≈0；"+23~25ms"是非紧邻对照被漂移污染的假象） | op105 |
| backward K1 dq D=128 档 BQ 32→64 扩 tile | **UB 逼死**（dq_acc+q/k/v/go 全 fp32 tile 超 192KB） | op105 |

---

## §6 测量口径与时效

- **几何平均**：每 case 权重相同。小 N / 块稀疏最弱 case 多，**不能只优化大 case**——op14 把 laggard 从 <1× 拉到 >1.2× 是夺冠关键。
- **芯片间波动**：同一最终代码 0815 ↔ 0817 geomean 可差 ±10~30%，主因 framework 侧噪声。**优化判定一律用 impl_ms 紧邻 A/B**，禁用跨日 speedup 对比。
- **同芯片同时段漂移**：同代码两次全量 benchmark 的 geomean 差 **±2.3%**；分段性漂移可在连续 case 段落叠加 **±20~30%**。**非紧邻的"自然实验"完全不可信**；判定必须改动代码与同代码复跑时间交错（A/B/A'），或靠内置对照组佐证机器稳定。
- **历史结论随编译器/CANN 失效**：本文件所有数字标注来源算子与日期，**判据/机理优先于绝对数字**。
