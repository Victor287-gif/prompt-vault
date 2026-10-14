---
type: Guide
title: gqa 类算子设计约束（二·压缩类·head 共享）
description: GQA/MQA 家族（head 共享、router 选路、含投影 GQA，以及与块跳过/页表寻址叠加形态）的经验：判别特征、Layer 1
  硬约束（任务单元按 kv_head 分组 / 掩码在 scores 上 / tile 禁切片）、骨架与技巧、证伪方向。继承 flash_attention_constraints.md。
tags: [task.design, topic.attention, topic.compute.composite, topic.implementation.masking, topic.tiling, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-20T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# gqa 类算子设计约束（二·压缩类·head 共享）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)；稠密 FA 主链约束继承
> [flash_attention_constraints.md](flash_attention_constraints.md)；家族内写法类路由见
> [attention_writing_classification.md](attention_writing_classification.md)。

本卡负责 **二 压缩类 · 细分 1「head 共享」**：取数层的改动只有 `kv_head = q_head // group` 这一处。
该细分下的算子是 **21 GroupedQueryAttention · 22 MQA · 33 AdaptiveAttention**（33 是门控选路变体，
按样本 argmax 选一路权重，主链仍是 head 共享）。含投影的 `gqa-proj` 形态也在本文件。

> **与 [block_sparse_attention_constraints.md](block_sparse_attention_constraints.md) 的分工**：那份卡片负责 **三 稀疏类 · 细分 2「块跳过/top-k」**
> （`sp-blockmask` / `sp-paged-topk` / `sp-pooled-topk` / `sp-bwd`），即**选择表怎么建、怎么按压缩索引跳块**。
>
> **叠加形态两份都读**：GqaSparse 系是「三·2 块跳过 ＋ 二·3 页表寻址 ＋ head 共享」三者叠加，
> **选择层与跳块看那边，head 分组的任务划分、页内数据布局、tile/UB 预算看这边**。
>
> **与 [paged_attention_constraints.md](paged_attention_constraints.md) 的分工**：那份卡片负责 **二·3 页表寻址·不跳块** 的纯形态
> （整段 KV 都要算的 paged decode/prefill，op37/38 实证：block_ptr 红线 / helper op 顺序
> 触发流水线 / aux_mask 表 load / split-q persistent）。本文件的分页条目只服务于行 1 交叉的
> 块跳过叠加形态，并仍作为其页内布局（§3.4）与 tile/UB 预算（G3/§5.2）的补充参考。

- **§1 通用经验 G1-G7** · **§2 Layer 1（L1.1~L1.11）** · **§3 Layer 2 骨架** · **§4 Layer 3 技巧** · **§5 优化点与证伪** · **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与 FA 的分工**

> **证据基础**：四个 GQA 类算子各 50 case 的完整优化轨迹，同一台 910B2C（24 aicore / 48 vectorcore）、
> triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0 / bishengir-compile 1.2.0：

| 算子 | 子类标签 | 形态 | **最终 speedup**（新口径） |
|---|---|---|---|
| `21_GroupedQueryAttention` | `gqa-proj` | 4 个投影 GEMM + 稠密 GQA | **2.5119** |
| `33_AdaptiveAttention` | `gqa-router` | router MLP 选 1 路，N∈{2,3} 路完整 GQA | **4.0101** |
| `80_GqaSparseFwd` | `gqa-sparse-fwd` | 块稀疏 GQA 前向，分页 KV | **59.5558** |
| `81_GqaSparseDecode` | `gqa-sparse-decode` | 块稀疏 GQA decode，topk 页 + block_table 间接寻址 | **41.9099** |

四者精度均 **50/50**。两栏差异是**口径变了，不是代码变了**（§7.2）；正文里出现的 speedup 除本表外一律是旧口径，只用于方案间横向比较。

> ⚠️ **核心优化哲学**：与 FA 类一致——瓶颈是 **Cube↔Vector 跨核同步**，不是算术量、不是访存、不是 dtype。
> 但本类多了一条 FA 没有的硬约束：**KV 按页/块间接寻址，页与页之间不连续**，
> 于是「放大 tile 摊薄同步」这条 FA 的万金油**在本类算子上会直接撞 UB 墙**（§5.2 有完整实测）。
> 本类算子的目标函数是 **「循环体内 `tl.dot` 的个数 × KV 页迭代总数」**，**而不是** tile 的大小——
> 这一条与 FA 类的直觉相反，是本文件最重要的一句话。

---

## §0 适用范围与算子分类

| 子类标签 | 判别特征 | 优化重心 |
|---|---|---|
| `gqa-proj` | `H_Q > H_KV` 的稠密 GQA，**自带 4 个 `nn.Linear` 投影** | **最大的一笔在投影 GEMM 的 tile 分档**（实测 +57%），不要一上来就攻 attention |
| `gqa-router` | 前置一个小 MLP，`argmax` 选 1 路，多路完整 attention，最后 one-hot 加权求和 | **只算被选中的那一路**（实测 nopts 倍收益）；难点全在精度与编译，不在性能 |
| `gqa-sparse-fwd` | KV 按块稀疏跳过，分页 cache | 页内连续 + 跳过判据外提 |
| `gqa-sparse-decode` | 每个 query token 由 `topk_idx` 选 K 个页，`block_table` 二级间接寻址；`S_Q` 退化成 1 | **任务单元必须按 kv_head 分组**（实测 6.18x）；之后成本按页线性，杠杆极少 |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件，**不要**落到 FA 卡：

1. KV 通过 `block_table` / `page_table` / `topk_idx` 等**间接索引**取，页与页在显存中不连续；
2. attention 主链被一个**路由/门控**选择性执行（`argmax` + `one_hot` 或等价形式）；
3. KV 维度上存在**块级跳过**（block-sparse），而不是一段连续区间的 causal / window 收缩。

只有稠密 GQA（KV 是连续区间，靠 causal/window 收缩）→ 用 FA 卡的 `fa-gqa`。两者都命中 → **两个文件都读**，本文件的 §2/§5 优先。

### §0.2 ★ 形态识别五问（设计第一步必须回答）

| # | 问题 | 影响 |
|---|---|---|
| **Q1** | KV 是连续区间还是按页间接寻址？ | 间接 ⇒ §2 L1.2 + §5.2 的 UB 墙全部适用；连续 ⇒ 回 FA 卡 |
| **Q2** | 有没有独立的投影 GEMM？ | 有 ⇒ §4.1（tile 按 dtype 分档）是最大的一笔 |
| **Q3** | 有没有路由/门控选路？ | 有 ⇒ §4.4（分支门控）理论收益 = 分支数；但**必须先做 §6.4 的 argmax 边距体检** |
| **Q4** | 参考实现的 attention 在什么精度上算？ | 决定 `p` 能否降精度、以及要不要二段拆分（§6.2） |
| **Q5** | 评测口径？ | 几何平均 ⇒ **每个 case 等权**。小 case 极多，把一个 0.5x 拉到 1.0x 与把 8x 拉到 16x 贡献完全相同 |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### G1 ★ 环境闸门必须前置，否则整片结论作废

1. **非交互 shell 读不到 `~/.bashrc`**（前几行 `[ -z "$PS1" ] && return` 守卫使 `source set_env.sh` 永不执行），静默跑在旧 CANN 上。实测：同一份代码在 CANN 8.5.1 + bishengir 0.1.0 上 **0.9438**，在 CANN 9.1.0 + bishengir 1.2.0 上 **2.2904**。⇒ **每轮测量前显式 source，并把 triton / triton_ascend / CANN / bishengir 四个版本写进产物**。
2. **卡被自家 kernel 打坏**（`aicore timeout 507014`）后进入**跨进程持久**损坏状态，此后一切精度结果都是噪声。用一行**与被测算子无关**的算子对拍 CPU：`c = (a.npu() @ b.npu())`（好卡 `absmax≈162.7`，坏卡 `absmax=5.67e14` 或 `nan`）。⇒ **精度异常时第一件事不是查代码，是查卡**，且要**反复查**（共享机器上中途会被别的会话打坏）。

### G2 tile 预算随编译器版本整片变动，结构性改动后必须重扫

同一份 kernel：bishengir 0.1.0 上预算约 `2^18` 且报误导性的 `root alloc` 错；1.2.0 上预算约 `2^20` 且报老实的 `UB overflow`。⇒ **不要查历史 tile 清单**，真判据是 UB 字节数。

### G3 UB 是本类算子的第一约束，动手前先记账

910B2C 单核 UB **192KB**。**必须把 `tl.trans(x)` 的物化算进去**（额外占一份与 x 等大的空间）。一个满预算例子（`gqa-sparse-decode`，BS=128/D=128/bf16）只剩 **4KB**——这解释了为什么该算子上一切扩张全部报 `PlanMemory Failed`。

> ★ **先读报错里的精确数字，再决定改多大。** `ub overflow, requires X bits while Y bits available`，**两者之差就是你要省出来的量**。一例只超 512 bit = 64 字节（0.03%）——去掉一个中间变量就够了，**不要一看到 UB overflow 就把 tile 砍半**。真正需要结构性改动的是**差一个数量级**的情形。

### G4 增益不叠加，先算账再动手

每个方向动手前先估：`收益 = 省下的每次迭代成本 × 迭代总数 / 当前总时延`。两个反复出现的账：访存冗余（实测合计页读 18398 / 去重 17310 = 冗余仅 1.06x，直接否掉）；理论天花板（把 DMA 成本归零后的时延，仍达不到目标说明方向根本不在访存）。

### G5 扫参子集必须含大/中/小三档

本类 case 分布普遍是「小 shape 极多 + 少数大 shape 吃掉 80% 耗时」，而评分是**几何平均（等权）**。只在大 case 上调参会得到与最终分数相反的结论。

### G6 ★★★ KV 维 padding 必须掩在 **scores** 上，**清零操作数解决不了、反而正是错因**

**实测：一个算子连续两轮 0/50 栽在这里。** `other=0.0` 把 K 的 padding 行变成 0 向量，于是那些列的 score 恰好是 **0**——而 0 在 softmax 里**不是**"无贡献"，`exp(0 - m)` 往往还是最大的那批，`l` 被系统性放大、输出被整体缩小。

**症状指纹**：impl 与 framework **同号但整体偏小**，比值在 0.3~0.7 之间且随 `S % BLOCK_KV` 变化。⚠️ 这个误差量级会被误判成"权重复刻错"。

```python
# ✅ 掩在 scores 上，用有限极小值
kv_valid = offs_kv < S
scores = tl.dot(q_tile, tl.trans(k_tile)) * inv_sqrt
scores = tl.where(kv_valid[None, :], scores, -3.0e38)
```

四条配套：① 操作数仍要清零（保证 dot 归约不吃脏数据）——但它**只保证 dot 正确性，不构成 softmax 掩码**；② 用**有限极小值**不用 `-inf`（整块 KV 无效时 `max=-inf`，`exp(-inf-(-inf))=NaN`）；③ `m` 初值用 `-1e30`；④ **出口除法必须 guard**：某 q 行 KV 列全被掩时 `l==0` ⇒ `acc/l` 是 **0/0=NaN**，`out = acc / tl.where(l > 0, l, 1.0)`。症状是 **Implementation 有大量 NaN 而 Framework 一个都没有**——出现时先查这里。

### G7 forward 里的宿主侧代码受 AST 校验器约束

- `ast.For` 在 forward 里是**无条件违规** ⇒ 多分支要**展开成 `if`**；
- `self.xxx(...)` 一律被判「疑似 nn.Module 前向调用」⇒ 辅助函数写成**模块级函数**；
- `torch.cat` **不在**白名单；拼权重要用 `torch.empty` + `.copy_()`；
- 允许的 torch 构造：`empty/zeros/ones/full/tensor/arange/linspace/as_tensor`；允许的张量方法：`contiguous/to/view/reshape/permute/transpose/t/copy_/fill_` 等。

---

## §2 Layer 1: 设计约束（首次生成就要全部满足）

### L1.1 ★★★ 任务单元：**decode 按 kv_head 分组；prefill 必须按 query 分块**

> ⚠️ **先看适用范围，照搬会亏 4.4 倍。** 本条只对 **`gqa-sparse-decode`**（`S_Q == 1`）成立。**把它套到 prefill / 含投影的稠密 GQA 上是重大负优化**：实测把任务单元写成 `total_tasks = B * KH * S`，精度 50/50 但 **speedup 只有 0.5681**（差 4.4 倍）。

| 子类 | 任务单元 | q tile |
|---|---|---|
| `gqa-sparse-decode` | `(token, kv_head)` ← 本条 | `[BLOCK_G, D]`，BLOCK_G = 该 kv_head 的 G 个 q head |
| `gqa-proj` / `gqa-router` / `gqa-sparse-fwd`（prefill） | **`(b, q_head, q_block)`** | `[BLOCK_Q, D]`，**BLOCK_Q 必须按 UB 预算取到上限** |

**⛔ prefill 下 `total_tasks = B * KH * S` 是硬性禁止的**。**`S_Q > 1` 时看到 `tl.arange(0, BLOCK_G)` 出现在 q tile 上，就是错的**。

**以下仅适用于 decode。这是该子类最大的单笔收益（实测 0.2233 → 1.3791，6.18x）。**

坏味道：kernel 里出现 `kh = h // G`（或 `h * KH // H`），而循环变量是 **q head**：
```python
# ❌ 同一个 kv_head 组内的 G 个 q head 各自把同一份 [BS, 2D] KV 页重新读一遍
for head_idx in range(pid, total_q * H, num_cores):
    kh = head_idx // g_size
    ...  # 读 KV 页

# ✅ 一个 program 处理整组 G 个 q head，q tile 由 [1, D] 变成 [BLOCK_G, D]
pid  -> (t, kh)
q_tile = load Q[t, kh*G : (kh+1)*G, :]      # [BLOCK_G, D]
```
三重收益：① KV 访存降到 **1/G**（主导 case 全是 G=16）；② QK dot 的 M 从 1 变成 `BLOCK_G`（Cube fractal 粒度）；③ PV 从向量广播归约变成一次 `tl.dot`。**可迁移判据**：任何 GQA/MQA 算子，只要 kernel 里出现 `kh = h // G` 且循环变量是 q head，就一定在按 G 倍重复读 KV。

### L1.2 ★★★ 禁止「运行期标量除 + 向量下标 gather」组合——会让 aicore 挂死

```python
# ❌ 实测在 bishengir 1.2.0 上直接 aicore timeout 507014，并把整块卡打成持久损坏
b_idx = offs_m // S                      # S 是运行期标量
sel   = tl.load(sel_ptr + b_idx, ...)    # 按向量下标离散 gather

# ✅ 把 b 提升为 tile 坐标之一，sel 退化成一次标量 load
pid -> (b, s_tile, n_tile)
if tl.load(sel_ptr + b) == BRANCH:  ...
```
同源坏味道还有 `a % b`（运行期标量取模）——用 `a - (a // b) * b`，或除数是 2 的幂时用位与。

### L1.3 ★ `tl.dot` 的个数是成本单位，**不是 tile 的大小**

这是本类算子与 FA 类最反直觉的差异，**三轮独立实测互相印证**：`gqa-sparse-decode` 上每个循环内的 `tl.dot` 自带一份约 2 us 的固定 CV 握手，与 tile 大小、与数据依赖关系无关（B 轮把 PV 错开一轮，`serial_ratio` 几乎没动 ⇒ 握手是**结构性插入**的）。

> ⚠️ **但这条不能跨子类外推。** 在 `gqa-sparse-fwd` 上实测结论**相反**：`dot(p_hi,v)+dot(p_lo,v)` 双原生 dtype（3 dot）0.3086ms / 55.33，而 `dot(p_f32, v.to(f32))` 单 fp32 dot（2 dot）0.5491ms / 27.97——dot 数少的那个慢 1.78 倍（fp32 dot 不走 Cube 原生通路 + 每页多一份 f32 转换缓冲）。
⇒ **正确用法是先测再定**：一次 A/B 确认本算子属于哪一侧（握手主导 ⇒ 压 dot 数；MAC/转换主导 ⇒ 保原生 dtype 宁可多一个 dot）。判别信号：`aic_mac_ratio` 极低（0.02~0.05）且 `serial_ratio` 接近 2 ⇒ 握手主导。

### L1.4 成本模型必须先反推出来，再谈优化

`kernel_us ≈ (每页代价 a × 总页数 + 每 program 启动 b × program 数) / 核数`，误差 <10%。实测（`gqa-sparse-decode`）：**a = 5.81 us/页，b = 0.71 us/program**。a 的构成：两个 dot 的固定 CV 握手 ≈ 4.1 us + DMA 1.69 us + MAC ≈ 0.05 us。⇒ 启动开销可忽略，成本全部按页线性；`aic_mac_ratio` 只有 0.02~0.05，Cube 基本没在算。**这个账要在动手前算。**

### L1.5 ★ `tl.dot` 契约：原生 dtype 操作数 + fp32 累加

- q/k 从低精度内存加载 ⇒ **保持原生 dtype**，升 fp32 不增加有效位；
- ★★★ **两操作数必须显式 cast 到同一 dtype**——最常见的诱因是 `tl.where(mask, x, 0.0)` / `tl.full(..., 0.0)` 里的 Python 浮点字面量把整块 tile 提升成 fp32，而另一侧还是 fp16/bf16。补零与 cast 要一起做：`q_tile = tl.where(qd_mask, q_tile, 0.0).to(k_tile.dtype)`。
- ⚠️ **fp32 `tl.dot` 的适用条件要收窄**：满足 ①操作数是 fp32 ②输出宽度 `N == 64` ③位于**运行期 trip-count** 的 `scf.for` 内时被 bishengir 1.2.0 **误编译**——症状是输出**恰好一半、且是偶数列** NaN。打破任意一条即规避。**可迁移方法**：低精度输出的 NaN/垃圾，先看**列的分布**。

### L1.6 ★ 投影权重禁止以 `nn.Linear` 原始 `[out, in]` 布局喂给 GEMM kernel

> ⛔ **但不要过度套用——只有喂给 `tl.dot` 的权重才需要预转置。** 实测一例：`gqa-router` 的 `w2`（router 第二层）是唯一不能转置的，因为 `router_argmax` 用的是**向量点积** `tl.sum(w2 * hv[None, :], axis=1)` 而非 `tl.dot`。把它 `.t()` 后**布局与 kernel 索引不匹配** ⇒ 选错分支 ⇒ 相对误差恰好 **~1.0 量级**（与"权重复刻错"症状相同但根因不同）。

喂给 `tl.dot` 的那几个：`nn.Linear.weight` 是 `[out, in]`，`y = x @ Wᵀ`。直接喂会让 b tile 最内轴 stride = `in_features`，变成离散 gather。**host 侧预转置成 `[in, out]`**，kernel 内最内轴 stride == 1，且省掉每次迭代的 `tl.trans`。拼接用 `torch.empty` + `.copy_()`（`torch.cat` 不在 AST 白名单）。

### L1.7 Grid 收缩到 `min(核数, tasks)` + 核内步长循环，且 grid 与步长必须一致

```python
grid = (min(NUM_AICORE, total_tasks),)        # NUM_AICORE 实测 24，不要硬编码 14
kernel[grid](..., NUM_CORES=grid[0])          # 必须把同一个数传进去
```

### L1.8 ★★★ host 侧禁止 `permute(...).contiguous()`——输入输出都不许

参考实现里的 `transpose(1, 2).contiguous()` 会变成一个独立的 `Transpose` kernel（实测 19.46 us）。**实测代价：同一个算子，tile 完全相同，只因为多了 4 处 `.contiguous()`，impl 从 0.0715 变成 0.1287 ms（慢 1.8 倍），speedup 2.5119 → 1.4420，从达标掉到不达标。**
**自查**：forward 里出现 `.contiguous()`、`.permute(...).contiguous()`、或对 kernel 输出再做一次重排——都要问「这次搬运能不能用 stride 消掉」。本类算子里答案几乎总是「能」。

### L1.9 ★★★ 权重复刻：**先对拍再写 kernel**，否则精度全崩且症状会骗人

**判别**：golden 的 `_layers()` / `__init__` 里出现 `torch.manual_seed(42)` + `nn.Linear(...)` ⇒ 权重由 RNG 现场生成，实现侧必须**逐位复刻**。

**按误差量级分两类，只用来决定"先查哪一边"**：MERE 刚刚超阈（1.4e-2 vs 7.8e-3）⇒ 精度余量不够，查 `p` 降精度；相对误差 **0.5~1.5 量级** ⇒ 结果整个是错的，先查权重复刻。⚠️ 第二行**不能反推成"一定是权重错"**（实测反例：权重逐位相同，真正的错在计算侧）。**权重对拍是"秒级、零成本、能一次性排除一整条链路"的第一步，不是答案本身。**

可直接套用的配方（已在两个算子上 50/50 验证）：用 `torch.Generator()` 独立 RNG（不碰全局），`g.manual_seed(42)`，CPU fp32 采样，`kaiming_uniform_(w, a=math.sqrt(5), generator=g)` 逐字复刻 `nn.Linear.reset_parameters`，调用顺序/次数与 golden 书写顺序一一对应，`bias=False` 时不抽 bias，最后 `.to(device, dtype)`。

四个必须与 golden 一致的点，**漏一个就全崩**：① 采样在 CPU、fp32；② 调用顺序与次数；③ `bias=False` 时不抽 bias；④ `.to(device, dtype)` 放最后。⇒ **写 kernel 前先花一分钟在 CPU 上对拍**（`torch.equal(ref.weight, mine)` 必须逐位相等，不是 allclose）。

### L1.10 ★★★ tile **不能切片、不能下标赋值、不能拼接**——这是首版生成的头号杀手

**实测：两个算子的首版生成 50/50 全挂，根因都是这一条。** Triton 的 tile 不是 numpy 数组，**没有任何形式的部分读写**。

| 你想做的事 | 正确写法 |
|---|---|
| 从一块 `[BS, 2D]` 的 tile 里分别拿 K 和 V | **不要切**。发**两次 `tl.load`**，各自用自己的指针偏移（`+0` 和 `+D`）；或走 §3.4 整页方案（`dot(q补零, kvᵀ)` 与 `dot(p, kv)` 的列掩码 + 指针偏移 `-D`） |
| 把若干小块拼成一个大 tile | **拼不了**。`tl.cat` 对 2D 不支持。改成「一次 `tl.load` 用一个大的 offs 向量直接读出整块」 |
| 分段填充一个大 tile（如按 g 分组填 q） | **一次读完**。用 `offs = tl.arange(0, BLOCK_G * ...)` 构造完整下标，分组语义靠 `offs // G` / `offs % G` 算进指针里 |
| 按条件跳过一段计算 | 循环体内**不支持 `continue`**，用 `if` 把整段包住 |

> **自查**：写完 kernel 后 grep 一遍 `[` 后面跟 `:` 的模式、以及等号左边出现 `]` 的行。命中任何一条都要改掉——这些错误在 Phase 3 才会以极不友好的报错暴露出来。

### L1.11 ★★★ 「按分支/路由选权重」**不等于**「按行选权重」——不许因此把 M 维退化成一行一 program

这是本子类**复现次数最多的一条**（`33_AdaptiveAttention` 上连续 7 轮命中）。**错误推理**：「每个样本选的分支不同 ⇒ 相邻行的权重不同 ⇒ 行之间不能合并成 tile ⇒ 一 program 一行」。**事实**：`sel` 的形状是 **`[B]`**，索引是 `sel[b]`，一个 batch 内的**全部 `S` 行共享同一个分支**。所以 `M = B*S` 天然是「`B` 段、每段 `S` 行同权重」的结构：**段内可以自由开到 `BLOCK_M=128`**，只是不能跨 batch 边界拼 tile。grid 写三级 `B * cdiv(S, bm) * cdiv(N, bn)`（拆出 b 占一级），`BLOCK_M` 必须传进 kernel。

**强制要求**：① 写之前先看 `sel.shape` 再决定，不许凭直觉推；② **不要把 tile 计算封装成辅助函数**（一旦返回三元组，调用处就会写成 `_, bn, bk = ...` 把 `bm` 丢掉）——**就地写死一行**；③ grep 自查 `return 1, ` ／ `_, bn` ／ `grid = min(核数, M * cdiv(...))`，正确的 grid 里**一定出现 `cdiv(S, bm)`**。

> ⚠️ 本条属于 **Layer 1（硬性设计约束）**。同一条规则先后写进 Layer 2 骨架 5 次**全部无效**——**凡是"必须这么写"的规则，一律写进 Layer 1；写进骨架/技巧层等于没写。**

---

## §3 Layer 2: 算法骨架（参考方向，输出必须是全新草图）

### §3.1 `gqa-sparse-decode` 主骨架

```
pid -> (t, kh)                         # ★ L1.1：按 kv_head 分组，不是 q_head
req = t // DQL;  q_off = t - req*DQL
kv_len = max(seq_lens[req] - DQL + q_off + 1, 0)
rtopk  = min(ceil(kv_len / BS), TOPK)
kv_len_f = float(kv_len)               # 位置比较走 f32（整数 LT 会标量降级）

q_tile = load Q[t, kh*G:(kh+1)*G, :]   # [BLOCK_G, D]
m, l, acc = -inf, 0, 0
for i in range(rtopk):                 # 运行期 trip-count
    blk  = load TOPK_IDX[kh, t, i]     # 标量
    page = load BLOCK_TABLE[req, blk]  # 标量索引 → 标量 load（★ 不要向量 gather）
    k, v = load KV[page, kh]           # 页内连续
    scores = dot(q_tile, trans(k)) * sm_scale
    scores = where(pos_f < kv_len_f, scores, -inf)
    m_new = maximum(m, max(scores, 1))
    p     = exp(scores - m_new)
    pv    = dot(p, v)                  # CV 重排：Cube 先发，Vector 后算
    delta = exp(m - m_new)
    acc, l, m = acc*delta + pv, l*delta + sum(p, 1), m_new
out = acc / l
```

### §3.2 `gqa-proj` 稠密 GQA 前向骨架（含投影）

```
host:  x[B,S,D] --一次融合 GEMM--> qkv[B*S, D + 2*KH*dh]     # 权重预转置 [in,out]，L1.6
kernel: pid -> (b, h, q_tile)                                 # grid = min(核数, tasks) + 步长循环
        kh = h // (H // KH)                                   # GQA 映射
        for kv0 in range(0, S, BKV):
            kv_valid = (kv0 + arange(BKV)) < S                # ★ G6：先算好有效列
            scores = dot(q_tile, trans(k_tile)) * inv_sqrt
            scores = where(kv_valid[None, :], scores, -3.0e38)  # ★★★ G6：掩在 scores 上
            ... (online softmax)
        store acc / l  ->  out[b, s, h, dh]                   # ★ 直接按目标布局落盘，L1.8
```
两个最大的收益点都不在骨架里而在参数上：**投影 GEMM 的 tile 按 dtype 分档**（§4.1，实测 +57%）与 **grid 收缩 + 核内步长循环**（L1.7）。

### §3.3 `gqa-router` 的四段结构

router 必须拆成三段（单 kernel `grid = B` 意味着最多 4 个核在读 `D²/4` 的 W1，且 W1 tile 直接 UB 溢出）：`router_mean`（xm = x.mean(dim=1)）→ `router_fc1`（relu(xm@W1ᵀ+b1)）→ `router_argmax`（sel = argmax）→ 分支侧展开成 `if`（QKV 融合投影 + attention + out_proj 各一次发射）。

⚠️ `router_fc1` 的 **M 只有 `B ≤ 4`**：既不能开 128（撑爆 UB），也不能因为 M 小就退化成 1D 向量——必须保持 `[BLOCK_M, BLOCK_K]` 二维 tile + `m_mask = offs_m < M` 掩码。**分支 kernel 整体只发一次**（每分支发一次会把未选中分支的整个 task 空间也走一遍，实测 9 次发射 → 3 次，impl 0.1386 → 0.0969 ms，1.43x）。`BLOCK_NOPT = next_pow2(NOPT)`（NOPT∈{2,3} 非 2 的幂）+ 掩码，否则 3 路 case 编不过。

**kernel 内 argmax + 最小下标 tie-break 的唯一写法**（tile 不能用下标访问）：
```python
offs_o = tl.arange(0, BLOCK_NOPT)
o_mask = offs_o < NOPT
logits = tl.where(o_mask, logits, float("-inf"))       # 无效槽不参与
best = tl.max(logits, axis=0)
idx  = tl.where(logits == best, offs_o, BLOCK_NOPT)    # 命中处放下标，否则放哨兵
tl.store(sel_ptr + b, tl.min(idx, axis=0))             # 取最小 => tie-break 与 torch 一致
```

### §3.4 页内连续 vs 跨步半行

KV cache 常见布局 `[num_blocks, KH, BS, 2*D]`，一行是 `[K(D) | V(D)]`。分两次 load（行距 `2D`）与一次连续 load 整行，实测差 **2.6 倍**（读 K+V 两次跨步半行 1.692 us/页 vs 一次连续读整页 0.642 us/页）。怎么用一条 load 分别拿到 K 和 V：QK 把 q 补零成 `[BLOCK_G, 2D]`（后 D 列恒 0）则 `dot(q2, kvᵀ) == q·Kᵀ`；PV `dot(p, kv)` 得 `[BLOCK_G, 2D]`，后 D 列正是 `p·V`，store 时用**列掩码 + 指针偏移 `-D`** 取后 D 列。⚠️ **前置条件**：这条会把 UB 需求推高一倍，**必须先把 PV 的 fp32 缓冲换成 fp16**（§4.3），否则必然 `PlanMemory Failed`。

---

## §4 Layer 3: 关键技巧

### §4.1 ★★★ 投影 GEMM 的 tile 按 dtype 分档（`gqa-proj` / `gqa-router` 最大的一笔）

实测 `linear_kernel` **128.8 us → 35.9 us（3.6x）**，端到端 **+57%**。⛔ **不要自己拍一组**（`128/64/128` 或 `64/128/64` 都不行），**照抄**：
```python
if x.dtype == torch.float32:  tgt_m, tgt_n, tgt_k = 128, 256, 64
else:                         tgt_m, tgt_n, tgt_k = 128, 256, 128
bm = max(16, min(tgt_m, triton.next_power_of_2(M)))
bk = max(16, min(tgt_k, triton.next_power_of_2(K)))
bn = max(16, min(tgt_n, triton.next_power_of_2(N)))
```
机理：`BLOCK_M=64` 时 M-tile 数 = `ceil(M/64)`，**权重被整份重读这么多遍**（实测 M=1149 下 18 遍 × 26MB ≈ 468MB GM 流量）。UB 可行域大致 `BM*BN*BK ≤ 2^20`，但 **fp32 下 BK=128 比同乘积的 BK=64 更吃 UB**（操作数字节翻倍），不能只看乘积。

### §4.2 attention tile 基本没有空间，不要在这里花时间

全档扫描实测：`128/128` 只比各形状最优档慢 **3~8%**。把时间花在投影侧（§4.1）或任务划分（L1.1）上。

### §4.3 ★ PV 段用 fp16 原生 dot 换 UB（不是换速度）

三种写法实测：`dot(p_f32, v.to(f32))`（2 dot）1.3791、`dot(p.to(bf16), v)`（2 dot）MERE 超阈 FAIL、**`dot(p.to(f16), v.to(f16))`**（2 dot）1.3088 通过。关键：**bf16 → fp16 在本类取值域内逐位无损**（bf16 8 bit 有效位装进 fp16 的 11 bit），`p` 尾数从 8 bit 涨到 11 bit，补上余量。

⇒ ⛔ **fp16 路线是负优化，不要用它来提速**（干净 A/B：fp16 慢 1.42 倍，多出来的是两次 Vector 转换）。它唯一的价值是**腾出 32KB UB**，只有当你确实需要那 32KB 去装别的东西（如 §3.4 整页 load）且能证明收益更大时才值得。

> ⛔ **这条只在 bf16 主导的算子上成立**：判据是**精度阈值**不是「能不能无损转 fp16」。`21`（含 fp16 case，阈值 `2^-10`）上 fp16 单 dot 只有 24/50——算子里只要有 fp16 case，就不要走这条。

### §4.4 ★ 路由门控：只算被选中的分支

理论收益 = 分支数的几何平均（实测 `3^0.7 × 2^0.3 ≈ 2.656x`）。门控只是一次标量 load：`if tl.load(sel_ptr + b) == BRANCH: ...`。

⛔ **写出必须覆盖全部行——这是门控最容易漏的一处**（实测 14 个 case 的 `impl` 整片为 0，相对误差恰好 1.0）。参考实现是 `(stack(cands,1) * one_hot).sum(1)`，**每一行都有值**。三个自查点：① `sel[b] ∈ [0, nopts)` 对所有 b 成立；② forward 展开的 `if` 段覆盖全部 nopts；③ kernel 内 n-tile/m-tile 提前退出条件不能把该写的行也跳掉。**症状指纹**：`impl` 恰好为 0、相对误差恰好 1.0 ⇒ 一定是没写，不是算错。

⚠️ **语义前提（必须实测）**：跳过 ≠ 乘 0。仅当未选中分支会产出 NaN/Inf 时两者不等价。跑一遍参考实现统计 NaN/Inf，实测 `cases with non-finite: 0` 后才成立。

### §4.5 位置比较走 f32，不要用整数 LT

整数比较在本类算子族上会标量降级（实测 +23%）。`kv_len_f = kv_len.to(tl.float32)` 循环外、`pos_f = (blk * BS + n_offs).to(tl.float32)` 循环内。

### §4.6 CV 指令重排（成本极低，试一次，但不要预设收益）

把 Cube 的 `dot(p, v)` 先发、Vector 的 `sum`/`exp` 后算。实测收益在噪声量级（+1.9%）。**不要指望它解决 CV 串行**（见 L1.3 B 轮）。

### §4.7 冗余边界运算消除

当 `BLOCK_D == D` 时 `d_mask` 恒全 true，掩码 + `tl.where` 是纯冗余：constexpr 分支走无掩码 load，让 AIC 可以 GM→L1 直载，省掉 AIV 的 vsel 清零与 GM 暂存往返。

---

## §5 优化点清单

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测收益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ 环境闸门（正确工具链 + 健康卡） | 0.9438 → **2.2904** | 无条件，第一件事 |
| 2 | ★★★ attention 用 `tl.dot` + online softmax 重写 | 0.0016 → 1.4518（**~900x**） | 基线里没有 `tl.dot` |
| 3 | ★★★ 任务单元改按 kv_head 分组（L1.1） | 0.2233 → **1.3791（6.18x）** | kernel 里有 `kh = h // G` 且循环变量是 q head |
| 4 | ★★★ 投影 GEMM tile 按 dtype 分档（§4.1） | 1.4518 → **2.2839（+57%）** | 有独立投影 GEMM |
| 5 | ★★ 路由门控只算选中分支（§4.4） | 理论 = 分支数（端到端 **2.9033**） | 有 argmax/one-hot 选路 |
| 6 | ★ QKV 三合一 + 权重预转置 `[in,out]`（L1.6） | 含在 #2 内 | 有多个共享输入的 `nn.Linear` |
| 7 | ★ 输出直接按目标布局落盘（L1.8） | 消掉一个 19.46 us Transpose | golden 里有 `transpose(1,2).contiguous()` |
| 8 | 整页连续 load（§3.4） | DMA 1.692 → 0.642 us/页 | 页布局 `[K|V]` 同行，且已做 §4.3 |
| 9 | 冗余边界运算消除（§4.7） | 数个百分点 | `BLOCK_D == D` |
| 10 | CV 指令重排（§4.6） | +1.9%（噪声量级） | 成本≈0，顺手做 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱，不要重跑这些死路**）

| 方向 | 结论 | 证据 |
|---|---|---|
| **多页合并成一个大 tile 摊薄 CV 握手** | **UB 墙，编不过** | NP=2 的三种写法（标量构造 / `tl.cat` / `tl.join`+3D `tl.trans`）**全部** `PlanMemory Failed`。记账见 G3：基准已占 188KB/192KB |
| **用行基址向量 gather 拼多页** | **NP=1 就只有 0.52x** | gather 形态走不了 Cube 的 GM→L1 直载，被迫由 AIV 搬进 UB 再转交 Cube |
| **PV 错开一轮，打断 Cube→Vector→Cube 依赖** | **1.3791 → 1.2647** | `serial_ratio` 几乎没动 ⇒ **握手是按 dot 结构性插入的，与数据依赖无关** |
| **转置 QK 形态**（省 `trans(k)`） | **慢一倍** | 转置布局的 q load 跨步 + `axis=0` 归约更贵 |
| **`p_hi/p_lo` 二段拆分提精度** | **性能 -49%** | 循环内 dot 2→3；本类瓶颈是 dot 个数 |
| **PV 段换低精度 dot 提速** | **不提速** | dot 个数没变 ⇒ 时间持平。价值只在腾 UB（§4.3） |
| **编译参数调优** | **无收益，其中一个有害** | 9 个全在 ±1.6% 内；`ops_reorder` **x0.759** |
| **索引链软件流水**（blk/page 做成循环携带变量） | **编译器拒绝** | `Failed to collect vector loop tiling info` |
| **两级 gather 拆成独立建表 kernel** | **impl +12%** | 标量占比不是索引载入链造成的 |
| **把两个小 kernel 融成一个** | **该段慢 2.6 倍** | 融合要求一个 program 独占一行，操作数退化成 `[1,K]`，`tl.dot` 用不了（要求 M≥16）——等于把 GEMM 从 Cube 搬到 Vector |
| **`p` 二段拆分在稠密 attention 段** | **数值成立，UB 不可行** | 三份 `[BLOCK_M,BLOCK_N]` 同时活着，超 13% |
| **靠调整表达式顺序 / 就地覆写省 UB** | **完全无效** | bishengir 按 tiling basic block 统计 UB，不看源码级活跃期；省 UB 只能缩 tile |
| **`tl.load(mask=m, other=0.0)` 替代 load + where** | **不等价，精度崩** | 34/50——被掩掉的 lane **不保证真的写成 0** |
| **掩码恒全真时改走无掩码 load** | **9/50** | torch 标杆自己就产生 NaN，验证要求逐位置复现；那几行 `where` 是**承重的** |
| **K 预转置消除每迭代的 `tl.trans`** | **更慢或编不过** | fp32 档 x0.969；低精度档直接 UB overflow |
| **靠 tile 继续压 attention** | **没空间** | 最优档只快 3~8%（§4.2） |
| **堆同组 q head 复用 KV** | **账上不成立** | tile 行数被 UB 钉死，总 KV 访存量不变 |
| **利用 token 之间的 topk 页复用** | **账上不成立** | 实测冗余仅 1.06x |
| **split-K / flash-decoding** | **净亏** | 收益只在 `programs < 核数` 的少数 case，而多一个 merge kernel 给**每个** case 加时延 |
| **`tl.where` 数量/位置导致编译崩溃** | **证伪** | 删掉 3 处冗余 `where` 后失败集合逐个相同 |
| **「tile 某一维不能是 128」** | **证伪** | 真判据是 UB 字节数，且上限随编译器版本变（G2） |

### §5.3 三个会误导的 profiling 字段

| 字段 | 直觉解读 | 实际含义 |
|---|---|---|
| `aic_scalar_ratio` / `aiv_scalar_ratio` 高（0.5~0.73） | 标量运算太多 | **同步等待被计入标量管线**（IR 里对应 100+ 个 `set_flag`/`wait_flag`/`pipe_barrier`/`sync_block_*`） |
| `cube_utilization` 高（70~93） | Cube 很忙 | 与 `aic_mac_ratio`（0.02~0.15）一起看才有意义——Cube 是**被占用**，不是在算 |
| `scf.for` 计数 | 循环嵌套层数 | Mix kernel 的 IR 里**每层循环有 Cube/Vector 两份**。4 个 `scf.for` = 2 层循环 × 2 份 |

---

## §6 精度闸门（先过闸门，再谈性能）

### §6.1 判据

三项 AND：`allclose(atol, rtol)` / `matched_ratio ≥ 0.9` / **`MERE < rel_threshold`**。`rel_threshold` = fp16 `2^-10`、bf16 `2^-7`、fp32 `2^-13`。

### §6.2 `p`（softmax 输出）不能直接降到输入 dtype

`p` 舍入到 bf16 的半 ulp `2^-9` 在 `Σp·v / Σp` 上被放大越阈。两条出路：**fp16 中转**（§4.3，若输入 bf16 且值域在 fp16 内，首选）；`p_hi/p_lo` 二段拆分（精度最好但多一个 dot，性能负项）。

> ⚠️ 「`.to(dtype).to(fp32)` round-trip 会被编译器消除」这条在 bishengir 1.2.0 上**没有复现**——最朴素的 `p_hi = p.to(dt); p_lo = (p - p_hi.to(f32)).to(dt)` 即可。

### §6.3 先用 CPU 模拟筛精度方案

不占 NPU、秒级出 50 个 case 的三项判定。多个候选精度方案可以在不写 kernel 的情况下先淘汰掉。

### §6.4 ★★ `gqa-router` 专属：argmax 边距体检（动手前必做）

选路靠 `argmax(routing_logits)`，**一旦重算的 logits 把 top1/top2 翻转，输出直接是另一路的结果**。在 CPU 上把 `top1 - top2` 边距扫一遍。实测 fp32 那些安全（~400x 余量）；**吃紧的是 bf16**：logits 尺度 0.046、bf16 量子 ≈1.8e-4，边距只有 **12 个量子**。⇒ **router 必须逐层复刻 golden 的精度语义**：`mean` fp32 累加后回铸 `x.dtype`；两层 Linear 都 fp32 累加后回铸，让 argmax 比较发生在同一量化格点上；tie-break 取最小下标。**不要「为了稳一点全程 fp32 不回铸」——那反而会与 torch 分道扬镳。**

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

### §7.1 环境闸门

见 G1。每轮至少三道闸门：**benchmark 全局互斥锁**、**必须测满全部 case**、**工具链指纹前后对照**。实测被拦下的两次无效测量：① 别的会话中途重装 triton，我方只测到 13/50 却算出一个漂亮的 2.83x；② 跑错工具链。

### §7.2 ⚠️ benchmark 时延口径已变更（决定要不要拆 kernel）

**旧版**按 kernel 名 groupby 取末尾 repeats 行 ⇒ 同一个 kernel 一次 forward 内发射 N 次**只按一次计**（把不同名 kernel 合并成同名会虚高）。**新版**由整组行数反推发射次数 L，**一次 forward 内多次发射如实全部计入**。⇒ 新口径下多发射不再有度量红利，按真实 wall-clock 设计即可；**跨版本比较历史数字时必须核对用的是哪一版脚本**。实测影响：`80`/`81` 单 kernel 算子的 speedup 因 torch 侧 Python 循环参考实现被旧口径低估 ~30 倍而大涨（2.2904 → 59.5558 / 1.3791 → 41.9099）。

### §7.2b ★ 共享机器上单点测量一律不可信，且「比值抗污染」不是普遍成立的

共享机器上被别的会话占满时，**绝对时延整体放大 6~10 倍**。比值抗污染程度**取决于 torch 侧本身有多慢**：golden 是 Python 循环的（发上千小 kernel）比值只变 −7%，golden 只有几个大 GEMM 的比值变 −30%。⇒ ① 任何版本对比都必须**同时段、同一块卡、背靠背连测**；② 开测前 `npu-smi info` 看 AICore 占用，>50% 就换卡或等。

### §7.3 `speedup_vs_torch` 是各 case speedup 的几何平均

小 case 与大 case 等权（G5）。本类有一个额外陷阱：**torch 侧 framework 时延与 case 规模基本无关**（实测恒在 60~130 us），于是 `speedup ≈ 常数 / impl_us`——**优化目标退化成「压 kernel 绝对时间」**。

---

## §8 陷阱表

| 陷阱 | 症状 | 处理 |
|---|---|---|
| 非交互 shell 读不到 CANN `set_env` | 无报错，分数系统性偏低一半 | 显式 source + 版本指纹校验（G1） |
| 卡被 aicore timeout 打坏 | 精度结果随机、跨进程持续 | 无关算子对拍 CPU（G1）；换卡 |
| 运行期标量除 + 向量下标 gather | aicore timeout 507014 | 把该维提升为 tile 坐标（L1.2） |
| fp32 dot + N=64 + 运行期 trip-count | **偶数列** NaN | N 抬到 128 / 改原生 dtype（L1.5） |
| `tl.trans(x)` 的隐藏 UB 开销 | `PlanMemory Failed` | 记账时算成一份与 x 等大的空间（G3） |
| forward 里写 Python `for` | AST 校验器无条件判违规 | 展开成 `if`（G7） |
| `torch.cat` 拼权重 | AST 校验器拒绝 | `torch.empty` + `.copy_()`（L1.6） |
| 相对误差 **0.5~1.5 量级** | 结果整个是错的 | **不要**去调 dtype。先用 L1.9 的 CPU 对拍排除权重复刻；没问题再查计算侧 |
| **tile 切片 / 切片赋值** | `ValueError: unsupported tensor index` 或裸 `AssertionError()` | **首版生成的头号杀手**。见 L1.10 |
| 循环体内 `continue` | Triton 不支持 | 用 `if` 包住整段（L1.10） |
| **decode 的任务划分套到 prefill** | 精度全过但 speedup **< 1** | `S_Q > 1` 必须按 `(b, q_head, q_block)` 分块，见 L1.1 |
| **grid 把 M 维塌掉 / 漏传 `BLOCK_M`** | 精度全过但 speedup **< 2**；GEMM 段慢一个数量级 | `grid = min(核数, B * cdiv(S,BLOCK_M) * cdiv(N,BLOCK_N))`，`BLOCK_M` 必须传进 kernel |
| **host 侧缓冲形状与 kernel 索引不匹配** | 先 507015 越界，修完仍选错分支（误差 ~1.0） | 每个缓冲对着 kernel 的索引表达式核一遍维度 |
| **对不喂 `tl.dot` 的权重也做预转置** | 相对误差 **~1.0 量级**（权重值是对的） | 见 L1.6（w2 用向量点积，不能 `.t()`） |
| **打包缓冲用 `torch.empty` 而非 `zeros`** | 三种 dtype 均匀失败、相对误差 100 倍量级 | 各分支宽度不等 ⇒ 窄分支填充列是未初始化内存；一律 `torch.zeros` |
| **`tl.dot` 两侧 dtype 不一致** | `AssertionError('Both operands must be same dtype')` | `tl.where(mask, x, 0.0)` 把一侧提升成 fp32；补零后 `.to(other.dtype)`（L1.5） |
| **`tl.dot` 操作数退化成 1D** | `AssertionError('Both inputs must be either 2D or 3D')`，整批全灭 | M 小（≤4）时保持 `[BLOCK_M, BLOCK_K]` + 掩码，`BLOCK_M` 取 16 |
| **`tl.arange(0, dh)` 用运行期标量当长度** | `arange's arguments must be of type tl.constexpr`，50 个 case 全灭 | host 算好 `BLOCK_DH = next_power_of_2(head_dim)` 传 constexpr |
| **`ptr.dtype` 当成元素类型用** | `cannot cast fp32[...] to pointer<fp16>`，一处写错 50 个全灭 | 回铸一律写 `ptr.dtype.element_ty` |
| **KV padding 只清零操作数、没掩 scores** | impl 与 framework 同号但整体偏小（0.3~0.7） | **首版生成第二杀手**。见 G6 |
| 只在大 case 上扫参 | 结论与最终分数相反 | 大/中/小三档（G5） |
| **把某个算子的结论跨子类外推** | 照搬后性能不升反降 | 「dot 个数是成本单位」只在握手主导形态成立，见 L1.3 反例。**先 A/B 再定** |
| 拿 ratio 当结论 | framework 漂移让更快的版本分数更低 | 同时看 `impl.avg_latency_ms` 绝对值 |
| `topk_idx` 选出超过 `kv_len` 的整页 | Implementation NaN 位置不匹配 | 用 `if blk * BLOCK_BS < kv_len:` 包裹页处理体；guard `out = acc / l` 除零 |
| 读 `multiprocessor_count` | `KeyError` | 用 `torch.npu.get_device_properties(dev).cube_core_num` 取核数 |

---

## §9 与 `flash_attention` 卡的分工

| 条目 | FA 卡 | 本文件 |
|---|---|---|
| KV 形态 | 连续区间，靠 causal/window 收缩扫描区间 | **按页间接寻址**，页间不连续 ⇒ 区间收缩完全不适用 |
| 放大 tile 摊薄同步 | 主力手段（BLOCK 开到 UB 上限） | **撞 UB 墙**：基准已占 188/192KB，多页合并全部编不过（§5.2） |
| 目标函数 | 循环体内**互有 UB 依赖的向量算子数** × KV 迭代数 | 循环体内 **`tl.dot` 的个数** × KV 页迭代数（L1.3） |
| `p` 二段拆分 | 正收益（+2.7%） | **负收益 -49%**（多一个 dot） |
| 任务单元 | 按 (batch, head, q_tile) | **必须按 (token, kv_head)**（L1.1，6.18x） |

两者共通、不要重复踩的：`tl.dot` 契约、UB 预算随编译器变动、`aic_scalar_ratio` 的真实含义、grid 收缩 + 核内步长循环、投影权重预转置。
