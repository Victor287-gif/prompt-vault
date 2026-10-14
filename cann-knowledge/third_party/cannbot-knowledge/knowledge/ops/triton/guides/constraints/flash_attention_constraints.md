---
type: Guide
title: flash_attention 类算子设计约束
description: flash_attention 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.attention, topic.compute.normalization, topic.implementation.memory_management, topic.implementation.masking, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# flash_attention 类算子设计约束

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## FlashAttention 类算子优化经验

本文档是 **FA 主链（`Q@Kᵀ → softmax → @V`）+ KV 分块 + online softmax** 这一类算子的经验合集，覆盖 Phase 2/3/4：

- **§1 通用经验**：跨形态共有的工程约束
- **§2 Layer 1 设计约束**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（Phase 2 参考方向）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**（含**证伪方向全表**）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表**

> ⚠️ **本文件与 `attention.md` 的分工（重要）**：
> 凡满足 §0.1 判别特征的算子（KV 分块 + online softmax 状态量，或带 causal / sliding-window / softcap 的 attention 主链），
> **一律使用本文件**，不要读 `attention.md`——后者的 L1.1（禁用 fp32 `tl.dot`）/ L1.3（`next_pow2`）/ L1.6（全程 fp32）
> 三条在 FA 路径上**已被实测推翻**（差异与证据见 §9）。
>
> **证据基础**：三个 FA 类算子各 50 case 的完整优化轨迹——`FlashAttentionV2`（含 4 个投影，3.68 / 换精度契约后 3.30）、
> `FlashAttentionFwd`（无投影、可变 mask，2.17 → 6.24）、`MultiQueryAttention`（含投影 + KV 共享，0.1052 → 2.3724）。
> 三者形态、精度契约、编译器版本各不相同却**独立收敛到同一瓶颈结论**，这是本卡可外推的根据；
> 三者之间的 **3 条冲突结论**及其判别前置条件见 §2 末尾的「跨算子结论冲突的统一判据」。
>
> ⚠️ **核心优化哲学**：这类算子的瓶颈是 **Cube↔Vector 跨核同步**，不是算术量、不是访存、不是 dtype。
> 一切优化等价于**减少「循环体内互有 UB 依赖的向量算子数 × KV 迭代总数」**。
> 生成时**禁止混用**其他类别的经验——尤其不要套用 `transformer-inference.md` 的多 kernel 拆分策略（§5.3 已用访存账否决）。

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/flash_attention__flash_attention.md`](../operator_playbooks/flash_attention__flash_attention.md) — FlashAttention
- [`operators/flash_attention__precision_gate_and_measurement.md`](../operator_playbooks/flash_attention__precision_gate_and_measurement.md) — FlashAttention

## Attention 家族写法分类（姊妹卡）

本卡是 attention 家族**标准类主链**（KV 分块 + online softmax）的共享底座。家族内其余写法类按
[attention 家族写法分类](attention_writing_classification.md) 路由，各卡**继承本卡 §1 通用经验
与 Layer 1（L1.1~L1.16），只写该写法独有的 delta**：

- [`mha_constraints.md`](mha_constraints.md) — 一·标准类·含投影段（MHA / SelfAttention / CrossAttention，4 个 `nn.Linear` 投影）+ 反传链 + 单头大 C SDPA
- [`gqa_constraints.md`](gqa_constraints.md) — 二·压缩类·head 共享（GQA / MQA / 路由）+ 页表寻址
- [`linear_constraints.md`](linear_constraints.md) — 二·压缩类·低秩投影（Linformer 等线性复杂度注意力）
- [`block_sparse_attention_constraints.md`](block_sparse_attention_constraints.md) — 三·稀疏类·块跳过/top-k
- [`sparse_mask_constraints.md`](sparse_mask_constraints.md) — 三·稀疏类·掩码型（继承 FA 只写掩码 delta）

> 凡命中上述写法类的算子，判别表见 `attention_writing_classification.md`；本卡的 §1/Layer 1 条款对其同样成立。

## §0 适用范围与算子分类

| 算子 | 子类标签 | 计算特征 | 优化哲学 |
|------|---------|---------|---------|
| MultiHeadAttention | `fa-mha` | 4 个 `nn.Linear(D,D)` 投影 + `softmax(Q@Kᵀ/√d)@V`，`S_Q ≠ S_K`，`head_dim` 普遍非 2 的幂 | host 侧忠实复刻 `nn.Linear` 语义 + kernel 侧 online softmax；**投影侧与 attention 侧分别优化** |
| self/cross-attention、SDPA | `fa-sdpa` | 同上但 q/k/v 由外部传入 | 同上，去掉投影段 |
| flash-attention 前向 | `fa-fwd` | KV 分块，不物化 `[S_Q,S_K]`；常带 causal / window / softcap **运行时属性** | 重点在 **KV 扫描区间收缩 + mask constexpr 特化** |
| GQA / MQA | `fa-gqa` | `H_Q > H_KV`，KV head 共享；**通常自带 4 个投影 GEMM** | 同 `fa-fwd`；⚠️ 若含投影，**最大的一笔在投影侧**（L1.12，实测端到端 5.94x），不要一上来就攻 attention |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. kernel 内对 KV 维度分块循环，且跨迭代滚动 online softmax 状态量 `m` / `l` / `acc`；
2. attention 主链带 `causal` / `window_left` / `window_right` / `softcap` 等 mask 属性；
3. 计算图含 `scaled_dot_product_attention` 或 `softmax(Q@Kᵀ/√d)@V`，且 `S` 大到不宜物化 `[S_Q,S_K]`；
4. ⭐ `forward` 里出现 `torch_npu.npu_xxx(...)`（如 `npu_fusion_attention`）→ **golden = CANN op 的真实行为**，文件内同名的纯 torch fallback 只是 docstring 参考语义、**不是 golden**。满足本条 ⇒ 除本文件外，还**强制**先走 §6.1 CANN 语义探针（见 [FA 精度闸门与测量口径](../operator_playbooks/flash_attention__precision_gate_and_measurement.md)）再谈性能（否则 §3.2 区间收缩、§3.3 趟数结构等方向会直接算错）。

不满足（如极小 `S` 的朴素一次性 attention、无 KV 分块）→ 仍用本文件，跳过分块流式相关的 Layer 2/3，只取 Layer 1 与通用经验。（历史遗留的朴素 attention 卡 [attention_constraints.md](attention_constraints.md) 保留，按旧判别表路由。）

### §0.2 ★ 形态识别五问（**设计第一步必须回答，答案决定后续哪些章节适用**）

| # | 问题 | 影响 |
|---|------|------|
| **Q0** | **golden 是 CANN op 还是 torch 数学实现？** | CANN（`forward` 里有 `torch_npu.npu_xxx`）→ **先走 §6.1 探针**，golden 真实行为可能与文档/数学语义相反；torch 数学实现 → 直接看 Q1-Q4 |
| **Q1** | 有没有独立的投影 GEMM？ | 有 → §4.4 / §4.5（权重预重排、分块分档）是最大的一类收益；**没有则这两条完全不适用** |
| **Q2** | mask 是固定的还是可变属性？ | 可变 → §3.2（区间收缩）+ §2.7（constexpr 特化）合计实测可达 **+58%**；固定 causal 拿不到这么多 |
| **Q3** | 参考实现的 attention 在什么精度上算（有没有 `.float()`）？ | 决定 `p` 能否降精度，以及**趟数结构**（见 §6.3） |
| **Q4** | 评测口径？ | 几何平均 ⇒ **每个 case 权重相同**，把一个 case 从 0.5x 拉到 1.0x，与把另一个从 8x 拉到 16x 贡献完全相同。**不能只优化大 case** |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### F1 归约维 padding 必须显式排除，`other=0.0` 不是掩码

- **必须** `S_K` 非 `BLOCK_KV` 整数倍时，`max` 前 `tl.where(kv_valid, scores, NEG)`，`sum` 前 `tl.where(kv_valid, p, 0.0)`。
- **禁止** 只靠 `tl.load(..., other=0.0)` 就认为越界列已被排除。
- **Why**：`other=0.0` 只让越界列 score 变 0，而 `exp(0-m) > 0` —— 这些"幽灵列"会混进 softmax 分母，把 `acc/l` 整体缩小。实测某 MHA 的 50 个 case **全部** `S_K % 32 != 0`，单这一条的 MERE = 8.16e-01（超 fp32 阈值 6700 倍）。
- **验证集要求**：主动构造 `S_K % BLOCK_KV == 0` 与 `!= 0` 两类用例，否则该缺陷会逃逸成"48/50 通过"这种更难诊断的形态。

### F2 掩码用**有限极小值**，不用 `-inf`

- **必须** 用 `-3.0e38`（fp32）/ `-1e30`，且声明为 `tl.constexpr`（否则 `NameError: Cannot access global variable`）。
- **Why**：`-inf` 会在整行被掩时产生 `exp(-inf-(-inf)) = NaN`，并逼着 `tl.maximum` 保留 NaN 处理，与 §4.2 的性能项**直接冲突**。
- **安全性论证（写进注释）**：causal + online softmax 下 KV 循环从 0 开始，第一块必含合法列 ⇒ 每行首次更新时 `m` 即为有限值，后续整块被掩的行 `p = exp(-3e38-m) = 0`，不污染。

### F3 `BLOCK_D` 用 `ceil16(D)`，**不用** `next_pow2(D)`

- **必须** `BLOCK_D = ceil16(head_dim)`，并对 `d_offs < D` 做 mask。
- **Why**：triton-ascend 支持非 2 的幂 `tl.arange`。D=96→96 vs 128、D=160→160 vs 256、D=192→192 vs 256，`next_pow2` 白白浪费 **30~60%** 的 tile 面积，直接转化成迭代数与同步开销。
- **回退**：仅当目标版本 `tl.arange` 不支持非 2 的幂时才退回 `next_pow2`。
- **禁止** 硬编码 `BLOCK_D`（如 16），也**禁止**把 `BLOCK_D` 放进 `@triton.autotune` 搜索空间——`head_dim > BLOCK_D` 时只算前 `BLOCK_D` 维，**静默错**。

### F4 UB 预算：中间张量元素数必须有与输入无关的上界

- **绝对禁止** `BLOCK_* = triton.next_power_of_2(<运行期变量>)` 且不加 `min()` 封顶。实测 `BLOCK_N = next_pow2(S)` 在 `S=577` 时需 792 KB，直接溢出。
- **面积上限有明确来源**：`scores` tile 是 `[BQ, BKV]` fp32 ⇒ `BQ*BKV ≤ 8192`（32KB）是**算出来的**，不是拍脑袋；再叠加 `acc [BQ, BLOCK_D]` fp32。UB 共 192KB（`ub overflow, requires ... while 1572864 bits available`）。
- ⚠️ **UB overflow 会污染设备，导致级联假失败**：一次 `ub overflow` 可触发 vector core exception，使**后续用例连带失败**（实测一次全量从 49/50 掉到 36/50，真正硬失败只有 3 个，其余 11 个是陪葬）。
  → **诊断原则**：看到大批连续失败时，先定位**第一个**失败用例单独复跑；不要按失败总数估计缺陷数量。

### F5 结构性改动之后必须重扫 tile，**不要查历史清单**

实测：一个算子在 KV 区间收缩后，最优 tile 从 `128x64` 变成 `64x128`（与旧结论**相反**）；另一个在权重重排后重扫，结论未变——但那是**验证过**的，不是假设。

### F6 扫参子集必须含大/中/小三档

⚠️ **几何平均口径下最容易犯的错**：拿 5 个大 case 扫参得到的最优配置直接全局套用。
实测一次：诊断子集 +10%，正式测量 **−3.7%**——退化全发生在诊断子集没统计的中等 case 上。
**画像用于定位，判定只能用完整的 verify + benchmark。**

### F7 ★ 精度仲裁优先序：宽容差标杆时以 verify 容差余量为 judge

- **冲突场景**：FA L1.4/L1.5/L1.16-镜像（"全 fp32 参考 ⇒ 任何中间量不许落低精度"）
  与低精度杠杆（hi/lo 分解、PV1 丢 p2 残差、p 降原生 dtype 走 cube）在**纸面矛盾**。
  两卡原先都无优先级声明 ⇒ 保守侧默认获胜——正确草图会被 Step 3 门禁误判为 A 类
  精度错误（实测：同一 CrossformerAttention 异环境批跑被压到 geomean 0.4921）。
- **仲裁规则（判据是 verify 容差余量，不是参考实现计算路径）**：
  - 严标杆（逐元素 MERE / matched_ratio≥0.9 卡阈值）⇒ 镜像规则成立，全程对齐参考
    精度路径；
  - **宽容差标杆（atol/rtol 显著宽于参考链精度，atol≥~1e-2）⇒ 允许利用余量**：
    hi/lo 是 p=p1+p2 **分解**（p1 原生 dtype 精确表示，非"直接降"），PV1 丢 p2 残差
    以**全量 verify 通过**为前提。
  - **禁止**把 PV 操作数 `.to(tl.float32)` 升精度 dot（fp16 cube 4x 吞吐损失）；禁止
    追"与参考链逐位对齐"（verify 不要求，参考链口径 ≠ 判据口径）。
- **完整判据**（SPLIT_D 并行双半硬反例、MERE 擦边只回退擦边桶、tile 环境相关性）见
  [mha_constraints.md](mha_constraints.md) §6.5；910b2 复跑落地见
  [operators/attention__multiheadattention.md](../operator_playbooks/attention__multiheadattention.md) §2.2。

---

## §8 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Cube 空转、benchmark 超时无数据 | 用逐维外积代替矩阵乘 | L1.1 |
| 非 fp32 case 全部编译失败 | `tl.dot` 两操作数 dtype 不一致 | L1.2 |
| softmax 分母偏大、结果整体偏小 | KV padding 列未置 `NEG` 就进了 `sum` | F1 |
| 输出大面积 NaN | 掩码用了 `-inf`，整行被掩时 `exp(-inf-(-inf))` | F2 |
| `head_dim` 大时 `ub overflow` | 某 block 维由运行期 shape 决定且无上界 | F4 |
| 大批用例连续失败但逐个复跑只错几个 | `ub overflow` 触发 vector core exception 污染设备 | 定位**第一个**失败用例单独复跑，不按失败总数估计缺陷数 |
| `BLOCK_D` 白白浪费 30~60% | 用了 `next_pow2(D)` | F3 |
| 放大 BLOCK_KV 只有 +1.2%，误判方向无效 | `kv_lo` 仍在对齐，把收益吃掉了 | L1.8，两者成对测 |
| 诊断子集 +10%、正式测量 −3.7% | 扫参子集只有大 case，中等 case 块数塌陷 | F6 + §4.5 并行度保护 |
| 小 shape 性能上不去，想拆 kernel 模块化 | MIX kernel 每次发射 ~4.5us 固定开销 | §5.3，**绝不要拆** |
| **精度全过但 speedup 只有 1.0~1.2 上不去** | **BLOCK 被写死成 32×64 这类小值**，面积只有预算上限的 1/4 ⇒ KV 迭代数与跨核握手多 4 倍。这是本类算子最常见的性能天花板成因 | **L1.13**：按面积预算 `BQ*BKV<=16384` 取到上限，禁止硬编码 |
| 大 case 的 attention kernel 时延居高不下 | `tl.maximum` 默认展开成 7 条向量指令，每次迭代都执行 | **L1.14**：多块路径开 `propagate_nan=ALL`，单块路径必须关 |
| `aic_scalar_ratio` 73%，改了五次标量代码全无效 | 同步等待被计入标量管线 | §5.4，先 dump IR |
| 权重在 NPU 上重采样 / 投影算成 `x @ W` | CPU 与 NPU 是两条独立 RNG 流；`nn.Linear.weight` 是 `[out,in]` | L1.11 |
| **投影 GEMM 很慢、`MTE2>60%`，attention 侧却没问题** | **权重仍是 `[out,in]` 原始布局，b tile 最内轴 stride = `in_features` ⇒ 离散 gather**。坏味道：kernel 注释写着"w 形状 `[N,K]`（`nn.Linear.weight` 布局）"、host 侧传 `w.stride(0), w.stride(1)` | **L1.12**（实测 182x / 端到端 5.94x，首次生成就要做对） |
| 转置了权重但收益不及预期、投影 kernel `MTE2` 仍 >90% | 布局改了**没重扫分块**：转置把算子从"B 转置"变成"都不转置"，推荐分块随之改变；`BLOCK_K=32` 的 fp16 只有 64B，是 512B 对齐线的 **1/8** | **L1.12 配套硬约束**（`BLOCK_K*elem_size>=512`，实测再 +13.2%，MAC 17.9%→40.3%）。该洞在同类算子上挂了二十多轮 |
| **大 shape 报 `RuntimeError: grid should be less than 65536`** | **辅助 kernel（RoPE/位置编码/bias 预处理）的 grid 按 `B*N*S` 直接展开**。判别：失败集合严格按 `B*N*S ≥ 65536` 划分，且是 RuntimeError 而非精度不匹配 | **L1.9**：算子内**每个** kernel 都用 `grid=min(核数,tasks)` + 核内步长循环；不要用 `TRITON_ALL_BLOCKS_PARALLEL=1` 绕过 |
| **带位置编码的 case 全部 `CompilationError`**，其余全过 | `arange's arguments must be of type tl.constexpr` —— 交错重排里用运行期值构造 `tl.arange` 或动态 `tl.reshape`。判别：失败集合严格按 `use_rope`/`use_alibi` 这类布尔属性划分 | **L1.10 交错重排配方**：constexpr `tl.arange` + `tl.where` 就地选择，全程不改形状 |
| **小 shape 尚可、大 shape 崩塌（<0.01x）**，每次迭代成本达数百 us | **输入变换（RoPE/位置编码）被放进 KV 循环内逐块重算**，且其中间量占 UB 把 BLOCK 逼小——同时挨两刀 | **L1.15**：用独立 Triton kernel 外提，再重扫 BLOCK（实测 0.005x → 1.386x） |
| 看到 `BLOCK = min(..., 64)  # avoid UB overflow` 这类预防性封顶 | 封顶通常是**症状**不是病因：某段本可外提的计算占了 UB | **L1.15**：先移出占 UB 的计算，再重扫 BLOCK。**直接删封顶会编译失败**（已实测） |
| **pse case 输出 77~96% 错 / 编译产物错乱** | pse 按实际维数分档：2D `view(1,1,1,-1)` 广播、4D `S=1` 广播、4D dense 三种语义不同；`[B,N,1,SK]` 一维 load + 广播加在 native dot 下编译产物错乱（smax 差 135） | **统一 2D load**（`offs_q*stride_s + offs_kv*stride_k`，`S=1` 时 stride=0 语义相同）；先探 golden 的 pse 广播行为（§6.1） |
| **`npu_*` op 输出 shape 不匹配、查不到原因** | `npu_fusion_attention` 返回 7 元组，`softmax_max/sum` 是 `[B,N,Sq,8]`（8 列同值）、`reserved` 空 | **§6.2 输出契约表**：按 7 元组契约分配输出，再写 kernel |
| 改了参考实现后老结论突然失效 | 参考的算术路径变了 | L1.4，先钉死契约再优化 |
| **fp16 near-zero 输出 MERE 爆炸（case MERE 0.2+）** | MERE 分母 `|golden|+1e-7`，softmax 近均匀（scale 小）→ 输出≈mean(V)≈0 时相对误差病态放大 | **CANN 契约：softmax probs 先 cast 到输入 dtype 再 PV dot**（L1.5「禁降 p」在本算子被证伪）；实测 case61 MERE 0.219→4.6e-6 |
| **sink case MERE 略超阈值** | sink 用 `l_init=1.0`（数学等价但舍入顺序不同） | **sink 契约：m_init=sink、l_init=0、末尾 `l += exp(sink-m)`**（对齐 CANN） |
| **大 D（512）UB 溢出 / 编译失败** | scores/p/acc fp32 三块 + k/v 双缓冲放大，编译器多缓冲使实际用量 ~1.7x 估计值 | **UB 预算实测校准**：`total = BQ*(BLOCK_D*4+8*BKV+BLOCK_D*elem) + 4*BLOCK_D*BKV*elem <= 160000B`；D=512 需 BQ=8/BKV=32 微块（BQ=16/BKV=32 溢出 1608192 bits） |
| **`npu_*` op 的 `.py` 内含同名纯 torch fallback** | fallback 只是 docstring 参考语义，**不是 golden**（探针实锤 CANN 行为） | 先 §6.1 CANN 语义探针钉死契约（p 先降 dtype、sink 末尾加、smax/ssum [B,N,Sq,8]），再谈算法与性能 |

---

## §7 FusionAttention（`npu_fusion_attention` 复刻）归档 2026-08-31

> 融合搜索 29 基因编码存档（R1 初始 OKF 基因 → R8 收敛 1.5933x）见
> [transformer_training__fusionattention.md](../operator_playbooks/transformer_training__fusionattention.md) §2.5。

### Layer 1 设计约束（硬性）

- **PV dot 必须 `p.to(input_dtype)` 再 dot**（fp32 归约）——对齐 CANN；L1.5「禁止降 p」在本算子**证伪**（p 保持 fp32 会让 near-zero 输出 MERE 病态）。
- **sink 契约**：`m_init=sink[n]`、`l_init=0`、KV 循环末尾 `l += exp(sink-m)`。禁止 `l_init=1.0`（舍入顺序不同致 MERE 超标）。
- **smax/ssum 恒 fp32 `[B,N,Sq,8]`**（8 列同值），`reserved` 空张量，seed/offset/mask_length=0。
- **mask 值必须精确 `-3.4028234663852886e38`（-FLT_MAX）**；sparse_mode 0/2/3/4 被 CANN 忽略（全可见），禁 KV 区间收缩；仅 sparse_mode=1 用 atten_mask。
- **UB 预算实测校准**（k/v 双缓冲 4x 放大）：`BQ*(BLOCK_D*4+8*BKV+BLOCK_D*elem) + 4*BLOCK_D*BKV*elem <= 160000B`；D=512 用 BQ=8/BKV=32 级微块。
- `forward` 返回 7 元组契约（attention_out, smax, ssum, empty, 0, 0, 0）。

### Layer 2 算法骨架

- **grid-stride persistent kernel**：外层任务 = (b,n,sq_block) 展平，`for task in range(pid, total, NUM_CORES)`；`NUM_CORES=24`（910b2）。
- **QBLOCK_MERGE=2**：同一 head 连续 2 个 q 子块共享一份 K/V tile load（大 case sq>=256 净胜，小 case 因 occupancy 回归）。
- **online softmax 延迟归一化**：`out = acc/l`（fp32），smax/ssum 用最终 m/l 写 8 列。
- **d64_s512 反向倾斜 G023**：BQ=128/BKV=128（dot M 维 64→128，缓解 Cube 半利用）。
- GQA：kernel 内 `kv_head = n // group_size`；k/v 按 nkv 头索引。

### Layer 3 关键技巧

```python
# PV dot：p 先 cast 输入 dtype（CANN 契约，L1.5 证伪点）
if IS_FP16:
    acc += tl.dot(p.to(tl.float16), v)          # fp32 归约
# sink 末尾加（m_init=sink/l=0 对应）
if HAS_SINK:
    l_i = l_i + tl.exp(sink_v - m_i)
# KV 循环流水线（num_stages=3 实测回归，固定 2）
for kv0 in tl.range(0, skv, BLOCK_KV, num_stages=2):
    ...
```

### 性能基准

| 版本 | 日期 | geomean | 精度 | 关键改动 |
|------|------|---------|------|---------|
| 本插件 iter_1（单 kernel） | 2026-08-31 | **0.6075x** | 61/61 | 精度契约对齐；UB 预算 tiler；无流水线/无合并 |
| **opt_iter_0（收敛）** | 2026-08-31 | **1.2441x** | 61/61 | QBLOCK_MERGE + BKV≤256 + tl.range 流水线 + G023 反向倾斜 + 24 核 |
| 前序 fusion-search R8 | 2026-08-30 | **1.5933x** | 61/61 | 同结构基因进一步收敛 |
| 复跑 iter_0（2026-09-01） | 2026-09-01 | **0.3119x** | 61/61 | 单 persistent kernel，D=128 BQ=32/BKV=64 |
| 复跑 opt_iter_2 | 2026-09-01 | **0.6047x** | 61/61 | D=128 retile BQ=64/BKV=64 + pn 按 D 分档 |
| 复跑 iter_0（2026-09-01 晚） | 2026-09-01 | **0.571x** | 61/61 | 标准流程单 kernel，BQ=64/BKV=64，D512 BQ=8（0.01x 病态） |
| 复跑 opt_iter_1（收敛） | 2026-09-01 | **0.5626x** | 61/61 | D512 BQ=16/BKV=16（5 case device -70%）+ 连续块；同负载 A/B 基线 0.503 → +11.8%，impl 延迟 -15.9% |
| 全新 fresh run iter_0 | 2026-09-03 | **0.6895x** | 61/61 | 从零单 persistent kernel（见 2026-09-03 补充）；host BNSD 视图无拷贝 + BQ/BKV 按 D 分档 + tl.range(2) |

### 2026-09-03 fresh run 补充（从零生成基线参考）

- 标准流程从零生成（不读 fusion-search Layer 4），单 `@triton.jit` persistent flash kernel 一次通过 61/61，geomean **0.6895x**（高于 2026-09-01 各 fresh rerun 0.31~0.60x）。关键正确做法：host 侧 `view/transpose/permute` 转 BNSD（**禁 `.contiguous()`**）、输出按原生 layout 连续写回（不用 q 的 stride）、pse size-1 维 stride 置 0 实现 broadcast、GQA `kv_head = n // group`、BLOCK_D=ceil16(D)。
- **propagate_nan 按 kv_blocks>1 & D<512 分档**在本 61-case 集上净偏负（0.6895→0.6646，落在 ±5% 噪声带内），未复现 2026-09-01 的 +9%；**CV 重排（dot 提前/sum 延后）同样噪声带内中性**（→0.6697）。→ 此 shape 集 FA 微调收益 < 测量噪声，需靠结构性改动或 simulator 证据，勿再单点微调。
- 确认结构性上限（与 §7 一致）：d512 微 dot M（BQ16/BKV16 已是最优逃逸，仍 0.04~0.22x）、d64_s512 Cube 半利用 0.31x、大 B*N 同步开销。

- **结构性上限**：Triton 复刻 torch_npu 原生融合算子存在硬件天花板（d64_s512 Cube 半利用、d512 M=32 微块；simulator 实测 MMAD 12% + Cube↔Vector 串行）。物化解耦（G022 split-KV）、num_stages=3、BKV 超 UB 上限均证伪/无收益。

### 2026-09-01 复跑补充（非显然新洞察）

- **propagate_nan 必须按 dot M 维分档，不能只按 KV 块数**（§2.8 只提块数）：D<512（M≥64）开 `propagate_nan=ALL` +9%；**D=512（M=8 微 dot）开 ALL 均匀 -42%**（5 case 全部）——BAR 负收益压过单条 vmax 节省。分档后 0.5794 → 0.6047。
- **GQA 头合并触发编译器缺陷**：`q_tile` 用 `n_full[:,None]*qn` 的 tensor 化头索引（结构化 gather）与 `tl.range(num_stages=2)` KV 循环组合 → aicore 超时 507014，全部 case 挂起。最小原语（product arange、独立 gather）单独均正常——是组合缺陷。合并头以提升 dot M（8→32）的路线在 910B2 当前编译器上不可行。
- **大 BQ 可致边缘 MERE 微超**：D=64 用 BQ=128（per-head）致 case54 fp16 MERE 1.055e-3 微超阈值（9.77e-4），回退 BQ=64/BKV=128 恢复。**FP 边缘 case 微调 tile 后必须全量 verify**，不要假设"数学等价即精度不变"。
- **连续 vs 交织任务分配无显著差异**：连续块（每核取连续 task 区间）与交织（`range(pid,total,cores)`）geomean 相当；checklist #6 要求的连续块无性能代价。
- **D=128 BLOCK 开到 UB 上限是最大单点杠杆**：BQ=32→64（106KB→147KB）任务数减半，D=128 geomean 0.325→0.765（+135%）。
- **D512 GQA BQ=8/BKV=32 是 0.01x 病态，BQ=16/BKV=16 可救回**：D=512 用 BQ=8（dot M=8 微 dot）在标准流程 benchmark 下 5 case 仅 0.01~0.2x；改 BQ=16/BKV=16（UB 预算 116KB，dot M 8→16）后 5 case device 时延直测 -70%（如 case39 5.41ms→1.47ms）。BQ=8 的单 dot M=8 是 Cube 半利用的极端形态。
- **Layer 1 新增：out_bnsd store 必须用连续偏移** `(b*NQ+n)*(SQ*D)+s*D+d`，**禁止复用输入 q 的 stride**：q 在 BSH/SBH/BSND 下是 transpose/view 非连续视图（BSH→[B,N,S,D] stride=(S*H,D,H,1)，q_s_stride=H=N*D），用其 stride 写新分配连续 out_bnsd 会全部错位。**判别信号：验证失败集合严格按 layout 划分（BNSD 全过 / BSH·SBH·BSND 全挂），smax/ssum 全对仅 attention_out 错**。kernel 用到 shape 标量 D 时必须显式传入运行期 D（constexpr BLOCK_D 不可替代，否则 `NameError: D is not defined`）。
- **QBLOCK_MERGE=2 在 BAR-bound 结构下不减少 BAR 数**：共享 K/V 只省 HBM 访存，不降「循环内向量算子数 × KV 迭代数」，直测中性偏劣（+4.8%，dot M 64→32 更亏）。对 Cube↔Vector 同步主导的 FA 主链，QBLOCK_MERGE 收益需其他条件（如 memory-bound）才成立。
