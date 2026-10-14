---
type: Guide
title: mla 类算子设计约束（MultiHeadLatentAttention 矩阵吸收）
description: MLA 形态识别与矩阵吸收主链（ckv 同时作 K 与 V、q 拆 nope/pe 双 dot、分页/块表间接寻址）的判别与 Layer
  1 硬约束；Ascend 可证仿射是头号约束。
tags: [task.design, topic.attention, topic.compute.composite, topic.precision, topic.implementation.masking, topic.compute.normalization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:23:14Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# mla 类算子设计约束（MultiHeadLatentAttention 矩阵吸收）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；
> attention 主链约束（online softmax、BLOCK 面积预算、精度闸门、测量口径）继承
> [flash_attention_constraints.md](flash_attention_constraints.md)；
> 家族内写法类路由见 [attention_writing_classification.md](attention_writing_classification.md)。

本卡负责 **MLA（Multi-head Latent Attention）矩阵吸收主链**这一族算子（BatchMLAPagedAttention /
BatchDecodeMlaPaged / TRT-LLM MLA / DSV4 稀疏 MLA / vLLM MLA）：
`ckv` 同时作 K 与 V，`o = softmax(sm_scale·(q_nope·ckv + q_pe·kpe)) @ ckv`，q 拆 `nope`/`pe` 双 dot
（或融合维 kernel 内 constexpr 切分）＋ 分页/块表间接寻址。
**核心优化哲学**：头号约束是 Ascend 后端的**可证仿射（provable affine）**——任何由运行时
`div`/`mod` 推导的行号都会触发逐元素标量 scatter（单 `[16, Dc]` store 毫秒级、甚至间歇死循环）。
**宁可缩小 `BLOCK_QO` 换仿射，也不要多 query 混排。**
生成时**禁止混用**其他类别经验——尤其不要套用 `flash_attention.md` 的 `BLOCK_Q` 大 tile / `kv_lo` 对齐策略
（两卡差异声明见源 §9）。

## 本类别的算子级实现手册

- [`operators/mla__multi_head_latent_attention.md`](../operator_playbooks/mla__multi_head_latent_attention.md) — MLA 类算子的 Layer 2 五骨架（paged 主骨架 / decode / sparse / preprocess / vector 归约）、Layer 3 关键技巧与陷阱表

## §0 适用范围与算子分类

子类标签：`mla-paged`（FlashInfer 主样例形态）、`mla-paged` decode 特化、`mla-sparse`（DSV4 双池）、
`mla-vllm`（非吸收边界算子）、`mla-preprocess`（上游 GEMM 生成阶段）。布局差异三种：**拆分双张量**、
**融合单张量**（head_dim 拼接 `kv_lora_rank+qk_rope_head_dim=576`，kernel 内切分）、**双池稀疏**（`-1` 槽掩码）。

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件；其中**矩阵吸收是硬判别器，先于其他 attention 家族（mha / gqa / flash_attention）排序**：

1. **矩阵吸收**：`ckv` 同时作 K 与 V，`o = softmax(scores) @ ckv`（不是 `@V` 独立张量）；
2. q 拆 `nope`/`pe` 双 dot（或融合维 kernel 内按 constexpr 切分）+ 分页/块表间接寻址（虚拟位置 → 物理 cache 行）；
3. `lse` 输出（底数逐字对照参考 kernel：带 `return_lse_base_on_e`/`log2` 语义 → base-2；参考用 `tl.log` → base-e，见 F8/L1.8）。

**Q0（先用本文件前的第一问）是否为吸收主链**：`mla-vllm`（vLLM 风格非吸收变体）K 用全 64 维、V 取前
`headdim_v` 维**前缀切片**，为**边界算子**——共享分页寻址与 UB/仿射经验，但主链数学不同，生成时需单独核对参考语义；
`mla-preprocess`（MlaPreprocessOperation）是上游 GEMM 生成阶段，不在本卡主链。
不满足上述任意一条（如无分页的连续寻址 `Q@Kᵀ→softmax→@V`）→ 退回 `flash_attention.md`。

### §0.2 ★ 形态识别四问（**Phase 2 第一步必须回答，答案决定后续哪些章节适用**）

| # | 问题 | 影响 |
|---|------|------|
| **Q0** | 是否为吸收主链（`ckv` 作 V）？ | 否 → `mla-vllm` 类边界算子，单独核对参考语义；`mla-preprocess` 走 GEMM 骨架 |
| **Q1** | 双张量拆分 or 融合维（kernel 内切分）？ | 融合维 → 布局特化；`query [..,576]` 按 constexpr 切 q_nope/q_pe |
| **Q2** | 页表布局（indices 乱序 / block_table 稠密 / sparse `-1` 掩码）？ | 决定仿射约束与稀疏 gather 骨架 |
| **Q3** | 参考精度（fp32 全程 or 原生 dtype）？ | 决定 `P_SPLIT` 是否必开（16-bit 输入） |
| **Q4** | `q_len` 是否 >1（增量因果掩码）？ | 是 → 主循环变二维 `[q_len, kv]`，掩码按 constexpr 特化 |

## §1 通用经验（跨形态，首次生成必须遵守）

### F1 ★ 仿射寻址 > 一切：`BLOCK_QO=1` + GH head 共享

MLA decode 的正确姿势是**一个 program 固定处理同一 request 的 GH 个 head，`BLOCK_QO=1`**：
`rows = tl.arange(0, BLOCK_M)` 直接作 head 索引，`h = h_group * GH + rows` 为**仿射**、`q_row` 为**标量**，
于是所有 Q/O/LSE 的 load/store 对 Ascend 后端**可证仿射**，回到向量化 store。
任何由 `rows//GH`、`rows%GH` 推导的行号即使恒为 0 也会触发逐元素标量写（L1.1 决定性证据）。

### F2 页表映射必须 kernel 内做，`BLOCK_KV` 与 `page_size` 解耦

虚拟 KV 位置 → 物理 cache 行在 kernel 内映射：`(pages[j // (page_size*H)] * page_size + (j // H) % page_size) * H + j % H`
（C 序 head 交错展平）。`BLOCK_KV` 按 UB 预算自适应，**不得依赖** `page_size` 对齐。

### F3 KV 复用：一个 program 服务 GH 个 head，KV tile 只 load 一次

`BLOCK_QO=1` 的收益来源——GH 个 head 共享同一 KV tile，KV 只加载一次（生成版是每个 head 各自扫一遍 KV）。

### F4 16-bit dot + fp32 累加；fp32 输入保持 fp32 dot

对齐参考算术路径：fp32 输入走 fp32 dot，不要强行降到 16-bit。

### F5 ★ 16-bit PV 精度判据式：先单 f16 dot，超差再 P_SPLIT 双点积（非 blanket）

16-bit 输入**先试单次 f16 PV dot**（`p.to(f16)`，尾数 11bit，多数场景够用）直接过验证闸门；
只有单 dot 超差（紧闸门 / 大数值范围）才用第二次 cube dot 补回（`p_hi + p_lo` 双点积，对齐 fp32 全程）。
**不要一上来就双点积**——省一次 cube dot。

### F6 split-KV 长 KV 才开 + reduce 标量循环

`ns = max(1, min(8, max(ns_par, ns_kv)))`；长 KV 切到多个 program，小 reduce kernel 归并 partial `(acc, m, l)`。
reduce kernel **禁止** 3D masked load（Ascend 后端触发 vector core timeout），改用 `for s in range(num_splits)` 标量循环。

### F7 UB 预算静态估算自适应（含编译器额外缓冲余量）

`esz*(bm*Dc + bm*Dp + 2*bkv*(Dc+Dp)) + 4*(bm*Dc + 2*bm*bkv) + (2*bm*bkv*esz if P_SPLIT else 0)` < 140KB；
bm 从 32/64（Dc≥512 取 32）往下减、bkv 从 128 往下取，取首组满足预算的配置。

> ⚠️ **预算必须含编译器额外缓冲**：triton-ascend 编译会再分配 ~10+KB 临时缓冲，静态估算贴满 140KB
> 的配置会超 UB 编译失败；实测 ~12KB 是成败分界。

### F8 ★ lse 底数契约 = 逐字对照参考 kernel（禁止默认 base-2）

- 参考用 `tl.exp`/`tl.log`（base-e，与 torch 参考 `torch.log` 一致）→ **全程 base-e**：内部 `p = tl.exp(s - m_new)`、
  `lse = m + tl.log(l)`，**禁止 ×LOG2E**（曾有实现因 ×LOG2E 把 base-e 输出变 base-2，MERE 指纹 ≈ log2(e)−1 = 0.442695）。
- 参考用 `exp2`/`log2` 或带 `return_lse_base_on_e` flag（FlashInfer 约定）→ 内部 base-2 + LOG2E，输出按 flag 归一。
- kernel 内只能用 `tl.log`/`tl.exp`（triton 自带自然底），**禁止 `torch.*` 接口**。

### F9 掩码区间收缩：KV 循环 trip 从容量全量缩到每 query 有效长度（paged/sparse 通用）

带掩码的 KV 扫描不要傻扫 `capacity` 全量：SWA 段 `range(0, n_swa, BLOCK_N)`、压缩池段
`range(prefix_len, topk_len, BLOCK_N)`（前缀固定偏移）、decode 段直接 `range(0, kv_len, BLOCK_N)`。
**收缩后尾迭代的掩码仍不可删**（精度红线，越界列 / `-1` 槽仍必须显式 mask）。

## Layer 1 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 ★★★ `BLOCK_QO=1` 仿射寻址（决定性）

- **必须** 固定 `BLOCK_QO = 1`，`h = h_group * GH + rows`、`q_row` 标量，全部 Q/O/LSE load/store 对 Ascend **可证仿射**。**禁止**多 query 混排 / `rows//GH` 取行号。
- **Why:** Ascend 后端将无法证明仿射的 store（行号由 `rows//GH` 等除法/取模推导，即使恒为 0 也无法折叠）lowering 为逐元素标量写，且该路径间歇死循环；判别：profiling 单 store 毫秒级或间歇 aicore timeout。这是端到端最大收益的一步。
- **How to apply:** `rows = tl.arange(0, BLOCK_M)` 直接作 head 索引，`q_row` 为标量 → 回到向量化 store。

#### L1.2 页表映射与 `BLOCK_KV` 解耦；越界页显式掩码；索引公式逐字复刻参考

- **必须** 虚拟位置 → 物理 cache 行在 kernel 内映射（int32），`BLOCK_KV` 按 UB 预算自适应、**不得依赖** `page_size` 对齐；越界页 `other=0` 显式掩码；**展平索引逐字复刻参考**（torch 参考 `reshape(-1, Dc)[:kv_len]`，或参考 kernel 的 `PAGE_SIZE*H` 展平循环）。
- **Why:** 页可乱序、可冗余；越界页未掩码产生"幽灵列"（padding 陷阱）。展平错位是最频精度失败（matched_ratio≈0 且 H>1 全崩）。
- **How to apply:** 虚拟位置 j → `(page=j//(page_size*H), offset=(j//H)%page_size, head=j%H)`（F2 公式）。**head 交错 cache（`H_cache == H` 且逐 head 独立数据）下每个 head-slot 都是序列中的独立 KV 位置**；`GH` 分组"共享 KV / load head-0 only"**仅当 cache head 维 == 1（跨 head 共享）才允许**——head 交错 cache 下禁止 head-0 替代整组（曾有实现只 load head-0 → H>1 全量失败）。首次生成必须读参考的 reshape/切片/permute 推导索引。

#### L1.3 精度契约先钉死：fp32 输入 fp32 dot；16-bit 输入按 F5 判据

- **必须** 对齐参考算术路径：fp32 输入走 fp32 dot；16-bit 输入先跑单 f16 PV dot，闸门超差才补 `p_hi+p_lo` 双点积。
- **Why:** 复制 FA 卡"fp32 误用 16-bit dot"经验会把 fp32 输入降精度；精度失败通常集中在特定 dtype。
- **How to apply:** `p.to(f16)` 单 dot 过闸门，超差（紧闸门 / 大数值范围）才开 `P_SPLIT`（F5）。

#### L1.4 KV 复用（GH 共享）

- **必须** 一个 program 处理同一 request 的 GH 个 head，共享每个 KV tile。
- **Why:** `BLOCK_QO=1` 的收益来源——KV 只 load 一次。
- **How to apply:** grid 按 `(num_requests, num_splits)`，program 内 GH 个 head 复用同一 KV tile。

#### L1.5 split-KV + reduce 标量循环

- **必须** 长 KV 必开 split-KV；reduce kernel 用 `for s in range(num_splits)` 标量循环。
- **Why:** reduce kernel 的 3D masked load 在 Ascend 后端触发 vector core timeout（死循环）。
- **How to apply:** 长 KV 切到多个 program，小 reduce kernel 归并 partial `(acc, m, l)`。

#### L1.6 mask constexpr 特化

- **必须** `causal` / 增量因果（`j > kv_len - qo_len + i`）/ `window_left` / `logits_soft_cap` / `softcap` 全部作为 `tl.constexpr` 传入，禁止运行时标量分支。
- **Why:** 运行时标量分支在 Ascend 前端不可靠；constexpr 特化可让 decode 单 query 主循环折叠。
- **How to apply:** mask 属性走 constexpr 参数下发，kernel 内按 constexpr 分支。

#### L1.7 UB 预算自适应（<140KB 静态估算）

- **必须** 用 F7 公式静态估算，且预算留出编译器额外缓冲余量（~10+KB）。
- **Why:** triton-ascend 编译会再分配临时缓冲，贴满 140KB 的配置会超 UB 编译失败（实测 ~12KB 是成败分界）。
- **How to apply:** bm/bkv 双层循环取首组可编译配置（bm 从 32/64、bkv 从 128 往下取）。

#### L1.8 lse 底数逐字对照参考（F8）

- **必须** 参考 base-e（`tl.log`）→ kernel 内 `tl.exp`/`tl.log`，禁止 ×LOG2E；参考 base-2 / 带 `return_lse_base_on_e`（FlashInfer 约定）→ 内部 LOG2E，输出按 flag 归一。
- **Why:** ×LOG2E 会把 base-e 输出变 base-2，MERE 指纹 ≈ 0.442695。
- **How to apply:** kernel 内只用 `tl.log`/`tl.exp`，禁止 `torch.*` 接口。

#### L1.9 host 不做 permute / contiguous / pad

- **必须** 分页语义下禁止 host 侧任何重排（permute / contiguous / pad）。
- **Why:** host 侧重排破坏虚拟位置 → 物理行的映射。
- **How to apply:** 分页/块表 KV 以原始布局传入。

#### L1.10 融合维切分：kernel 内按 constexpr 切 `nope`/`pe`

- **必须** `query [.., 576]` / `kv_cache [.., 64]` 在 kernel 内按 constexpr 切 q_nope/q_pe（融合维布局），host 不拆张量。
- **Why:** TRT-LLM 等把 `kv_lora_rank`+`qk_rope_head_dim` 拼成融合维，拆分只是布局差异，核心吸收模式相同。
- **How to apply:** 融合维时用 constexpr 下标的二维 load 分别取 nope/pe 段。

#### L1.11 掩码用有限常量；softcap 用 native `tl.math.tanh`

- **必须** 掩码一律用有限常量 `NEG_INF = -1.0e9`（或 -3.0e38），**禁止 `float("-inf")`**；`logits_soft_cap > 0` 时打分后/softmax 前的 tanh **一律用 `tl.math.tanh`（native 饱和）**，**禁止 exp 公式手写 `_tanh(x) = (e^x−e^{-x})/(e^x+e^{-x})`**。
- **Why:** 全掩码 tile/行（window 前置区、GH 组内被 mask 的 head 行）在 online softmax 中 `exp(-inf - (-inf))` 产生 NaN；exp 公式手写 tanh 遇 NEG_INF 大负值（x = −1e9/cap ≈ −2e7）使 `e^{|x|}=inf` → `-inf/inf = NaN` → 经 `tl.max` 传播整 tile 全 NaN。
- **How to apply:** 掩码用 `-1.0e9`；softcap 用 `tl.math.tanh`（tanh(大负)→-1 不溢出）。

#### L1.12 ★★★ cube dot 的 M 维分界：M=1 禁区（vector 归约），M≥8 的 head 组可用 cube dot

- **必须** 禁止不设防的 M=1 cube `tl.dot`（单头 `H=1`，或 `GH` 被 UB/语义钳到 1）。**二选一**：① 保证 head-group 落点 ≥ 8（head 数足时 `BLOCK_QO=1` + GH 共享天然满足，GH=8 实测稳定通过且最优）；② 整体改走 vector 归约（`grid = (total_q * H,)` 每 program 单 head，QK/PV 全用 `tl.sum(逐元素乘, axis=…)`，**零 `tl.dot`**）。
- **Why:** Ascend cube `tl.dot` 的 M 维分界是 **M=1 才是禁区**——整组只有 1 行的 dot 在 aicore 上**挂死**（间歇 `ACL stream synchronize failed 507014`，全量 timeout）或严重退化。**不要把"M=1 禁区"泛化成"所有 M<16 都不能用 dot"**：head 组 M≥8 的 cube dot 完全可用，且比 vector 归约快。
- **How to apply:** 判定：`H` 单头，或 `GH` 被钳到 1 → 直接走 vector 归约骨架；否则 `BLOCK_QO=1` + GH≥8 的 cube dot 可用。

## §常见陷阱与证伪方向

| 陷阱 | 原因 | 避免方法 |
|---|---|---|
| 仿射 store 逐元素化 | 行号由 `rows//GH` 等推导，无法证明仿射 | `BLOCK_QO=1` 固定（L1.1） |
| 页表越界幽灵列 | 页可乱序可冗余，越界页未掩码 | `other=0` 显式掩码（L1.2） |
| 3D masked load 死循环 | Ascend 后端 vector core timeout | reduce 用标量循环（L1.5） |
| `BLOCK_QO>1` 多 query 混排 | 想省启动，触发非仿射 store | 宁可缩小 BLOCK_QO 换仿射 |
| fp32 输入误用 16-bit dot | 复制 FA 卡经验 | fp32 保持 fp32 dot（L1.3） |
| head 交错展平索引错位 | GH 分组只 load head-0，忽略 head 维参与序列 | 位置公式逐字复刻 F2；用 H=1 case 快速 sanity |
| `float("-inf")` 掩码 NaN | 全掩码 tile/行 exp(-inf-(-inf)) | 掩码统一有限 NEG_INF（L1.11） |
| lse 底数照搬 base-2 | 卡片默认 base-2，参考用 `tl.log`（base-e） | 用 `tl.log` 直接输出、去 ×LOG2E（F8） |
| softcap 手写 exp 公式 tanh 溢出 NaN | `e^大正数=inf` → `-inf/inf=NaN` 经 `tl.max` 传播整 tile | softcap 一律 `tl.math.tanh`（L1.11） |
| M=1 的 cube dot 挂死 | 单头 GH=1 或 head-group 落点=1 | 保证 GH≥8 或整体 vector 归约（L1.12） |
| 掩码区间收缩后删尾迭代掩码 | 想省空迭代，越界列 / `-1` 槽未 mask | 收缩 trip 后尾迭代掩码保留（F9） |
| UB 预算贴满 140KB 编译失败 | 未计编译器额外缓冲（~10+KB） | 静态估算留余量（F7） |
| `tl.dot` 2D 结果做标量索引 `[0]` | Ascend 前端不支持 tensor 索引 constexpr[0]（编译失败） | 小 M 直接用 vector 归约 1D 结果，不做 dot 2D 索引 |
| reduce kernel 病态 | ns=1 case 归并路径慢 | 按 Name 分组定位 + split-KV 参数自适应 |

**⛔ 结构性下限**：小 shape 上**绝不要**拆 kernel——MIX 每次发射 ~4.5us 地板；decode 单 request 数据量小时拆多
kernel 只会被启动开销吃光（同 FA 卡 §5.3 访存账结论）。
