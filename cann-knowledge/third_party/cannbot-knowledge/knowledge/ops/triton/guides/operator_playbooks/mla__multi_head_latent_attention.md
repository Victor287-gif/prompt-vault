---
type: Guide
title: MLA 类算子（MultiHeadLatentAttention） 实现手册
description: '典型特征: ckv 同时作 K 与 V 的矩阵吸收，q 拆 nope/pe 双 dot + 分页间接寻址'
tags: [task.implementation, topic.attention, topic.compute.composite, topic.precision, topic.implementation.masking]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:23:14Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# MLA 类算子（MultiHeadLatentAttention） 实现手册

> 类别约束见 [../mla_constraints.md](../constraints/mla_constraints.md)（L1.1-L1.12 全部硬约束），
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §2 MLA 类算子（mla）

**算子类别**: `mla`　**典型特征**: ckv 同时作 K 与 V 的矩阵吸收，q 拆 nope/pe 双 dot + 分页间接寻址
**性能基准**: 证据基础 `BatchMLAPagedAttention` 全量 case 完整优化轨迹——基线（reduce kernel 平均耗时病态）
→ 结构重写（片上 acc + KV 共享 + split-KV + UB 自适应 + P_SPLIT）→ **`+ BLOCK_QO=1` 仿射化修复（决定性一步）**。
Ascend910B2C，triton-ascend 3.2.x / CANN 8.5.x。（源 template 未给出单数几何平均，不虚报。）

### Layer 1 设计约束（算子专属）

Layer 1 全量硬约束 L1.1-L1.12 见 [../mla_constraints.md](../constraints/mla_constraints.md)，本算子生成时**必须全部满足**。
最关键的 ★★★ 两条：

- **L1.1 `BLOCK_QO=1` 仿射寻址（决定性）**：`h = h_group*GH + rows`、`q_row` 标量，全部 Q/O/LSE load/store
  对 Ascend 可证仿射；**禁止**多 query 混排 / `rows//GH` 取行号。
- **L1.12 cube dot M 维分界**：M=1 是禁区（挂死 507014）——单头/head-group 落点=1 时整体走 vector 归约；
  GH≥8 的 head 组 cube dot 完全可用且最优。

### Layer 2 算法骨架（参考方向，输出必须是全新草图）

#### §3.1 主骨架（`mla-paged`，含 decode 特化）

```text
grid = (num_requests, num_splits)
program:
  load q_nope/q_pe [BLOCK_QO=1, Dc/Dp]（逐 request）
  m/l/acc 初始化为 -inf / 0 / 0（fp32 片上）
  for j in range(0, kv_len, BLOCK_KV):
    虚拟 KV 位置 → 页表映射 → 物理 cache 行（kernel 内，int32）
    load ckv tile / kpe tile
    s = dot(q_nope, ckvᵀ) + dot(q_pe, kpeᵀ)；s *= sm_scale
    mask 特化（causal / window / softcap，constexpr）
    m_new = max(m, rowmax(s))；p = exp((s - m_new))（base-e 对照参考，F8）
    l = l*exp((m - m_new)) + rowsum(p)；acc = acc*缩放 + dot(p, ckv)
  split-KV：写 partial (acc, m, l) → reduce kernel 归并
  输出 o [BLOCK_QO=1, Dc]；lse 底数逐字对照参考（F8）
```

**decode 特化**：无因果、可 window 收缩（`window_left` 区间折叠）、`logits_soft_cap` 用 `tl.math.tanh`
（native 饱和，softcap 在打分后、softmax 前；**禁止 exp 公式手写 tanh**——NEG_INF 大负值溢出 NaN，L1.11）。
结构**几乎直接复用 paged 主骨架**；`q_len>1`（MTP / spec-decode）时主循环变二维 `[q_len, kv]` + 增量因果掩码。

#### §3.3 `mla-sparse` 稀疏 gather 骨架

SWA 128 前缀 + 压缩池**双池**；`sparse_indices` 扁平槽索引，`-1` 索引掩码为 0（不参与 softmax）。
页表布局决定仿射约束（`mla_constraints.md` §0.2 Q2）。

#### §3.4 `mla-preprocess` GEMM 骨架（上游生成，非吸收主链）

低秩投影（`q_lora_rank`/`kv_lora_rank`）+ RMSNorm + GPT-J RoPE，GEMM 为主。

> **GEMM 链性能注意（有成功先例，大 B 仍是重灾区）**：多 kernel 拆解（GEMM→RMSNorm→GEMM→RMSNorm→RoPE，
> 中间 buffer 落盘 GM）在小/中 B 有明确收益；大 B（4096-8192）仍因**小 tile + 大 grid** 剧烈劣化
> （固定小 BLOCK 时 grid 可达数千 program，调度/发射开销主导；K 循环内反复重载逐行归一化系数放大约束）。
> 成功配置方向：**grid 封顶 + 持久化交织循环**（`grid=(min(tiles_m*tiles_n, cube),)` + `for tile in range(pid, total, nprog)`；
> **禁止无界 2D grid**——program 数随 M×N 二次膨胀时 tile 数与核数不匹配）；**BLOCK_M 按 T 自适应**
> （小 32 / 中 64 / 大 128）、BLOCK_N=128、BLOCK_K=256；RoPE 用**奇偶列分段 2D load + `tl.interleave`**；
> RMSNorm **两遍扫描**（先整段 sum_sq 求 rstd 再乘，与 ln 权重融合）；数学函数统一自然底 `tl.exp`/`tl.log`，
> 位置与频率计算走 fp32。复测对比 torch 批量 `aclnnMatmul`。

#### §3.5 `mla-paged` 小 M vector 归约骨架（单头 / head-group 落点 =1 时，L1.12 路径 2）

```text
grid = (total_q * H,)                      # 每 program 一个 (q_token, head)，零 tl.dot
program:
  q_nope_vec = load(q_nope[pid//H, pid%H, :Dc]).to(fp32)   # 1D 向量
  q_pe_vec   = load(q_pe[pid//H, pid%H, :Dp]).to(fp32)
  m/l/acc = -inf / 0 / zeros([BLOCK_DC])                    # acc 1D，PV 无 dot
  for sk in range(0, kv_len, BLOCK_SK):                     # BLOCK_SK = page_size (≤32)
    sk_offs = sk + arange(BLOCK_SK)
    页表映射 → ckv_tile [BLOCK_SK, Dc]、kpe_tile [BLOCK_SK, Dp]（kernel 内，int64 地址）
    nope_score = tl.sum(q_nope_vec[None,:] * ckv_tile, axis=1)   # QK = vector 归约
    pe_score   = tl.sum(q_pe_vec[None,:] * kpe_tile, axis=1)
    score = sm_scale * (nope_score + pe_score); 掩码有限 NEG_INF（L1.11）
    s2 = score * LOG2E; m_new = max(m, max(s2)); p = exp2(s2 - m_new)
    acc = acc * exp2(m - m_new) + tl.sum(p[:,None] * ckv_tile, axis=0)   # PV = vector 归约
  输出 o = acc / l_safe；lse = m + log2(l_safe)（底数对照参考，F8）
```

`BLOCK_SK` 取 `page_size`（16/32）保仿射；UB 内放 `[BLOCK_SK, Dc]` tile，不需 cube。

### Layer 3 关键技巧（可参考不可照抄，变量名/结构必须重新设计）

- **★★★ 仿射 store 修复（决定性，SINGLE 开关 A/B）**：Ascend 后端将无法证明仿射的 store（行号由 `rows//GH`
  等除法/取模计算，即使恒为 0 也无法折叠）lowering 为逐元素标量写（且该路径间歇死循环）。修复：`BLOCK_QO=1`
  固定，`h = h_group*GH + rows` 为仿射、`q_row` 标量 → 回到向量化 store。
- **KV 复用（GH 共享）**：`BLOCK_QO=1` 的收益来源——一个 program 服务 GH 个 head，KV tile 只 load 一次。
- **P_SPLIT 双点积（16-bit 输入）**：`p` 转 16-bit 的精度损失用第二次 cube dot 补回——`p_hi`（高段）+ `p_lo`
  （残差）双点积，对齐 fp32 全程。仅单 dot 超差时开（F5 判据式，非 blanket）。
- **host 侧倒数乘法（sm_scale）**：`sm_scale` 用 host 侧倒数乘法传入，kernel 内不做 fp32 张量除法。
- **BLOCK 开到 UB 上限**：F7 预算内把 `BLOCK_KV` 取到最大压迭代数；**但 `BLOCK_QO` 恒为 1**（L1.1）。
- **vector 归约替代 cube dot（小 M 专用，L1.12 路径 2）**：`BLOCK_QO=1` 单 query 时 QK/PV 可整体改 vector FMA
  归约，`acc` 降为 1D（`nope_score = tl.sum(q_nope_vec[None,:]*ckv_tile, axis=1)`；PV 同理 axis=0）。与 GH≥8 的
  cube 版相比是**降级**，**只在 M=1 时用**——此时 cube 是地雷。⚠️ **Ascend 前端陷阱**：`tl.dot` 返回的 2D tensor
  **不支持标量索引 `[0]`**（`ValueError('unsupported tensor index: constexpr[0]')`）——曾有实现想用 `USE_DOT`
  开关把 vector 归约改回 `tl.dot` 提速，编译即失败，回退 vector 版；优化时不要走这条路。
- **页对齐大块加载（剩余空间方向）**：`BLOCK_KV` 受 UB 限制（Dc≥512 fp32/bf16 大 case，KV 扫描本身 dominate，
  bkv≈32），可考虑页对齐大块加载（一次拉整页）进一步降低迭代开销（未验证）。

### 性能基准 / 优化记录表

| 优先级 | 方向 | 作用/适用 |
|---|---|---|
| 1 ★★★ | 仿射 store 修复（BLOCK_QO=1） | 消除非仿射 store 病态（核心约束） |
| 2 ★★ | KV 复用（GH head 共享） | 结构重写阶段的组成项 |
| 3 ★★ | 16-bit PV 单 f16 dot，超差才 P_SPLIT（F5） | 精度过闸门的前提（判据式） |
| 4 ★ | split-KV + reduce 标量循环 | 长 KV 病态消除 |
| 5 ★ | UB 预算自适应 | 让 bkv 取到上限，压迭代数 |
| 6 ★ | M=1 用 vector 归约替代 cube dot（L1.12） | 规避 M=1 cube dot 禁区 |
| 7 | 页对齐大块加载 | 剩余空间方向，未验证 |

### 陷阱 / 证伪方向

| 陷阱 | 原因 | 避免方法 |
|---|---|---|
| 仿射 store 逐元素化 | 行号由 `rows//GH` 等推导，无法证明仿射 | `BLOCK_QO=1` 固定（L1.1） |
| 页表越界幽灵列 | 页可乱序可冗余，越界页未掩码 | `other=0` 显式掩码（L1.2） |
| 3D masked load 死循环 | Ascend 后端 vector core timeout | reduce 用标量循环（L1.5） |
| reduce kernel 病态 | ns=1 case 归并路径慢 | 按 Name 分组定位 + split-KV 参数自适应 |
| `BLOCK_QO>1` 多 query 混排 | 想省启动，触发非仿射 store | 宁可缩小 BLOCK_QO 换仿射 |
| fp32 输入误用 16-bit dot | 复制 FA 卡经验 | fp32 保持 fp32 dot（L1.3） |
| head 交错展平索引错位 | GH 分组只 load head-0 的 KV，忽略 head 维参与序列 | 位置公式逐字复刻 F2；用 H=1 的 case 做快速 sanity |
| float("-inf") 掩码 NaN | 全掩码 tile/行（window 前置区、GH 被 mask 的 head 行）exp(-inf-(-inf)) | 掩码统一用有限 NEG_INF（-1.0e9 / -3.0e38） |
| lse 底数照搬 base-2 | 卡片默认 base-2，但参考 kernel 用 `tl.log`（base-e）且无 `return_lse_base_on_e` | 用 `tl.log` 直接输出、**去掉 ×LOG2E**；仅参考带 base-2/`base_on_e` 语义时才 ×LOG2E。kernel 内禁止 `torch.*` 接口 |
| softcap 用自定义 exp 公式 tanh 溢出 NaN | `logits_soft_cap>0` 且 NEG_INF 大负值进入 `_tanh(x)=(e^x−e^-x)/(e^x+e^-x)`，`e^大正数=inf` → `-inf/inf=NaN` | softcap 一律用 `tl.math.tanh`（native 饱和，tanh(大负)→-1 不溢出）；禁止 exp 公式手写 tanh |
| M=1 的 cube dot 挂死 | 单头 GH=1 或 head-group 上限被钳到 1，dot M 维只有 1 行 | 保证 GH≥8 或整体改 vector 归约（L1.12 / §3.5） |
| 掩码区间收缩后删尾迭代掩码 | 想省空迭代，越界列 / `-1` 槽未 mask | 收缩 trip 后尾迭代掩码保留（F9 / §6） |
| UB 预算贴满 140KB 编译失败 | 未计编译器额外缓冲（~10+KB） | 静态估算留余量（F7） |
| `tl.dot` 2D 结果做标量索引 `[0]` | Ascend 前端不支持 tensor 索引 constexpr[0]（编译失败） | 小 M 直接用 vector 归约 1D 结果，不做 dot 2D 索引 |

**⛔ 结构性下限（§5.2）**：小 shape 上**绝不要**拆 kernel——MIX 每次发射 ~4.5us 地板；decode 单 request
数据量小时拆多 kernel 只会被启动开销吃光（同 FA 卡 §5.3 访存账结论）。

**会误导的 profiling 字段（§5.3）**：
- **profiler 伪影实为真慢**：偶发 aicore timeout / 设备 hang 常被归为"伪影"，实为逐元素标量 store 的真实耗时。
- 病态耗时集中在 **ns=1 的 fwd 和多数 case 的 reduce**——先按 Name 拆开看，不要整 kernel 调。
- **aicore timeout（507014）归因顺序（先 kernel 后环境）**：`ACL stream synchronize failed` 是异步 kernel 在
  同步点暴露的真慢/挂起——先查 **M=1 的 cube dot**（L1.12）、再查非仿射 store 逐元素化（L1.1）、再查
  **`kv_len/BLOCK_KV` 串行迭代爆炸**（长 KV + 小 BLOCK_KV，split-KV 只除 NS 倍），全部排除后才考虑环境竞争 /
  参考侧问题；**禁止**判为"瞬态设备竞争"而跳过修 kernel。
