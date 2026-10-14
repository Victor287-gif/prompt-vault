---
type: Guide
title: MultiHeadAttention 算子（mha） 实现手册
description: '典型特征: queryB,S_Q,D / key,valueB,S_K,D，n_heads 为 attr。含家族相邻写法（无投影单头大
  C SDPA，mha-plain-sdpa）的 910b2 复跑实测。'
tags: [task.implementation, topic.attention, topic.compute.composite, topic.tiling, topic.compute.normalization, topic.data_movement.layout_transform]
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
# MultiHeadAttention 算子（mha） 实现手册

> 类别约束见 [attention_constraints.md](../constraints/attention_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。
>
> **家族通用规则引用**：投影段硬约束（M1~M4）、家族通用经验（F-M1~F-M5：权重预转置 /
> fp32 ieee dot 头号杠杆、权重复刻配方、精度契约三档）与测量口径方法论见
> [mha_constraints.md](../constraints/mha_constraints.md) §2.4 / §2.5 / §2.6；§6 形态的从零首版
> 检查表（9 项）见 [mha_constraints.md](../constraints/mha_constraints.md) §6.7。

## §2 MultiHeadAttention 算子（mha）

**算子类别**: `mha`
**典型特征**: `query[B,S_Q,D]` / `key,value[B,S_K,D]`，`n_heads` 为 attr；4 个 `nn.Linear(D,D,bias=False)` 权重由参考实现内部以 `manual_seed(42)` 现场初始化；fp32/fp16/bf16 混合；`head_dim = D/n_heads` 普遍非 2 的幂
**精度基准**: 50/50 pass（fp32 17 / fp16 17 / bf16 16；覆盖 `head_dim ∈ {20,24,32,48,56,60,64,72,78,80}`、`S_Q` 最大 897 / `S_K` 最大 907）。**性能未测** —— 已验证实现为 accuracy-first，fp32 GEMM 走 rank-1 外积，须在 Phase 4 换 tiling

### §2.0 首次生成必读：先对齐 host 侧语义，再写 kernel

该算子最致命的两条缺陷**不在 kernel 里**，而在 host 侧那二十行 PyTorch 胶水代码：权重复现与 `nn.Linear` 转置。二者各自独立致命且互相掩盖 —— 修好任意一条 `passed_cases` 仍为 0，迭代循环拿不到梯度信号。

**因此首次生成必须按此顺序**：① 权重与参考逐位一致 → ② `y = x @ Wᵀ` 语义正确 → ③ 再写 attention kernel。规则见 `ops/triton-op-coding/SKILL.md`「参考实现的 nn.Module 语义复刻」C1/C2；verify 已内置隐藏状态探针，违反 ① 时以 `HiddenStateMismatch` 独立错误类型上报。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 FP32 路径禁用 `tl.dot`

- **禁止** 对 `torch.float32` 输入使用 `tl.dot(q, k_t)` —— Ascend 上会产生 NaN 或错误结果。
- **必须** 回退到逐维度循环（见 A2 路径 (a)）或 rank-1 外积。
- **How to apply:** 见 A2 路径 (a)。

#### L1.2 `tl.dot` 两操作数必须显式 cast 到同一 dtype

- **必须** 两侧都 `.to(a_ptr.dtype.element_ty)` —— `tl.load(..., other=0.0)` 会把结果推成 fp32。
- **Why:** 否则 `semantic.dot` 断言失败：`Both operands must be same dtype. Got fp32 and fp16`。实测该错误导致 33/50 个非 fp32 case 全部编译失败。

#### L1.3 `BLOCK_D` 必须覆盖完整 `head_dim`

- **必须** `BLOCK_D = triton.next_power_of_2(head_dim)`，并对 `d_offs < D` 做 mask。
- **禁止** 硬编码 `BLOCK_D`（如 16），也**禁止**把 `BLOCK_D` 放进 `@triton.autotune` 搜索空间。
- **Why:** `head_dim > BLOCK_D` 时只计算了前 `BLOCK_D` 维，静默错。

#### L1.4 Grid 必须按实际 tile 数裁剪

- **必须** `num_cores_actual = tl.minimum(num_cores, total_tiles)`，其中 `total_tiles = B * H * cdiv(S_Q, BLOCK_SQ)`。
- **Why:** tile 数少于物理核时产生空核调度开销。

#### L1.5 online softmax 必须同步缩放 `acc` 与 `l`

- **必须** 每轮 KV tile：`m_new = max(m, rowmax(scores))`；`delta = exp(m - m_new)`；`l = l * delta + sum(p)`；`acc = acc * delta`；`m = m_new`。
- **禁止** 只更新 `l` 而漏掉 `acc *= delta`（结果偏大）或漏掉 `m = m_new`（下一轮基准错）。

#### L1.6 归约与累加全程 fp32

- **必须** load 后 `.to(tl.float32)`，`scores` / `p` / `acc` / `l` 全部 fp32，最后再 cast 回输出 dtype。
- **Why:** 参考实现普遍写作 `q.float() @ k.float().T` + `F.softmax(...)`，attention 主链本身就是 fp32 语义。

#### L1.7 张量的运行时索引与元素赋值均不支持

- **禁止** `q_tile[:, d]`（`d` 为循环变量）→ `ValueError: unsupported tensor index: int32[]`；**禁止** `acc[:, d] = x`。
- **必须** 需要按维度取向量时直接从 GM 按 `d * stride` load；需要按维度写回时用 `acc += tl.where(d_offs[None, :] == d, x[:, None], 0.0)`。

---

## §2.2 家族相邻写法实测：单头大 C SDPA（CrossformerAttention，`mha-plain-sdpa`）

> 同一家族内**不含投影段**的相邻写法（经 [mha_constraints.md](../constraints/mha_constraints.md) §6
> 与 [attention_writing_classification.md](../constraints/attention_writing_classification.md) 行 6d 路由），
> 二次环境（910b2 / CANN 8.5.1 / NPU=5）完整复跑实测。本算子命中该写法时，首版按
> [mha_constraints.md](../constraints/mha_constraints.md) §6.5 宽容差仲裁落地，Phase 3 一次过 49/49。

### §6.5 落地：宽松容差下 hi/lo + PV1 合规不保守

- 该写法 fp16 标杆 **atol=9e-2**（松约 5 个数量级）⇒ 宽容差标杆：hi/lo 是 p=p1+p2
  **分解**（p1 原生 dtype 精确表示，非 L1.5 禁止的"直接降"），PV1 丢 p2 残差以**全量
  verify 通过**为前提。**禁止**把 PV 操作数 `.to(tl.float32)` 升精度 dot（fp16 cube 4x
  吞吐损失）。保守仲裁会把正确草图判成 A 类——异环境批跑 precheck 完整加载终态知识仍被
  压到 geomean **0.4921**（本卡 1.5114，差 3 倍）。完整判据见
  [mha_constraints.md](../constraints/mha_constraints.md) §6.5。

### 910b2 复跑实证（49/49，geomean 0.4698 → 0.6320 → 探针补丁 1.1495）

- **BQ/BKV 双向牺牲陷阱（硬）**：tile 档位调整**唯一安全方向 = BQ 升且 BKV 不降**
  （persistent task 数减半 + cube M 维微块摊销），且必须 UB 预算公式前置校验通过；
  反向对价（BQ↑BKV↓）与正向对价（BKV↑BQ↓）各实测一轮均劣化（-0.5% / -1.6%）。
  **UB 装不下就不动该桶**——"牺牲一个换另一个"在本结构域无例外。
- **小 D 桶 BQ 翻倍是最大单项收益**：小 C 档 BQ 32→64（BKV 保持、PV1=1 省 p2 buffer
  才装得下）带来桶内 case 近翻倍；plain/fp32 两侧同构成立。
- **simulator 读数陷阱**：910b2 实测 MMAD 16.7%（< 50%，非硬件极限），但两核 SCALAR
  20~27% 全是 kernel 入口参数装载（calls=1 的 LD/ST 固定开销），小 shape 采集下被放大，
  勿误判标量降级（规则 2）——判降级看 calls 是否 ≈ 元素数。
- **IR 层确认无 Triton 源码层剩余空间**：每 KV 迭代 scores/acc 经 GM workspace 往返
  （CC→GM→UB + UB→GM→cbuf）为 MIX 架构固有 staging 税；#19/#21/#7/#10 语义均不适用
  （无 atomic 归约、单 pass、无共享中间量、q 已外提），#29 手写 PIPE_STAGES 超出 Triton
  源码层可表达范围——与 [mha_constraints.md](../constraints/mha_constraints.md) §6.4 的 910_93 诊断同构互证。

### 定向探针补丁（0.6320 → **1.1495**，49/49）

> 对 13 个 D 桶做定向探针（他环境档位作种子 + 折半回退序列，子进程隔离实测编译+跑，
> exp_perf_029 协议），11 桶换档，verify 49/49 一次过。worst case 普遍 2~4.5x；
> **静态估算判死的大 D 桶是全部剩余差距的来源**。

- **★ 静态 UB 估算禁止判死大 D 桶（硬）**：估算公式对大档系统性高估（编译器 buffer
  复用后实测占用远低于 `2*BKV*BC*esz + BQ*BD*4` 字面求和）——实测估算 400KB+ 的 plain
  大档真实编译运行且 3.2x。**"UB 硬墙"结论只能在子进程实测编译探测之后下**；M3 前置
  prune 的 cap 对探针必须放开，否则估算在 host 侧就把种子档杀掉。
- **真墙存在但要实测才配写**：plain@1024 的 BQ≥32 全档 plan-memory 编译失败（910_93
  同档可跑）、plain@512 BQ=64 同——**编译器分配策略是（芯片，CANN）的函数**；910_93 的
  "可编译"结论不能外推，910b2 的"墙"也不能回推。
- **hazard 也是环境函数**：[mha_constraints.md](../constraints/mha_constraints.md) §6.3 的 plain@896
  大档 N<8192 aicore 挂（910_93 B1 N3721 可复现）在 910b2 全档实测**不复现**
  （N=2173/3721 干净）——N 阈值守卫的阈值本身禁止跨环境照抄，换环境需本地实测复验后再
  决定留/删。
- **MERE 擦边机制二次验证**：fp16@768 在 910b2 复现擦边（MERE 9.78e-4 vs 阈 9.77e-4），
  dtype 级例外回退 hi/lo 后 49/49——[mha_constraints.md](../constraints/mha_constraints.md) §6.5 的
  擦边桶处置粒度按设计生效。
- **种子探针 vs 盲扫的效率**：他环境档位作种子时约 2/3 桶种子即最优或一步回退内最优，
  1/3 桶 910b2 最优点偏移（如 896 桶 BKV 256→128 反超）——**种子压缩搜索空间但不替代
  实测**，与 [mha_constraints.md](../constraints/mha_constraints.md) §6.5"档位是环境函数"互证。

---
