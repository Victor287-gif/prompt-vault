---
type: Guide
title: mha 类算子设计约束（一·标准类·含投影段）
description: MHA / SelfAttention / CrossAttention（4 个 nn.Linear 投影 + softmax(QKᵀ)@V）及相邻写法（反传链、无
  head 维单头大 C SDPA）的优化经验合集：判别特征、投影段与 attention 主链的分工、家族通用经验 F-M1~F-M5（§2.4）、补充硬约束
  M1~M4、测量口径方法论（§2.6）与多算子实证。attention 主链约束继承 flash_attention_constraints.md；§6.5 精度仲裁判据（宽容差标杆
  vs FA L1.x 镜像规则的冲突仲裁）+ §6.6 档位推导方法论 + §6.7 从零生成首版检查表。
tags: [task.design, topic.attention, topic.compute.composite, topic.compute.normalization, topic.data_movement.layout_transform, topic.tiling]
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

# mha 类算子设计约束（一·标准类·含投影段）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；
> attention 主链约束（Layer 1、F1~F6、精度闸门、测量口径）继承 [flash_attention_constraints.md](flash_attention_constraints.md)；
> 家族内写法类路由见 [attention_writing_classification.md](attention_writing_classification.md)。

本卡负责 **一 标准类 · 含投影段变体**：参考实现含 **4 个 `nn.Linear(D,D)` 投影**
（Q/K/V 投影 + 输出投影），attention 主链是三段式或分块流式 `softmax(Q@Kᵀ/√d)@V`。
该细分下的算子是 **17 MultiHeadAttention · 18 SelfAttention · 19 CrossAttention**
（self/cross 只是输入来源不同，写法同构）。

- **§1 判别特征与分类依据**（为什么独立成卡，附实测差异）
- **§2 与 `flash_attention` 卡的交叉使用**（投影段硬约束索引，两份 Layer 1 都生效）
- **§2.5 本卡 Layer 1 补充硬约束**（M1~M4：NUM_CORES 动态读取 / grid 恒定 / UB 前置剔除 / dot 链 ≤2）
- **§3 实证：SelfAttention**（13 轮，1.1154 → 1.797 旧口径 / 2.4690 新口径，三 kernel 终态架构）
- **§4 实证：CrossAttention**（18 轮，1.8208 → 2.1687，含 K/V 投影融合）
- **§5 实证：MultiHeadAttention**（17 轮，0.1531 → 2.0439，13.35x，split-K + ieee dot 链路）
- **§6 实证：CrossformerAttention**（无投影单头大 C SDPA，0.515 → 1.5114，PV1 / SPLIT_D / tile 桶治理；§6.5 与 FA 卡精度规则的冲突仲裁——宽容差标杆判据 + SPLIT_D / MERE 处置粒度；§6.6 档位推导方法论——五元组 tile 表 + 种子/折半回退/子进程实测裁决，逐桶数值不归档）
- **§7 实证：ScaledDotProductAttentionBackward**（反传链 dQ/dK/dV，0.6351 → 1.2969，S_K 二分混合分派 + fp32 ieee 红线）

> §6/§7 是经 [attention_writing_classification.md](attention_writing_classification.md) 行 6c/6d 路由进来的
> **不含投影段**相邻写法（反传链 / 无 head 维单头大 C SDPA）。同卡不同章：命中 6c/6d 的算子只取所指章节（§6 或 §7）
> ＋ [flash_attention_constraints.md](flash_attention_constraints.md) 的 Layer 1，§3-§5 的投影段约束对它们不适用。

> **证据基础**：三个算子各 50 case 的完整优化轨迹（fp32/fp16/bf16 混合，
> `head_dim ∈ {20..80}` 普遍非 2 幂，S ≤ 897），910_93 / CANN 9.1.0 / triton-ascend 3.2.2。三者精度均 **50/50**。
>
> **Layer 映射（Phase 2 读卡指引，对齐 template 四层模型）**：本卡按写法分章的实证体例，
> Layer 语义如下——**Layer 1（设计约束，negative_prompt 必守）**：§2.5 M1~M4、
> §3.2 冲突判据、§6.5 仲裁规则、§6.7 检查表、§7.2 精度红线；
> **Layer 2（算法骨架，参考方向）**：各章「终态架构」小节（§3.1 / §4.1 / §5.1 / §6.1 / §7.1）；
> **Layer 3（关键代码，可替代）**：§2.5 M1 的 NUM_CORES 读取代码、§6.1 终态 kernel 骨架。
> 生成器取用：Layer 1 逐条必守；Layer 2 仅作参考方向，输出必须全新草图；Layer 3 技巧禁直接复制。

---

## §1 判别特征与分类依据

### §1.1 判别特征（决定用不用本文件）

打开参考实现，**同时满足**：

1. 含 **4 个 `nn.Linear(D,D)` 投影**（Q/K/V 各一 + 输出投影；self 形态 `q=k=v=x`，
   cross 形态 `query` 与 `context` 分开输入、`key=value=context`）；
2. attention 主链是 `softmax(Q@Kᵀ/√d)@V`（三段式或 KV 分块流式均可）；
3. 投影权重常由参考实现内部 `manual_seed` 现场初始化，`head_dim = D/n_heads` 普遍非 2 的幂。

### §1.2 为什么不能落到已有行（实测差异）

- **与一·1 基础三段式 / 一·2 分块流式（`flash_attention` 卡）的差异**：
  本细分多了**投影段**这个独立优化对象，且实测**最大的一笔收益在投影侧，不在 attention 侧**——
  CrossAttention 权重预转置单笔 -15%（bf16 +25%），GQA 含投影形态投影侧 +57%（见 [gqa_constraints.md](gqa_constraints.md) L1.6）。
  FA 卡的注意力侧经验（online softmax、BLOCK 面积预算）只覆盖一半目标函数。
- **误分类代价有实测记录**（§4.5）：CrossAttention 曾被归入 `sdpa` 行，
  投影权重以 `[out,in]` 原始布局进 GEMM，Phase 4 花 17 轮才 rediscover 预转置。
  **不命中本卡 = 丢掉最大的一笔收益。**

## §2 与 `flash_attention` 卡的交叉使用 ＋ 家族通用经验（⚠️ 交叉行，两份 Layer 1 都生效）

本细分是**投影段（本卡）＋ attention 主链（FA 卡）**两段结构，优化对象互相独立
（profiling 里分别看 Duration 占比，主攻目标每轮会换，实测从 attention 64%/linear 36%
变到 46%/54%）。两份 Layer 1 必须**同时生效**：

| 段 | 必读条目（`flash_attention_constraints.md`） |
|---|---|
| 投影段 | **L1.11**（host 侧先对齐 nn.Linear 语义，权重错与转置错互相掩盖）、**L1.12**（投影权重禁止 `[out,in]` 原始布局喂 GEMM，★★★ 头号杠杆）、§3.4（两段结构）、§4.4（权重按 tile 预重排）、§4.5（投影分块 dtype 分档 + 并行度保护） |
| attention 主链 | L1.1（fp32 ieee dot）/ L1.3（online softmax 同步缩放）/ L1.4（精度契约）/ L1.13（BLOCK 面积预算）/ §4.6（half 路径 PV hi/lo 双 dot） |

另需摘录本卡 **§2.4 家族通用经验（F-M1~F-M5 / 家族优化点排序 / 家族证伪全表）** 与
**§2.5 的 M1~M4 四条补充硬约束**（原拟写入 FA 卡，经 review 权衡收敛到本卡，仅对本细分生效）；
**§2.6 测量口径为家族通用（§3-§7 全部适用）**——§6/§7 不含投影段的算子同样摘录。

⚠️ **不要**把 4 次投影和 1 次 attention 合并进同一个 `@triton.jit`（`MODE: tl.constexpr` 分支）——
见 [flash_attention_constraints.md](flash_attention_constraints.md) 测量口径（同名 kernel 合并会利用口径缺陷）。

### §2.4 ★ MHA 家族通用经验（跨 §3-§7 收敛；体例对应 FA 卡 §1 F1-F6 / 证伪全表）

> 各章（§3-§7）实证中反复出现、跨算子一致的家族级结论收敛到此；逐 case 数值证据留在各章。
> 命中 6a（含投影段）算子 precheck **全摘录**本节；§6/§7 不摘投影段专属条（F-M1/F-M2）。

**F-M1 ★ 头号杠杆两笔，首次生成必须做对（不是 Phase 4 探索项）**

- **权重预转置 [out,in]→[in,out]**（= FA 卡 L1.12）：投影段主收益 −15%（bf16 +25% / fp16 +18% / fp32 +5.5%）；误落 sdpa 行丢此杠杆 = 17 轮 rediscover（§4.5）
- **fp32 ieee dot**（= FA 卡 L1.1）：attn 每 k-iter 6 dot→2 dot（linear 微基准 3.8x），链路 +13.1%（1.6289→1.8419）；§7.2 反传唯一合规路径
- 各章排序差异只反映发现轮次；家族口径 = **投影段做预转置、attention 段做 ieee dot，都首版做对**

**F-M2 ★ 权重复刻配方（§3.1/§4.1 两处统一）**

- rng save/restore + `manual_seed(42)` + **CPU fp32 采样** ×4 顺序 q,k,v,o，采样完再 `.to(device,dtype)`（= FA 卡 L1.11）
- bound：参考用 `kaiming_uniform_(a)` ⇒ `bound = √3·√(2/(1+a²))/√D`；**a=√5（nn.Linear 默认）时 = 1/√D**
- 参考不现场建层则跳过；AST 禁 nn.Linear 时数值等价复刻

**F-M3 精度契约三档**（各章终态架构 §3.1/§5.1/§6.1/§7.2）

- **fp32**：全链 ieee dot，scores/p/acc/l 全 fp32，末尾单次 cast（单舍入点）
- **fp16/bf16**：p 拆 p1（原生 dtype 精确表示）+ hi/lo 双 dot 补偿（PV 单 dot 精度证伪）；宽容差标杆可丢 p2 残差（PV1，§6.5）
- **反传**：fp16 cube 内部累加 dtype 固定，唯一合规 = 全 fp32 ieee（§7.2）

**F-M4 scale 保留 tile 内乘，q 预缩放证伪**

**F-M5 布局级改动必重扫 tile**（§4.2 二次优化：预转置解锁 BK=128 档 +2.7%；§6.6 档位推导 step 5 同规）

**家族优化点排序**（§3-§7 各章收益归因的家族行收敛到此；逐算子特有项留在各章）

| 排序 | 家族级收益 | 实测 |
|------|-----------|------|
| 1 | = **F-M1**（预转置 + ieee dot） | 预转置 −15% / ieee +13.1% |
| 2 | BLOCK autotune + early_config_prune UB 剪枝 | +11.0% |
| 3 | 形态融合：MERGE_QKV（self）/ K-V 融合（cross）/ split-K+LSE（MHA） | 逐比特一致 / ≈0.96（与预转置合力 −15%）/ +0.4% |
| 4 | 微优化：ceil16 / propagate_nan / 指令重排 / 配置扩展 / 循环外提 | 各章 ≈0 或拒绝 |

**家族证伪全表**（跨章重复死路，不要重跑）

| 死路 | 家族实测（跨章一致） |
|------|---------|
| multibuffer=True | UB 紧张时全 shape 劣化 ~−10%；Crossformer 8 桶零差异——白试 |
| load-reorder（v 提前发射） | 小 shape 无净收益、大 D 桶 **−30~40%**（破坏 QK 操作数 staging） |
| ceil16 单独提交 | 净≈0（1.011）——首次直接用，别指望单独收益 |
| propagate_nan=ALL 全局 | 会话间翻号（投影主导低优先级）；单块 case 退化 25~45%；FA 卡 +2.7% 仅见于 attention 主导算子 |
| autotune 配置扩展 | 大 shape 增益被小 case do_bench 噪声对冲，tuner 误选，全量打平 |
| p 二次 where 删除 | 场景条件性：fp32 下溢逐比特等价 vs bf16 小 shape MERE 违规 → 默认保留归零 |

**结构域上限（家族判据）**：前向 fa-mha 1.5x 天花板（§6.4：Cube↔Vector 跨核同步 + 微块 staging 税）；反传 hybrid ~1.3x（§7.2：fp32 ieee 红线）

### §2.5 本卡 Layer 1 补充硬约束（M1~M4，与 FA 卡 Layer 1 一同生效）

> 这四条在 MHA 家族任务中发现，原拟写入 `flash_attention` 卡作为通用约束；
> 经 MR review 权衡**收敛到本卡**——避免对纯 FA 算子引入未在该路径验证过的硬边界。

- **M1 ★ `NUM_CORES` 禁止硬编码 24，模块导入时按设备动态读取一次**
  ```python
  # host（模块导入时动态读取一次，进程内仍是常量）
  try:
      import triton.runtime.driver as driver
      _dev = torch.npu.current_device()
      NUM_CORES = driver.active.utils.get_device_properties(_dev)["num_aicore"]  # 910_9372 实测 20
  except Exception:
      NUM_CORES = 20
  ```
  `NUM_CORES` 必须保持**进程内常量**——做成随 case/随 launch 变化的值会触发编译爆炸
  （实测 verify 超时、0 个用例启动）。硬编码 24（DAV 口径）在 910_9372（实际 20 核）的
  persistent 步长循环下失配 4/24，即尾部排队 + 负载不均（17_MHA 实测修正后 **+11.6%**；
  反传链 §7 第二实证 **+17.4%**——M1 不专属投影段，任何 persistent/grid 恒定形态适用）。
  `grid` 是启动参数，随 case 变化无妨。

- **M2 ★ BLOCK 由 tuner 注入时，`grid=(NUM_CORES,)` 恒定 + 核内 grid-stride**
  host 在 launch 时**无法预知 tuner 选中 config 的 tile 总数**，按旧 config 的 tile 数裁剪
  grid 会在 tuner 换 config 后产生 stride 空洞、**静默漏 tile**。
  正解：`grid = (NUM_CORES,)` 恒定，kernel 内 `for t in range(pid, TOTAL, NUM_CORES)`。

- **M3 ★ UB 溢出 config 必须在 `early_config_prune` 里按峰值估算前置剔除**
  `_bench` 只捕获 OutOfResources 等三类异常，UB 溢出的通用异常会**炸掉整个 autotune 进程**。
  峰值估算公式见 §3.1 K3 / §4.1 K1（CAP 168~200KB 按算子实测标定）。

- **M4 ★ 单 kernel 循环体内同 dtype cube `tl.dot` 链 ≤ 2 个（aicore 507014）**
  禁止同一循环体放 3 个 `tl.dot`（如 fp32 的 hi@hi+hi@lo+lo@hi 三 dot 补偿）——M>256 时
  aicore timeout（error 507014）必挂；join/reshape 堆叠成单大 dot 或 K 维堆叠 2 dots 会让
  bishengir-compile 直接崩溃。ieee 路径 QK 1 dot + PV 1 dot、half PV hi/lo（FA 卡 §4.6）
  均为 2 dot/iter，天然合规。

### §2.6 ★ 测量口径与判定方法论（MHA 家族通用，跨 §3-§7）

- **per-case geomean 唯一可信**：单 case 噪声 ±30%（同代码单跑波动 -31%）、跨会话漂移 ±2% ⇒ 单 case / 跨会话对比均不可信
- **判定必须同会话紧邻 A/B**：base→opt→opt→base，repeats≥30，读 `per_shape_results[].implementation.avg_latency_ms` 算 geomean(opt/base)；**ratio < 0.95 才算真加速，0.95~1.05 视为噪声**（run 间差 <5% 即噪声，环境地板 4.6%），不得写入报告当"优化成功"
- **`speedup_vs_torch` 只作趋势**：NPU 繁忙时 framework 膨胀幅度 > impl，speedup 被动抬高（同代码两次 benchmark 1.79~2.22 波动）
- **采样窗口必须足够长**（双 stream 短跑 +22~25% 虚高）；**微优化效应 <1%**（分档/ceil16/指令重排级）低于噪声地板，仅在「IR 级证据 + profiling 关键路径」时才值得跑
- **simulator 看不到 host 损耗**：autotune 的 256MB L2 清洗 buffer churn + 编译风暴在 msprof 采集里不可见——autotune 不可用时改为**离线调优决策梯固化**（按 D/BH 查表）
- **报数纪律**：report/summary 每个数字注明口径（紧邻 A/B impl_ms 还是单跑 speedup），宁可报真实 1.8x 也不报虚高 2.2x

---

## §3 实证：SelfAttention（`mha-self`）（13 轮，1.1154 → 1.797 旧口径 / 2.4690 新口径）

> 形态：`q=k=v=x`（self-attention），4 个 `nn.Linear(D,D,bias=False)` 投影
> （`manual_seed(42)` 现场初始化），`softmax(Q@Kᵀ/√d)@V`；50 case = fp32 17 / fp16 17 / bf16 16，
> `head_dim ∈ {20..80}` 普遍非 2 幂，S ≤ 897。架构 910_93 / CANN 9.1.0 / triton-ascend 3.2.2。
>
> ⚠️ **诚实口径**：最优为迭代 9（删 K3 冗余 p 二次 where，fp32 下溢为精确 0、逐比特等价），
> 双跑 per-case 均值 **1.797**（run1=1.8275 / run2=1.7616）。config target 0.8 达成，目标 2.0 未达：
> 分 dtype fp32 1.49 / fp16 1.91 / bf16 2.00（仅 bf16 破 2.0）。标题中的 2.4690 是 benchmark 口径修正
> （多 launch 少计，新仓 6e4dc13c）后**同一份代码**的重测值——纯测量修正，跨版本数字不可比（§3.2-④）。
> **剩余瓶颈**：worst6 中 5/6 为 K1 投影 GEMM 主导（66-77%），fp32 大 shape K1 40.6 TFLOPS
> vs torch 57 TFLOPS（Triton cube 调度差距）；msprof 采集证伪了"cube 物理极限"结论
> （K1 load-bound MTE2 33%、K3 vector-bound 44-60%，MMAD 均未进前三）。

### §3.1 终态架构（三 kernel 结构，逐项都有实测依据）

```
K1  _linear_kernel（投影 GEMM，每 forward 发 2 次）
  · MERGE_QKV：q=k=v=x ⇒ W_qkv = cat([Wq,Wk,Wv], 0)，单次 GEMM 与三次独立投影【逐比特一致】
  · W host 侧预转置 [K,N]（= FA 卡 L1.12，IR 实证消除 k-loop 内 tl.trans→vtranspose）
  · fp32 走 input_precision="ieee"（= FA 卡 L1.1）；fp16/bf16 原生 dtype 单 dot
  · BLOCK_M/N/K ∈ {64,128}×{64,128}×{64,128} 的 6 配置 autotune（key=['M','N','K']）
  · Cube 类 kernel 必须手写 Config（hints/自动模式仅支持 Vector）
  · grid=(NUM_CORES,) 恒定 + 核内 grid-stride（BLOCK 是 tuner 注入的，见 §2.5 M1/M2）

K3  _attn_kernel（online softmax）
  · QKV 物理布局 [B*S, 3D]：q 列 h*HD、k 列 D+h*HD、v 列 2D+h*HD —— strided 逻辑视图免 transpose（= FA 卡 L1.6）
  · fp32 路径：q/k/v 直载，QKᵀ 与 P@V 各 1 个 ieee dot（每 k-iter 6 dot → 2 dot）
  · half 路径：p kernel 内 hi/lo 分裂 2-dot 补偿（= FA 卡 §4.6；p 直接降 dtype 的误差 5e-4/4e-3 顶穿阈值）
  · BLOCK_SQ/BLOCK_SK ∈ {32,64,128}² autotune + early_config_prune UB 剪枝：
      ub = BSQ*BD*esz + 2*BSK*BD*esz + BSQ*BD*4 + 2*BSQ*BSK*4 [+ half 路径 2*BSQ*BSK*esz]
      CAP = 168KB（实测 (64,64,128) fp32 可编译 = 160KB，留 8KB 余量）
    ⚠️ UB 溢出的通用异常不在 _bench 捕获的三类里，会炸掉整个 autotune 进程 ⇒ 必须前置剔除
  · scale 保留 tile 内乘（q 预缩放已证伪，见 FA 卡证伪表）；末尾 out = acc * (1.0/l) 广播乘

host
  · 权重复刻：rng save/restore + manual_seed(42) + CPU fp32 uniform_(-1/√D) ×4 顺序 q,k,v,o
    （= FA 卡 L1.11；AST 禁 nn.Linear 时用数值等价初始化复刻）
  · 缓存 key=(D, H, device, dtype)，权重只建一次
```

### §3.2 与 FA 卡既有条目的冲突判据（4 条，搬运结论前先核对）

| # | 冲突 | SelfAttention 实测 | **判据** |
|---|------|--------------------|----------|
| ① | F3 `ceil16` | 打平（-0.15%）拒绝 | `ceil16` 省的是 K3 tile 面积；**K1 GEMM 主导的算子无收益**。先看 per-kernel 时延占比再决定投入 |
| ② | L1.14 `propagate_nan` 分档 | 无真实效果拒绝（会话间翻号） | `m_new` 更新占 K3 比例小 / K1 主导时低优先级；FA 卡的 +2.7% 来自 attention 主导的算子 |
| ③ | §4.1 末尾 `acc/l` 倒数乘 | 采纳（`1.0/l` 标量除 + 广播乘） | 与「循环外无收益」的失效边界记录冲突，case 相关；成本极低可顺手做，判定按打平处理 |
| ④ | benchmark 口径 | 新仓 6e4dc13c 已修正多 launch 少计 | 同一份代码旧口径 1.7970 → 新口径 2.4690，**纯测量修正**；跨版本数字不可比，报告必须注明 commit |

### §3.3 判定方法论（噪声地板，比优化本身更早决定成败）

> 家族通用的测量口径与紧邻 A/B 判定见 **§2.6**；以下为本算子早期实测依据。

- **单 case 噪声带 ±30%**：HD=64、代码路径逐比特相同的 case 单跑波动 -31% ⇒ **单 case 对比不可信，
  per-case 均值 geomean 是唯一稳定口径**（双跑取每 case 均值再求几何平均）。
- **跨会话系统漂移 ±2%**：逐比特相同代码的 27 个 case，跨会话对比均值 -2.40%，同会话背靠背对拍翻为 +1.27%。
  ⇒ **<±2% 的 delta 跨会话对比不可信；关键判定必须同会话背靠背（base→cand→base）对拍**。
- **微优化真实效应 <1%**（分档 maximum、ceil16、指令重排级别）低于噪声地板 ⇒ 这类改动只在
  「有 IR 级证据（指令条数变化）+ profiling 证明落在关键路径」时才值得一跑。
- 配置扩展的教训：**大 shape 微基准增益不能线性外推全量 geomean**——tuner 在小 case
  上被 do_bench 噪声误选的概率随配置空间增大而上升。

### §3.4 收益归因（本算子 13 轮的排序，指导同类算子的投入顺序）

| 排序 | 改动 | 收益 |
|------|------|------|
| 1 | fp32 ieee dot 替换 hi/lo 分裂链（微基准 3.8x，17_MHA 链路 +13.1%） | 数量级 |
| 2 | W 预转置 [K,N] + MERGE_QKV 单 GEMM（L1.12 + 融合） | 大 |
| 3 | BLOCK autotune + UB prune（L1.13 思路）+ grid-stride 解耦 | 中 |
| 4 | 冗余边界运算消除（p 的二次 where、q 预缩放证伪后的保守保留） | 小但零风险 |
| 5 | ceil16 / propagate_nan / 指令重排 / 配置扩展 | 0（全部拒绝） |

### §3.5 ⛔ 证伪方向（不要重跑）

| 死路 | 实测 |
|------|------|
| autotune 配置扩展（补 BLOCK_K=128 / 大组合） | **拒绝**：大 shape 微基准增益（-15%/-26% 时延）被小 shape 的 do_bench 噪声**误选**对冲，全量 geomean 打平偏差。配置空间一大，tuner 在小 case 上选错的概率也随之上升 |
| k/v load 提前到 QKᵀ dot 之前发射 | **拒绝**：提前量与 scores 计算在编译器调度下对冲，geomean 无净收益 |
| tile 按 M 分组重排发射顺序（grouped ordering） | **拒绝**：大 shape +8~23% 与小 shape -9~18% **极化对冲**；受益/回退的 M 区间重叠，无干净门控分界面 |
| `ceil16(HD)` 替代 `next_pow2(HD)` | **打平拒绝**（-0.15%）：HD∈{72,78,80} 的 case 瓶颈在 K1 GEMM，K3 的 BLOCK_D 面积收益对冲。见 §3.2-① |
| `propagate_nan=ALL` 分档 | **无真实效果拒绝**：两次独立会话对拍 gated 组 +1.63% → -4.12% 翻号，属跨会话噪声。见 §3.2-② |

---

## §4 实证：CrossAttention（`mha-cross`）（18 轮，1.8208 → 2.1687）

> 形态：`query [B,S_Q,D]` 与 `context [B,S_K,D]` 分开输入，`key=value=context`，
> 4 个 `nn.Linear(D,D,bias=False)` 投影（`manual_seed(42)` 现场初始化），
> `softmax(Q@Kᵀ/√d)@V`；50 case = bf16 17 / fp16 16 / fp32 17，`head_dim ∈ {20..80}`
> 普遍非 2 幂，S_Q ≤ 897。架构 910_93 / NPU=6 / CANN 9.1.0 / triton-ascend 3.2.2。
>
> ✅ **复用验证（0815 重生成）**：凭本卡经验重跑同一算子，首版即 **2.4123**
> （权重预转置 / ceil16 / fp32 ieee dot 首轮全部生效），仅一轮 K/V 融合微调
> （impl_ms -4.0%）收敛至 **2.4391**——本卡"首次生成直接命中"的目标已被实测兑现。

### §4.1 ★★★ 终态架构（三 kernel，首次生成直接采用）

```
K1  _ca_linear_kernel（通用投影 GEMM，每 forward 发 2 次：Q 投影 + O 投影）
  · W host 预转置 [out,in]→[in,out]（= FA 卡 L1.12，头号杠杆，首次生成必须做对）
      kernel 内 w_off = offs_k[:,None]*D + offs_n[None,:]（BN 方向 stride=1）
  · fp32 走 input_precision="ieee"（= FA 卡 L1.1）；fp16/bf16 原生 dtype 单 dot
  · BM/BK/BN ∈ {64,128}×{32,64}×{64,128} 7 配置 autotune + early_config_prune
      ub = 2*(BM*BK + BK*BN)*esz + BM*BN*4 < 200KB
  · grid = min(num_cores, tiles) + 核内 grid-stride（= §2.5 M1/M2）

K2  _ca_kv_linear_kernel（★ K/V 投影融合，1 次启动替代 2 次）
  · key=value=context ⇒ K=ctx@Wk 与 V=ctx@Wv 共享同一输入张量
    ⇒ 单 kernel 内双 dot 共享 x load，省 1 次 context memory pass + 1 次 launch
  · 循环体内：load x 一次 → wk/wv 两次 load → acc_k/acc_v 两个 fp32 累加器双 dot
  · UB 占用近乎翻倍（x + 2×w + 2×acc）⇒ autotune 配置数裁到 4，防 verify 超时
      ub = (BM*BK + 2*BK*BN)*esz + 2*BM*BN*4 < 200KB
  · 单独实测紧邻 A/B impl ratio ≈ 0.96（真实 ~4%），与权重预转置叠加后合力 -15%

K3  _ca_attn_kernel（online softmax，S_Q ≠ S_K）
  · K 直接 load 成 [BLOCK_D, BSK] 转置态，省 tl.trans（FA 2.12，load 路径收益）：
      kt_col_base = b*S_K*D + h*HD + offs_d[:,None]
      kt_off = kt_col_base + offs_k[None,:]*D      # d 方向连续
  · fp32：QKᵀ 与 P@V 各 1 个 ieee dot；fp16/bf16：QKᵀ 原生 dot，PV hi/lo 双 dot（= FA 卡 §4.6）
  · BLOCK_D = ceil16(HD)；BSQ/BSK ∈ {32,64,128}² autotune + UB prune（CAP 192KB）
  · p 掩码必须显式归零 p = tl.where(sk_valid, p, 0.0)
    （A1：Ascend 硬件 exp(-3e38) 残留 ~1e-16 非零，删二次 where 在 bf16 小 shape 上 MERE 违规）
  · scale 保留 tile 内乘 scores*scale_inv（q 预缩放已证伪，FA 卡证伪表）

host
  · 权重复刻：rng save/restore + manual_seed(42) + CPU fp32 uniform_(-bound,bound) ×4
    顺序 q,k,v,o；bound = √3·√(2/(1+a²))/√D（Kaiming linear 公式逐字复刻 = FA 卡 L1.11）
  · ★ 预转置放 host：w.t().contiguous() 不消耗 RNG，数值逐位不变，参考实现不受影响
  · 缓存 key=(D,H,device,dtype)，权重只建一次；cache.clear() 防多 dtype 累积
```

### §4.2 收益归因（18 轮排序，全部经紧邻 A/B impl_ms 判定）

| 排序 | 改动 | 紧邻 A/B impl_ms ratio | 结论 |
|------|------|----------------------|------|
| 1 | 权重预转置 [out,in]→[in,out]（L1.12，4 个投影全改） | 0.85（真实 -15%） | **主收益**，分 dtype bf16 +25%/fp16 +18%/fp32 +5.5% |
| 2 | K/V 投影融合（K2 kernel） | ≈0.96 | 真实小幅，与 1 叠加 |
| 3 | K 转置直读（FA 2.12） | 组合内难剥离 | 文档值 +0.7%，零风险顺手做 |
| 4 | 循环不变量外提 / V load 延迟 / ceil16 / qd 外提 | 1.00~1.01 | 全部噪声内，可做可不做 |

终态：geomean 1.8208 → 2.1125（+16.0% impl_ms），50/50 verify，bf16 2.3755 / fp16 2.2306 / fp32 1.7849。

**二次优化（0812 复盘补录）**：权重预转置（L1.12）落地后 BK 方向变连续，
**旧 tile 查表随之过期**——追加 BK=128 三配置（投影 GEMM）与 attention 128×128 档后
2.1125 → **2.1687**（+2.7%，impl avg 0.0877 → 0.085 ms）。教训：**布局改动会解锁新的
tile 档位，改完布局不重扫 tile = 白丢一档**；任何布局级改动（转置/融合/重排）落地后，
固化 tile 表必须重扫一次。

### §4.3 ⛔ 证伪方向（不要重跑）

| 死路 | 实测 |
|------|------|
| `multibuffer=True` | UB 紧张时**全 shape 劣化**（约 -10%），与 FA 卡证伪表记录一致 |
| K/V host 侧 `torch.cat` 后单 kernel | 每 call 拼接开销吃掉融合收益——**融合必须在 kernel 内做，不在 host 做** |
| ceil16 替代 next_pow2（单独提交） | HD∈{72,78,80} 个别 case +4~10%，HD≤32 回退，geomean 净 ≈ 0。首次生成直接用 ceil16 即可，但**不要指望它单独带来收益** |
| 删除 p 的二次 `tl.where` 归零 | cross_attention bf16 小 shape MERE 违规（49/50）；SelfAttention 上却 50/50——**场景条件性，默认保留归零** |
| qd（同 dtype cast）循环不变量外提 | no-op cast，0 收益 |

### §4.4 ★ 测量口径（与 FA 卡 §7 互补：`speedup_vs_torch` 会被框架噪声污染）

- **污染机理**：speedup = framework_ms / impl_ms。NPU 繁忙时 framework（torch 参考）膨胀
  幅度 > impl 膨胀幅度，speedup 被动抬高。**同一份代码两次 benchmark 可在 1.79~2.22 之间波动**。
- **正确判定**：紧邻 A/B —— baseline 与 optimized 在同一脚本内背靠背（base→opt→opt→base），
  repeats≥30，读 `per_shape_results[].implementation.avg_latency_ms` 算 geomean(opt/base)。
  ratio < 0.95 才算真加速；0.95~1.05 视为噪声，不得写入报告当"优化成功"。
- **报数纪律**：report/summary 中每个数字必须注明口径（紧邻 A/B impl_ms 还是单跑 speedup_vs_torch）。
  宁可报真实的 1.8x 也不报虚高的 2.2x。

### §4.5 分类教训（为什么会多走 10+ 轮）

本次曾把 CrossAttention 归入 `attention`（旧分类表把 cross-attention 列在 `sdpa` 行），
导致 L1.12 / §4.6 / ceil16 这些 FA 卡的硬规则**首次生成时没有生效**，
权重以 `[out,in]` 原始布局进了 GEMM，Phase 4 花了 17 轮才从零 rediscover。
**判别口诀：只要有"4 个 Linear 投影 + softmax(QKᵀ)@V"结构，无论 self/cross/MHA，
一律走本卡 + [flash_attention_constraints.md](flash_attention_constraints.md) 交叉行**
（[attention_writing_classification.md](attention_writing_classification.md) 判别表 6a 行）。

---

## §5 实证：MultiHeadAttention（`mha`）（17 轮，0.1531 → 2.0439）

> 形态：`query [B,S_Q,D]` / `key,value [B,S_K,D]` 分开输入，`n_heads` 为 attr，
> 4 个 `nn.Linear(D,D,bias=False)` 投影（`manual_seed(42)` 现场初始化）；50 case =
> fp32 17 / fp16 17 / bf16 16，`head_dim ∈ {20..80}` 非 2 幂，`S_Q ≤ 897 / S_K ≤ 907`。
> 架构 910_93 / CANN 9.1.0。终态 **2.0439**（三次运行 2.0227/2.0465/2.0439 取中位数），
> 50/50，分 dtype fp32 1.7953 / fp16 2.1285 / bf16 2.2469，达 target 2.0。
> 相比基线 **13.35x**——三个算子里基线最低、提升空间最大的一例。

### §5.1 终态架构（四 kernel，比 §3/§4 多 split-K 一路）

```
K1  _mha_linear_kernel（投影 GEMM ×4，autotune BLOCK_M/N/K + UB prune）
K2  _mha_attn_kernel（online softmax 主路，autotune BLOCK_SQ/SK + UB prune）
K3  _mha_attn_partial_kernel（★ split-K 部分态：小 grid 欠载时 S_K 分裂）
K4  _mha_attn_combine_kernel（LSE 合并 partial 结果）
  · split-K 门控：tiles*2 ≤ cores 且 nkb ≥ 4 才启用
  · 其余同 §4.1：W 预转置 / fp32 ieee dot / half PV hi/lo / ceil16(HD) / UB prune autotune
```

### §5.2 收益归因（17 轮排序，chip 8 口径）

| 排序 | 改动 | 轮次 / 效果 |
|------|------|------------|
| 1 | **fp32 原生 ieee cube dot 全链路**（`input_precision="ieee"`，推翻"fp32 禁 tl.dot"） | **+13.1%**（1.6289→1.8419）；attn 每 k-iter 6 dot→2 dot，linear 微基准 3.8x，→ FA 卡 L1.1 |
| 2 | linear+attn **BLOCK autotune + early_config_prune UB 剪枝**（tile 计数下沉 kernel） | **+11.0%**（→2.0439 达标） |
| 3 | pass 合并 / hi-lo 分裂 dot / 生产者预分裂等批量改造 | 0.1531→1.6106 |
| 4 | attn **split-K + LSE combine**（小 grid 欠载）+ 门控收紧 | +0.4% |
| 5 | IR 驱动 vdiv→倒数乘法（scores/epilogue） | +0.3% |

### §5.3 ⛔ 证伪方向（不要重跑）

| 死路 | 实测 |
|------|------|
| q/k/v 多指针融合成单次启动 | **-43%**（0.9086），坚决不做 |
| half 路径 PV 单 dot（省 hi/lo） | 精度证伪——PV 必须 hi/lo 双 dot（FA 卡 §4.6 同结论） |
| 独立预分裂 pass / k-loop 地址外提 | 打平，拒绝 |

### §5.4 与 §3/§4 的分工

- 17_MHA 是 **ieee dot（FA 卡 L1.1）与 dot 链 ≤2（本卡 §2.5 M4）两条硬约束的发现算子**；
- §3（self）特有 MERGE_QKV 单 GEMM（q=k=v=x 逐比特一致）；§4（cross）特有 K/V 投影融合
  （key=value=context 共享 x load）；§5（MHA）特有 split-K + LSE combine（S_Q≠S_K 且
  grid 欠载时的兜底）。三者投影段其余部分同构，按形态各取所需即可。

---

## §6 实证：CrossformerAttention（`mha-plain-sdpa`）（14 轮，0.515 → 1.5114）

> 形态：**无投影段、无 head 维**的单头 SDPA——`out = softmax((q·C^-0.5) @ kᵀ) @ v`，
> q/k/v 缺省取 x，shape `[B,N,C]`，**C = 64~1024**（超出 FA 卡 head_dim ≤ 256 经验域）。
> 49 case fp16/bf16/fp32，910_93 / NPU=6。经 [attention_writing_classification.md](attention_writing_classification.md) 行 6d 路由到本卡，
> attention 主链 Layer 1 仍由 [flash_attention_constraints.md](flash_attention_constraints.md) 提供。
> 终态 **1.5114**（基线 0.515 → +193%，speedup_vs_baseline 2.9348），49/49 verify；
> target 2.0 结构性不可达（§6.4）。
>
> **host 语义归一（精度全过的地基，从零首版必守）**：
> - `get_input_groups()` 只喂 `[x, group_size, q, k, v]`；`num_heads / mask / scale /
>   pos_bias / feature_shape` 均**不传 forward**；`feature_shape` 恒 None ⇒ 单组模式，
>   `group_size` 解包后弃用。
> - 实际计算 = **单头注意力（H=1, head_dim=C）**：`out = softmax((q·C^-0.5) @ kᵀ) @ v`，
>   **q/k/v 为 None 时取 x**；输入输出 3D `[B,N,C]`，dtype E ∈ {fp16, bf16, fp32}。
> - `scale_eff = scale if scale is not None else Dh ** -0.5`（49 case 恒后者）；
>   `B==0 或 N==0 ⇒ torch.empty_like(x)`（防御保留）。
> - ⛔ **x 全 NaN（末条 case）⇒ online softmax NaN 天然传播，禁止 clamp 掩盖**——
>   softmax m/l/p 全 fp32 + `propagate_nan` 分档，NaN 逐位穿透到输出。
> - 极小 case S=1/D=8 ⇒ tile 收缩（小 N clamp 到 `max(16, ceil16(N))`）。

### §6.1 终态架构要点

- **persistent task 循环**：grid = NUM_CORES（实测 20），核内 grid-stride 领 tile（= §2.5 M1/M2）
- **fp32 走 ieee + SPLIT_D；fp16/bf16 走 plain hi/lo 双路径**
- **PV1（p1-only PV）★ 头号杠杆**：plain PV 只保留 p1 段单 dot（丢 p2 补偿段），
  省下的 UB 让 SPLIT_D 能开更大 BQ——大 D 档解锁 **-50%+** 级收益；
  代价是小 D（96/128）桶回退 6~15%，**必须按桶分派**（per-bucket 开关，不是全局）
- **plain SPLIT_D**：两个半 D 的 p1 dot **并行**替代整 D dot（同一 p1 对 v 前后两半
  各做一次 dot，同级累加；⛔ 链式 = 前半 dot 输出作后半操作数，数学错误，反例见 §6.5），
  突破 BQ=16 微块 staging 税（BQ=32/64 随 D 档启用；D≥768 的 BQ=32 撞 UB 硬墙编译失败，禁用）
- **单舍入点**：fp32 累加全程，末尾单次 cast（对齐标杆链口径）
- **tile 双表（五元组结构）**：档位 = 五元组 `(BQ,BKV,BC,PV1,SD)`，主表 keyed
  `(path, BLOCK_D)`，EX 表 dtype 级覆盖（fp16@768 出现 MERE 擦边 case，需单独压
  tile 档）。`BC` 是 K/V 的 D 维**分块加载粒度**（UB 按 `2*BKV*BC*esz` 计而非
  `2*BKV*BD*esz`）——⛔ 丢 BC 退成四元组即 K/V 整 D 加载，大 D 桶 BKV 被 UB 压死、
  KV 迭代数成倍膨胀（证伪见 §6.3）

**终态架构（单 kernel + 内联辅助，从零首版直接按此结构落）**：

```python
# 1 计算 kernel：crossformer_attn_kernel；1 内联辅助 _qk_block（不独立启动，禁第二计算 kernel）
# grid = (min(NUM_CORES, total_tasks),)，核内 grid-stride（NUM_CORES 动态探测，910_93=20）
for task in range(pid, total_tasks, NUM_PROG):
    m = tl.full([BQ], -1e30, f32); l = 0; acc = 0        # 全 fp32；SPLIT_D 时 acc0/acc1 两半 D
    for kv0 in range(0, N, BLOCK_KV):                     # dense 全扫描（无 mask 语义）
        s = _qk_block(q, k, ...) * scale_eff              # QK chunk 循环（宽 BLOCK_C），原生 E 操作数
                                                          #   + fp32 累加；fp32 走 input_precision="ieee"
        s = where(kv_valid, s, NEG)                       # 幽灵列掩码（EVEN_KV 时 constexpr 消融），NEG=-3e38
        m_new = maximum(m, rowmax, propagate_nan=ALL)     # MULTI_BLOCK 分档
        alpha = exp(m - m_new); p = exp(s - m_new[:, None])
        l = l*alpha + sum(p); acc = acc*alpha             # L1.3 同步缩放
        if IS_FP32:    acc0 = dot(p, v0, acc0*alpha, ieee); acc1 = dot(p, v1, acc1*alpha, ieee)   # ★ SPLIT_D 并行双半
        elif PV1:      acc = dot(p1, v, acc*alpha)        # ★ 头号杠杆：p1-only 丢 p2 残差（宽容差 + MERE 守卫）
        else:          acc = dot(p1, v, acc*alpha); acc = dot(p2, v, acc)   # hi/lo 二段（小 D 桶 / 擦边桶保留）
        m = m_new
    out = (acc / l).to(E)                                 # ★ 唯一舍入点：单次 cast 回 E
```

> 关键句读：QK 用 `_qk_block` chunk 循环，scale 在 dot **输出侧**乘（操作数 load 直出 = FA L1.19.1）；
> PV 的 p 永远先拆 p1（原生 dtype 可精确表示）——PV1 是头号杠杆但**小 D（96/128）与 MERE 擦边桶
> 必须退回 hi/lo**（§6.5）；SPLIT_D 是**并行双半 dot**（同级累加），链式 = 数学错误（§6.3/§6.5）。

### §6.2 收益归因（14 轮排序）

| 排序 | 改动 | 轮次 / geomean |
|------|------|---------------|
| 1 | **PV1 + plain SPLIT_D 解锁**（17 case -50%+；PV1 per-bucket 分化；case4 MERE 擦边 → TILE_TABLE_EX） | 0.9937 → 1.3881（单轮 +40%） |
| 2 | tile 查表建表 → 扩充 → 五档精调 + 大 D BQ 上调（fp32 BQ=32/64、384 BQ=64） | 0.515 → 0.9937（+93%） |
| 3 | 残留桶深挖：448 (32,256,448) -26%、fp32 192 -40%、896 BKV=256 -13%（N≥8192 守卫）、case14 shape 级 pv1 | 1.3881 → 1.5114 |
| 4 | 溢出路径非降级写法 / 分核再平衡 | 并入上行轨迹的小步 |

### §6.3 ⛔ 证伪方向（不要重跑）

| 死路 | 实测 |
|------|------|
| plain SPLIT_D BQ=32 @ D≥768 | MLIRCompilationError / aicore 挂死（507014）——UB acc 硬墙 |
| BKV=192 | 三档全无增益 |
| fp32 768 BQ=64 | acc 192KB 超限编译失败 |
| EVEN_KV @ D≥768 | 数据依赖 NaN 缺陷（host 守卫规避） |
| plain 448 BQ=64 pv1（96/64） | MLIRCompilationError（UB 墙） |
| plain 768/640 (32,256,128) | 运行时 aicore 挂（load2d 写地址越界族） |
| plain 896 (32,256,128) @ N<8192 | B1 N3721 可复现 aicore 挂（N≥8192 干净）→ host N 阈值守卫 |
| 小桶 probe 结论 | fp32 64/96 同配置复测波动 >12%，信号不可分辨——**小 case tile 结论必须 benchmark 确认，probe 说了不算** |
| tile 表丢 `BC` 退成四元组 | 910b2 实测 geomean 0.30：D≥640 桶 BKV 被 UB 压到 16~32，大 N case KV 迭代膨胀 8~16x（s≈0.09），Phase 4 扫档 +4.5% 即撞结构天花板——正解=五元组 + K/V 按 `BLOCK_C` 分块加载（§6.1/§6.5），**生成器引本卡时核对元组字段数** |

> multibuffer（8 桶零差异）/ load-reorder（全桶 −30~40%，破坏 QK staging）→ 家族死路，见 **§2.4 家族证伪表**；
> 链式 SPLIT_D（数学错误，1.7e7）→ **§6.5**；MERE 擦边即全局回退 → **§6.5**；静态 UB 估算判死大 D 桶 → **§6.6**。

### §6.4 结构域上限证据（为什么 2.0 不可达）

- simulator（plain 1024 代表 shape [1,128,1024] fp16）：
  Cube 核 **MMAD 24.5%** < 50%，MTE1 28.4% + MTE2 27.9% + BAR 13.5%；
  Vector 核非关键路径（75K vs Cube 209K cycles）。
  诊断：**非计算硬件极限**，瓶颈 = BQ=16 微块 k/v 操作数 staging 税 + 跨核同步
  （fa-mha 结构域天花板）。
- 垫底 8 case 全翻倍 geomean 仅 +12%；同区间先例 26_ViTAttention 1.81 / 21_GQA 1.85。
- 修复方向全被 UB 墙或实测证伪（§6.3）——**1.5x 是该结构域的实测天花板**，
  报告按"结构性不可达"收官，不再追加轮次。

### §6.5 ★ 与 FA 卡精度规则的冲突仲裁（宽容差标杆算子，命中 6c/6d 必读）

> 实测依据：同一 45_Crossformer 在异环境（910b2 / CANN 8.5.1）批跑，precheck **完整
> 加载了 §6 终态知识**仍只到 geomean **0.4921**（本卡 1.5114，差 3 倍）——杠杆被
> "保守仲裁 + 错误处置粒度"逐个关掉，报告还误归因"工具链锁死"。三个断点都是规则
> 表述缺口，本节补齐。

**冲突现场**：precheck 同时摘录两组纸面矛盾的规则——FA L1.4/L1.5/L1.16-镜像
（"全 fp32 参考 ⇒ 任何中间量不许落低精度"）vs 本卡 §6.1（低精度走 hi/lo / PV1，
p 降原生 dtype 走 cube）。两卡原先都无优先级声明 ⇒ 保守侧默认获胜。

**★ 仲裁规则（硬）**：判据是 **verify 容差余量**，不是参考实现计算路径——

- 严标杆（逐元素 MERE / matched_ratio≥0.9 卡阈值，FA L1.16 场景）⇒ 镜像规则成立，
  全程对齐参考精度路径；
- **宽容差标杆**（atol/rtol 显著宽于参考链精度；本算子 fp16 atol=9e-2，松约 5 个
  数量级）⇒ 允许利用余量：hi/lo 是 p=p1+p2 **分解**（p1 原生 dtype 精确表示，非
  L1.5 禁止的"直接降"；正确形态见 FA 卡 §4.6）；PV1 丢 p2 残差，以**全量 verify
  通过**为前提（MERE 守卫见下条）。
- **禁止把 PV 操作数 `.to(tl.float32)` 升精度 dot**（fp16 cube 4x 吞吐损失；保守
  仲裁环境下 37 个低精度 case 全体 0.3~0.6x，是 0.49 的主因）；禁止追"与参考链
  逐位对齐"（verify 不要求，参考链口径 ≠ 判据口径）。

**⛔ SPLIT_D 反例（硬）**：唯一合法形态 = **并行双半 dot**（同一 p1 对 v 的前后两半
各做一次 dot，同级累加）；**禁止链式**（前半 dot 输出作后半 dot 操作数，数学错误）。
实测误差在 1e+ 量级（1.7e7）时先查索引/布局 bug，勿判架构死刑——保守仲裁环境由此
全局禁用 SPLIT_D、BQ 锁 16，大 D 桶 staging 税全数保留。

**★ MERE 擦边处置粒度（硬）**：擦边只回退**擦边桶**（EX 表 dtype 级例外 / shape 级
N 阈值），**禁止全局回退 PV1**。实测单桶擦边（fp16@768 小 N：1.0069e-3 vs 阈值
2^-10）被放大成全局 hi/lo 时，-50% 级头号杠杆全部丢失。

**tile 档位：归档结构，不归档数值**。双表结构（五元组 `(BQ,BKV,BC,PV1,SD)`；主表
keyed `(path, BLOCK_D)`，EX 表 dtype 级优先，host 侧 N 阈值守卫兜底）可复用；
**具体档位数值禁止跨环境照抄**——档位是 (芯片, CANN 版本, case 分布) 的函数
（FA F5：结构性改动后必须重扫 tile；FA L1.17 配套：tile 豁免边界禁止跨算子移植）。
重探走探针协议（exp_perf_029：逐配置子进程隔离、reps=3 取 min、~0.2ms 级小 case
波动 >12% 时只信 benchmark）。

> **通例（归档自检三问）**：证伪条目带**正解对照**了吗（缺 ⇒ 实现走样被记成架构
> 死路）？与通用卡冲突写**仲裁行 + 判据**了吗（缺 ⇒ 保守侧默认获胜）？擦边/例外
> 处置写**粒度**了吗（缺 ⇒ 单桶事件被放大成全局回退）？

### §6.6 档位推导方法论（新环境如何长出本环境的档位表）

> 逐桶数值不归档——档位是 (芯片, CANN) 的函数且会过期，可复用资产是
> **把档位推导出来的方法**。910b2 复跑实测 / 定向探针补丁的逐桶数字与收益归因见
> [operators/attention__multiheadattention.md](../operator_playbooks/attention__multiheadattention.md) §2.2
> （证据点保留，非查表）。

- **分桶结构（五元组 tile 表约定）**：`bd_max` 上界升序查表，桶内档位 = 五元组
  `(BQ,BKV,BC,PV1,SD)`（语义见 §6.1，**禁丢 BC 退四元组**）。桶边界按 case 分布的
  D 值设，不必等比——fp32@384/@448 因桶边界缺失跌进 512 桶（BKV 被压到 32）本身
  就是性能 bug，探针协议必须支持**插桶行**。
- **推导流程（每桶 ≤3 候选，13 桶一轮 ≈ 35 min）**：
  1. **种子**：有他环境实测值时取「上限更高环境的档」为种子；没有则按 §6.1 结构
     规则从小档起。
  2. **候选链**：种子 → 折半回退（先 BKV 后 BQ）；调整只走安全方向（BQ 升且 BKV
     不降，反向对价已证伪，见下条与算子手册 §2.2）。
  3. **实测裁决**：子进程隔离编译+跑（exp_perf_029 协议）——编译失败记真墙
     （plan-memory 与 aicore 挂分开记），跑通取 reps≥5 的 min ms 横向比，同步验
     MERE（擦边只回退擦边桶，§6.5）。
  4. **落盘前放开 M3 cap**：静态 prune 会在 host 侧把种子大档杀掉（判据见下条）。
  5. **布局级改动后必重扫**：转置/融合/重排落地会解锁新档位——§4.2 二次优化教训（改完布局不重扫
     tile = 白丢一档，家族规见 §2.4 F-M5）；推导档位表后凡触碰布局必须重走本节流程。
- **★ 静态 UB 估算禁止判死大 D 桶（硬）**：估算公式对大档系统性高估（编译器 buffer
  复用后实测占用远低于 `2*BKV*BC*esz + BQ*BD*4` 字面求和）——实测估算 400KB+ 的
  plain 大档真实编译运行且 3.2x。**"UB 硬墙"结论只能在子进程实测编译探测之后下**；
  M3 前置 prune 的 cap 对探针必须放开。真墙存在但要实测才配写——编译器分配策略是
  （芯片，CANN）的函数，他环境的"可编译"结论不能外推，本环境的"墙"也不能回推。
- **hazard 也是环境函数**：§6.3 的 plain@896 大档 N<8192 aicore 挂在他环境可复现、
  910b2 全档实测**不复现**（N=2173/3721 干净）——N 阈值守卫的阈值本身禁止跨环境
  照抄，换环境需本地实测复验后再决定留/删。
- **BQ/BKV 安全方向（tile 调整唯一安全方向 = BQ 升且 BKV 不降）**：persistent task
  数减半 + cube M 维微块摊销，且必须 UB 预算公式前置校验通过；反向对价（BQ↑BKV↓）
  与正向对价（BKV↑BQ↓）各实测一轮均劣化（-0.5% / -1.6%）。**UB 装不下就不动该桶**
  ——"牺牲一个换另一个"在本结构域无例外（实测数字见算子手册 §2.2）。
- **结构开关的选择判据（不依赖具体数值）**：
  - `BC=0`（K/V 整 D 载）：D≤512 且 UB 富裕时消 nd 内层循环，实测 +10~15%；D≥640
    必须 `BC<D` 分块加载，否则 BKV 被 UB 压死（§6.1/§6.3）。
  - `PV1/SD`：受 dot 链 ≤2（§2.5 M4）与 MERE 双重约束——half 路径 SD=1 必须
    PV1=1；fp16 大 D 桶 MERE 擦边时**只该桶**回退 hi/lo（PV1=0,SD=0），bf16 同桶
    不动（§6.5 与 910b2 复跑两环境各验证一次）。
- **两环境实测差异（证据点，非查表）**：896 桶他环境大 BKV 最优 → 910b2 折半反超
  （可编译但反慢）；1024/512 桶他环境大 BQ 可跑 → 910b2 plan-memory 编译墙逼 BQ
  折半；896 的 N<8192 aicore 挂守卫 910b2 不复现。⇒ 种子压缩搜索空间（约 2/3 桶
  种子即最优或一步回退内），但**每桶最终档必须本环境实测裁决**——种子给出起点，
  探针给出答案。
- **生成器引用本卡时**：主表按 §6.1 五元组结构生成，档位走本节流程实测填充；目标
  环境无实测记录时以他环境档为种子并放开 M3 cap，探针后回写本环境档位。

### §6.7 从零生成首版检查表（命中 6d 必过：首版精度全过 + 性能达标的落地清单）

| # | 检查项 | 反例（不做会怎样） |
|---|--------|--------------------|
| 1 | 判别：无投影、无 head 维、单头 SDPA（q/k/v 缺省取 x，C≥64）⇒ 行 6d → 本卡 §6，不落 FA 行 | 误落 FA 行拿不到 PV1/SPLIT_D/tile 桶治理，杠杆全丢（§6.5：异环境只到 0.49） |
| 2 | host 语义：q/k/v 缺省取 x、feature_shape/group_size 弃用、输出 [B,N,C]、H=1/C 判定 | 语义错 ⇒ 精度全挂（§6.1 host 归一） |
| 3 | fp32 契约：scores/p/acc/l 全 fp32，仅输出一次 cast；**NaN 禁 clamp** | 精度不达 / 末条 NaN 被掩盖 |
| 4 | p 禁单段降 E：hi/lo 分解（宽容差时 PV1 单段 + MERE 守卫；擦边只回退擦边桶） | MERE 超阈（L1.5）；全局回退丢 -50% 杠杆 |
| 5 | PV1 + plain SPLIT_D **并行双半 dot**（per-bucket 分派；小 D 桶保留 hi/lo） | 丢头号杠杆 / 链式数学错误 |
| 6 | tile 五元组 `(BQ,BKV,BC,PV1,SD)` 双表（主表 keyed (path, BLOCK_D)，EX 表 dtype 覆盖）；**禁丢 BC** | 丢 BC 大 D 桶 KV 迭代膨胀 8~16x（§6.3） |
| 7 | persistent `grid=(min(NUM_CORES,tasks),)` + 核内 grid-stride；NUM_CORES 动态探测 | 分核失衡 / 编译爆炸（M1/M2） |
| 8 | 档位以他环境档为种子 + 子进程实测裁决；**静态 UB 估算禁止判死大 D 桶** | 大 D 桶被误杀（§6.6） |
| 9 | 死路前置核查：§6.3 全表 + §2.4 家族证伪表（链式 SPLIT_D / 896 N<8192 地雷 / …） | 重跑已证伪方向，白费轮次 |

> 核对完 9 项再进 Phase 3；精度以 49/49 MERE 为准，性能以**同环境**基准探针为准
> （档位是环境函数，跨环境按 §6.6 重探，禁照抄数值）。

---

## §7 实证：ScaledDotProductAttentionBackward（`mha-bwd`）（10 轮，0.6351 → 1.2969）

> 形态：**反传链**——入参 dO/Q/K/V/O/LSE/is_causal，出参 dQ/dK/dV；无投影段。
> 50 case fp16/bf16/fp32，S 最大 4096，D = 64/128。经 [attention_writing_classification.md](attention_writing_classification.md) 行 6c 路由到本卡，
> 前向主链 Layer 1 仍由 [flash_attention_constraints.md](flash_attention_constraints.md) 提供。
> 终态 **1.2969**（轨迹 0.6351 → 0.8958 → 1.2969），50/50 verify；
> target 2.0 未达（fp32 ieee 精度契约决定 hybrid 上限 ~1.3x，§7.2）。

### §7.1 ★ 终态架构：S_K 二分混合分派（两路径各取所长）

```
path A（S_K < 256）两 kernel
  · dq kernel：dq 计算内联 delta（D = Σ dO∘O）
  · dkdv kernel：dk/dv 融合单 kernel
  · 保留小 case 低发射开销优势（tiny case 实测 2.4~3.7x）

path B（S_K ≥ 256）三件套（大 S 域净赢 34~58%）
  1. _bwd_dq_ws_kernel：dq 计算顺带把 ds = p·(gq−dlt) 按 [B,H,S_K,S_Q] 转置布局
     写入 fp32 workspace（指针算术完成转置，无 tl.trans）
  2. _bwd_dk_gemm_kernel：dk 变纯 GEMM（1 dot/iter）——softmax 重算与 CV 往返
     全部消除；causal q_lo 区间收缩；3D grid (num_kvb, H, B) 直接解码
  3. _bwd_dv_ws_kernel：2 dot/iter（重算 p + dv 累加），去掉批跑版冗余零 dot
  · 改进：kernel 内 .to(tl.float32) 替代 host 侧 5 次 .float() 全量转换；NUM_CORES 动态读取（= §2.5 M1）
  · tile：D>64 → (64,64)；D≤64 → (128,64)；ws：causal 用 zeros（收缩区外 p=0 契约），非 causal 用 empty
```

**架构结论（修正旧"硬件三重锁死"）**：锁死的只是两 kernel 融合架构下的空间；
**跨 kernel 物理解耦**（workspace 物化 dSᵀ + dk 纯 GEMM）在大 S 域净赢——
dk 侧 softmax 重算与 CV 往返的消除 > ws 写读往返的访存成本。

### §7.2 ★ 精度红线：fp16 cube 内部累加 dtype 固定（三次证伪）

- fp16 输入的 `tl.dot` **内部累加 dtype 不随 out_dtype 改变**（固定 fp16），
  反传链上 MERE 超阈 **48x**，hi/lo 拆分也不可修复。
- **唯一合规路径：全 fp32 ieee dot**——load 后 `.to(tl.float32)` +
  `input_precision="ieee"`，输出 cast 回输入 dtype（= FA 卡 L1.1 在反传链的强制版）。
- fp32 cube 吞吐天花板由此决定：hybrid 上限估算 ~1.3x，与实测 1.2969 吻合
  （fp16 cube 快但精度不合规，精度合规的 fp32 cube 慢——没有第三条路）。

### §7.3 ⛔ bishengir 动态循环三禁（不要重跑）

| 死路 | 实测 |
|------|------|
| 动态循环内 dot 操作数带指针算术 | MLIR 编译失败——指针算术必须在 load 前独立完成 |
| 循环体内动态标量 if 分支 | EZ9999 错误——标量分支退 host 判定或改哨兵值算术 |
| tile 贴 UB 上限 | -11.5%——UB 满载时调度余量消失，留 ~10% 余量 |

### §7.4 测量与调度陷阱

> 测量口径 / 调度陷阱（双 stream 降频、环境噪声地板 4.6%、simulator 看不到 host 损耗）
> 家族收敛见 **§2.6**；M1 第二实证（+17.4%）已在 §2.5 M1 标注。以下为早期实测细节。

- **双 stream 降频**：benchmark 双 stream 短跑 +22~25%，持续负载反而 **-17.6%**——
  采样窗口必须足够长，短跑数字虚高不可采用。
- **M1 第二实证**：NUM_CORES 动态读取（= §2.5 M1）在反传链 +17.4%——M1 不是投影段专属，
  任何 persistent/grid 恒定形态都适用。
- **simulator 看不到 host 损耗**：autotune 的 256MB L2 清洗 buffer churn + 编译风暴
  在 msprof 采集里不可见——autotune 不可用时改为**离线调优决策梯固化**（按 D/BH 查表）。
- **环境噪声地板 4.6%**：同代码两次 benchmark 差 4.6%（case 13 受集群离群污染，
  设备频率 1800→800MHz 漂移 77 次）；run 间差 <5% 不得当优化成功。
