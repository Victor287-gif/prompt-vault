---
type: Guide
title: diffusion_embedding 类算子设计约束（扩散模型条件嵌入）
description: 扩散模型条件嵌入类算子（timestep 正弦嵌入 + 小 MLP + 并行投影相加，batch 维极小）：判别特征 F1~F6、类别级通用经验
  DE1~DE10（skinny-M 禁 tl.dot 走向量外积 / 段级深度融合 / 逐舍入点 dtype 契约 / 共享 K 多路 pass-merge /
  tile 整域 law / 两次 run 认证）与常见陷阱。
tags: [task.design, topic.compute.composite, topic.compute.elementwise.activation, topic.precision, topic.programming_model.execution_unit.vector, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-31T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# diffusion_embedding 类算子设计约束（扩散模型条件嵌入）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；
> 与其他类别共用的 GEMM/变换约束继承 [matrix_compute_constraints.md](matrix_compute_constraints.md) /
> [tensor_transform_constraints.md](tensor_transform_constraints.md)，此处只写本类别特有条款。

**核心优化哲学**：batch 维极小的 embedding/投影链——**launch 开销主导 → 按计算阶段深度融合；
GEMM 走向量外积不走 tl.dot；tile 分档必须是覆盖全 batch 域的 law**。**禁止混用**其他类别经验：
不要套用 `linear` 的低秩归约 / attention hoist（本类无 attention 段），不要在 K-loop 内做
gather/加权 prologue，不要按大 M GEMM 的直觉选 tl.dot 或拆 kernel。

## 本类别的算子级实现手册

- [`operators/diffusion_embedding__flux_timestep_guidance_projection_embedding.md`](../operator_playbooks/diffusion_embedding__flux_timestep_guidance_projection_embedding.md) — 10_FluxTimestepGuidanceProjectionEmbedding（本类首个实例，§0/§1 全部规则的实测证据来源）：3-kernel 全融合 + 向量外积替代 tl.dot + BM 分档 law + 共享 K 双 GEMM pass-merge，性能基准 1.4083×、60/60。

## §0 适用范围与算子分类

**类别定义（结构特征，符号化）**——命中任意两条即入本类：

| # | 结构特征 | 符号化描述 |
|---|---------|-----------|
| F1 | 正弦时间/频率嵌入 | 标量或 `[B]` 信号 `×scale` → broadcast 外积 `freqs[D/2]` → `sin/cos` → `concat` → `[B, D_emb]` |
| F2 | 嵌入经小 MLP | `Linear(D_emb→D_inner) → 激活(SiLU/GELU) → Linear(D_inner→D_out)`，宽度为模型超参常量 |
| F3 | 并行投影路 | 另一输入（pooled text / label / 条件向量）经 `Linear(D_in→D_out)` 投到同一输出宽度 |
| F4 | 合并输出 | 两路**相加**（或 concat）→ cast 回输入 dtype |
| F5 | 形态参数 | GEMM 的 M 由 batch 维决定且 ∈ [1, 数百]（**skinny-M**）；N/K 为常量（数百~数千）；权重 MB 级、从 forward 入参传入（无 `__init__` 随机权重） |
| F6 | 精度契约 | 参考实现常显式全程 fp32（每个 `F.linear` 前 `.to(torch.float32)`），bias dtype 与权重 dtype 分离，层后带显式 dtype 回转 |

**家族来源**：Flux / DiT / Stable Diffusion 3 / PixArt 等扩散模型的条件编码器（timestep embedder + pooled text projection embedder），每个采样步执行一次，B 通常很小。

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| 10_FluxTimestepGuidanceProjectionEmbedding | `diffusion-embedding` | F1-F6 全命中；B∈[1,384]，宽度 {768, 3072} | 3-kernel 全融合 + 向量外积替代 tl.dot + BM 分档 law + 共享 K 双 GEMM pass-merge |

**入类流程**：新算子命中本类后，§1 全部约束直接适用；[playbook](../operator_playbooks/diffusion_embedding__flux_timestep_guidance_projection_embedding.md) 中的具体数值（B 分布、宽度、档位边界）是**本例的实测标定值**，新算子必须按自己的形状重新标定，规则形式不变。

## §1 通用经验（类别级，首次生成必须遵守）

> 以下约束按**结构特征 + 形态参数**表述。标注"实测证据"的数据来自 [playbook 实例](../operator_playbooks/diffusion_embedding__flux_timestep_guidance_projection_embedding.md)；规则本身与具体形状无关。

### DE1 ★★★ skinny-M GEMM 默认禁用 tl.dot，走向量外积-累加；入类先 probe 定路径
- **触发条件**：GEMM 的 M ≤ 数百（batch 维决定）且本类别 shape 族
- **默认** 用向量外积-累加替代 `tl.dot`：`acc += tl.sum(a[:, :, None] * w[None, :, :], axis=1)`
- **Why（实测，910B1）**：`tl.dot` 在该形态降级非 Cube 路径——fp32 路径 790µs–269ms、fp16 路径 13.5–67.9ms，比 torch 31–42µs 慢 **300–1600 倍**。TTIR 显示向量外积会被 pattern-match 成 `tt.dot(inputPrecision=tf32)`，该 lowering 在 skinny-M 性能塌陷。
- **决策程序**：首次生成前，用最大 M 的 case 各跑一次 tl.dot 与向量外积单点 probe；tl.dot 慢 10x 以上即全族禁用并记录，不需逐 tile 重试。
- **与 `linear` 的差异**：linear 要求"投影 GEMM 一律 tl.dot"，**本类是例外**——例外由 M 维形态触发，不是算子偏好。

### DE2 ★★★ 按计算阶段深度融合：kernel 数 = 阶段数（典型 2-4），段内禁止再拆
- **必须** 把参考实现的 ~10 个独立 torch op 按计算阶段聚合：`嵌入生成段` / `MLP 段` / `尾段合并（多路投影+add+cast）`，每段一个 kernel。
- **禁止** 把 add / cast / 激活等轻操作拆成独立 kernel；禁止按 shape 段拆分（重复 launch）。
- **禁止** 跨段做生产者-消费者融合（把上游段输出改为在下游 kernel 内即时重算），除非下游对上游输出**只读一次**。判据：若下游 kernel 按 N-tile 切分且每个 N-tile 都需要上游输出，融合会使上游计算放大 N-tile 数倍（实测嵌入段 sin/cos 折进 MLP 段后每 N-tile 重算三角函数放大 24x，MLP 段 79→302µs，60/60 全劣化后回退）。
- **Why（实测）**：小 B 段 launch 与中间量物化开销主导，融合是本类第一收益来源（B≤12 段融合后 **1.95x**）。

### DE3 ★★ 参考实现的每个显式 dtype 转换点都是位级契约，必须 constexpr 逐点复现
- **必须** 全程 fp32 累加；参考实现在线性层后显式 `.to(bias.dtype)`（或其他）回转的，kernel 内用 constexpr 分支逐点复现该舍入后再做后续运算。
- **必须** dtype 组合（权重 dtype × bias dtype × 输出 dtype）经 constexpr 分派到特化路径；跳过任何中间舍入会破坏位级对齐（数值近似对但 verify 以 HiddenStateMismatch 失败）。
- **Why（实测）**：加回中间舍入复现后 60/60 通过。

### DE4 ★★★ 共享 contraction 维的多路 GEMM 必须 pass-merge：单 K-loop 多累加器
- **触发条件**：多条 GEMM 的 K 完全相同且共享激活面板（F3+F4 的标配尾巴：MLP 尾层与并行投影路同 K 同输出宽度）。
- **必须** 融合为单 kernel：一个 K-loop 同时推进各路累加器，输出 tile 几何相同；各路先 round 到自己的中间 dtype（DE3），**之后**才 add/合并，最后统一 cast。
- **Why（实测双路）**：消除 (路数−1) 次 kernel 发射 + 一整趟 tile-index 标量算术 + 中间量 GM round-trip，geomean **+20.9%**。
- **推广**：N 路同 K 投影 → N 个累加器共用一个 K-loop；K 不同则不适用（勿强行合并）。

### DE5 ★★ tile 分档必须是覆盖全 batch 域的一条 law，禁止只碰少数 case 的定制 gate
- **必须** BM（及 BN 特化）分档写成整域 law（如三段式 tier），档位边界由"weight 面板重读次数 vs masked-lane 浪费"的权衡实测标定。
- **档位边界的标定尺**：向量外积路径下 GEMM kernel 时间与 **M-tile 数（NUM_M）近似成正比**（weight 面板 GM 重读是工作单元，masked lane 免费）——档位边界应画在"减少一档 NUM_M"的 B 值处，而非等间距取整（实测边界 32→17 使 B∈[17,31] 从 NUM_M=2 降为 1，5 个 case 各 +33~38%）。
- **Why**：tier 边界移动影响全部 case，在 geomean 层**可认证**；只影响 6/60 case 的单段 gate 即使单段 +57%，也会被未触碰 case 的 ±1-3% profiler 噪声淹没而**不可认证**。
- **禁止** 把小 BM 档扩入中大 B 域：weight 面板重读随 M-tile 数放大（实测 -20~-43%）。

### DE6 ★ no-op cast 编译期剪除
- **必须** 输入已是 fp32 时 `.to(tl.float32)` 仍生成指令；用 constexpr（如 `A_IS_F32`）在编译期剪除。
- **Why（实测）**：+1.5%；多 dtype 任务（fp16/bf16/fp32 混合输入）普遍适用。

### DE7 ★ tile 参数硬顶前置检查 + 起点值
- **必须** 生成前检查 `[BM,BK,BN]` 3D 乘积中间量 ≤ triton 1M numel 上限；`multibuffer=True` 默认开启。
- **起点值**：BM ∈ {16,32,64}（按 DE5 分档）、BK=128、BN=128（可条件加宽到 256）；BM=4/128 及低精度 3D 中间量直接编译失败（`ConvertLinalgRToBi`），不要试——低精度 3D 路线封死即封死，勿绕。
- **BN 加宽条件**（N 向程序数减半、削减主导的标量地址算术）：仅在"加宽后总程序数仍 ≥ 向量核数 × 安全系数"时启用，且必须整域 law 认证（DE5）；BK 缩小（sub-split / BK=8）全线回归，默认不做。

### DE8 ★ 多 kernel 级联禁止共享 grid 常量
- **必须** 每个 launch 独立计算 grid；不同 kernel 的 N 维 tile 数不同时，误共享 → verify 全挂且无报错指向（实测 K2 需 `N_inner//BN` 个 tile、K3+4 需 `N_out//BN` 个，误用同常量 verify 0/60）。

### DE9 ★ 环境：hostname 含特殊字符时 msprof 全挂，必须 sitecustomize shim
- **必须** benchmark 前 `PYTHONPATH=<shim_dir>`（sitecustomize patches `socket.gethostname`）；残余 flaky parse 用 `--max_retries 2`。
- **Why**：hostname 含 `=` → msprof "Invalid input path"，全部 case PROFILER_COLLECT_FAIL；shim 是强制项不是可选项。

### DE10 ★ 收益认证必须连续两次全量 run（同一代码），单次 geomean 不可作为采纳依据
- **必须** 每个"拟采纳"的改动跑两次完整 benchmark（代码不变），两次 geomean 及目标段收益方向一致才可 KEEP；单 flaky case（如 1/60 PROFILER_COLLECT_FAIL）重跑即过的可豁免。
- **Why（实测）**：同代码 run-to-run 噪声带宽 ~±0.2 个 geomean 点（同代码两次 1.3956/1.4083，case 级漂移可达 0.82→2.01）。
- **配套**：跨轮对比时先看两次 run 的自方差；profiler pipe-share 数值只作类型证据不作对比依据。

## §3 常见陷阱与避免方法

### §3.1 通用陷阱（类别级）

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 按大 M 直觉用 tl.dot | skinny-M 降级非 Cube，慢 300-1600x | DE1 probe 决策程序 |
| 漏掉参考实现的显式 dtype 回转 | 位级不对齐，HiddenStateMismatch 难定位 | DE3 逐舍入点 constexpr 复现 |
| 共享 K 的多路 GEMM 各自开 kernel | 多一趟 tile-index 标量算术 + 中间量 GM round-trip | DE4 pass-merge |
| 把 add/cast/激活拆独立 kernel | 小 B 段 launch 开销主导 | DE2 段级融合 |
| 只碰少数 case 的定制 gate | 未触碰 case 的 ±1-3% 噪声淹没收益，不可认证 | DE5 整域 law |
| 单 kernel probe 结果直接外推全量 | probe 环境与全量 run 不同（程序数/噪声） | probe 建议必须全量 run 认证 |
| 跨段生产者-消费者融合 | 下游按 N-tile 切分时上游计算放大 N-tile 数倍 | DE2 判据：仅当下游只读一次才可融合 |
| tier 边界按等间距/2 的幂取整 | 边界不在 NUM_M 降档点上，白付一趟 W 面板重读 | DE5 标定尺：边界画在 NUM_M 减 1 的 B 值处 |
| 把 kernel 内三角函数重算当"免融合收益" | UB live-set 翻倍（3D 中间量 ×2 份）超限编译失败 | DE7 硬顶前置检查 |
| 重算路径中索引解耦错误（cos 半区读 freqs 对错权重列） | sin/cos 两半共享 freqs 但落不同输出列 | 数值对不上先查索引映射（实测 rel err 3.6e-3 来源） |
| 单次全量 run 即下 KEEP/REJECT 结论 | 同代码 run-to-run 噪声 ~±0.2 geomean 点 | DE10 两次 run 认证 |
| K-loop 重排族（A-resident / N-span / dual-N-tile） | 小 B 下程序数变化引发向量核饥饿，收益被淹没 | 默认证伪；NJP 累加链仅支持 {1,2,4}，组合不当 cc overflow / aicore 507014 挂起 |
| 改代码后 verify 失败值"逐位不变" | stale `__pycache__` 缓存旧代码 | 清缓存再 verify |
| 不同 kernel 误共享 grid 常量 | N 维 tile 数不同，verify 全挂无指向 | DE8 独立计算 |
| benchmark 进程未退出就读 perf 结果文件 | 文件按 case 逐个重写 | 先 poll 进程退出 |
| 用 profiler pipe-share 数值做轮间对比 | 同代码不同轮 share 漂移 >20% | share 只作类型证据，不作可比数值 |
| msprof 全 case PROFILER_COLLECT_FAIL | hostname 特殊字符 → "Invalid input path" | DE9 sitecustomize shim（强制） |

### §3.2 方法论（可迁移）

- **TTIR 检查法**：`~/.triton/cache/*/...ttir` 确认外积被 pattern-match 成 `tt.dot(tf32)` 属 lowering 结构性产物，源码层剪不掉。
- **认证粒度律**：改动覆盖整个 B 段的 law → geomean 可认证；只碰 <10% case → noise-dominated 关门。
- **控制组方法**：改动只 gate 在某 B 段时，用未触碰段做对照组；真回归（如 2x）显著高于 ±0.05 噪声。
- **平台期判定**：tile 族优化点全部已试/证伪 + profiler 确认非硬件极限（如 MMAD<50%）但修复点无落地点 → 停止 tile 尝试，结构上限已到。
