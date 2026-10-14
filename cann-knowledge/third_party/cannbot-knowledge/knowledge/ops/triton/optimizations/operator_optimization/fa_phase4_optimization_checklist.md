---
type: Optimization
title: FA 形态 Phase 4 优化点清单
description: Flash Attention 在性能优化阶段的候选优化点清单与优先级。
tags: [task.optimization, topic.attention, topic.compute.normalization, topic.data_movement.layout_transform, topic.implementation.masking, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# FA 形态 Phase 4 优化点清单

> 类别约束见 [flash_attention_constraints.md](../../guides/constraints/flash_attention_constraints.md)，
> 全局约束见 [global_constraints.md](../../guides/constraints/global_constraints.md)。

## §5 Phase 4 优化点清单

映射到 `triton-latency-optimizer` 的**优化点 #30（Attention/FA 类算子专用优化）**。
完整判别流程、profiling 字段对应表、天花板估算模板见
[`flash_attention_optimization.md`](flash_attention_optimization.md)。

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测增益 | 适用条件 |
|---|------|---------|----------|
| 1 | ★ 全面 `tl.dot` 化 | 不可测量 → 2.36x | `aic_mac_ratio<10%` 且有 `for d in range(head_dim)` |
| 2 | ★ KV 扫描区间按 causal/window 收缩 | **+30%** | 有 causal 或有限 window |
| 3 | ★ `kv_lo` 不对齐 + tile 联合放大 | **+21%** | **两者必须成对做**；单独放大只有 +1.2% |
| 4 | ★ mask 分支 constexpr 特化 + 上下界折叠 | **+21%** | mask 为运行时标量 |
| 5 | ★ BLOCK 开到 UB 上限 | 大 case attention **时延减半** | 放大到编译失败为止 |
| 6 | ★ 删除 host 端 permute/contiguous/pad | **+16%**（保守） | 时间归属出现 `Transpose`/`ViewCopy` |
| 7 | ★ 权重按 tile 预重排 | **+5.0%** | 仅 `fa-mha`，`MTE2>60% 且 MAC<50%` |
| 8 | ★ 分块 dtype 分档 + 并行度保护 | **+2.7%** | 仅 `fa-mha`；缺保护那半 = **−3.7%** |
| 9 | `propagate_nan=ALL` 按块数分档 | 多块 case **−4.4% 时延** | **全局开是 −9%** |
| 10 | CV 指令重排 | 0 / +1.9% / +6.4% | 成本低值得试，不可预期 |
| 11 | 低精度 `p` 二段拆分 | **+2.7%** | 仅 fp16/bf16 |
| 12 | fp32 张量除法 → host 侧倒数 | −36% / 打平 | **仅当 `aiv_vec_ratio ≳ 0.35`** |
| 13 | K 直接读成转置态 | +0.7% | 机理假设已被 IR 证伪，收益来自 load 路径 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱，不要重跑这些死路**）

| 方向 | 结果 |
|------|------|
| 编译开关穷举（14 个逐个单测） | **命中率 0**。`multibuffer` −10%、`enable_mixed_cv` −9%（只在 910_95 生效）、`limit_auto_multi_buffer` 换契约后 −9%。**最多花 20 分钟测完就放弃** |
| 掩码链合并（12 条 → 4 条向量指令） | **0%** |
| `BLOCK_D == D` 时消掉 d 方向掩码（45/50 适用） | −1.3%（噪声内） |
| `libdevice.tanh` 消掉 softcap 的 fp32 除法 | **两次测量均打平** |
| bool 掩码换 fp32 加性 bias | **编译失败**（UB 超限） |
| 运行时 `scf.if` 只在对角/边界块做掩码 | 噪声内 |
| K/V permute 成连续布局 | **15/17 case 变慢**。先用 `.contiguous()` 花 5 分钟证伪，别去实现零拷贝方案 |
| KV 循环拆「无掩码整块 + 对角/尾块」 | **编译失败**（循环体复制使 UB 翻倍） |
| 单块时去掉 kv 的 `scf.for` | 一个算子 **−10%**（编译器失去 multi-buffer 流水机会）；另一个 **aicore exception 507015**，三种写法全崩 |
| 保留 `for`、把累加改赋值 | `'scf.for' op 0-th result` 编译失败 24/50 |
| grid 超订（24→48） | 无收益 |
| `exp2` + log2e 折进 scale | 无收益（`tl.exp` 本就走 exp2） |
| scale 与 mask 合并成一次 FMA | **编译失败**（UB 超限） |
| 波次量化单目标代价模型 | **−21%**（模型不含访存项，会选出 16×64 这类小 tile） |
| tile 按代价排序 + 编译失败自动回退 | 完全打平（次优候选之间排序不可靠） |
| **纯 Vector 版 attention**（去掉 `tl.dot` 想编成 `AI_VECTOR`） | **小 case 子集 −33%**。根因：triton-ascend 按"有没有向量侧计算"决定 kernel 类型，**去掉 `tl.dot` 仍是 `MIX_AIC`**，只是把矩阵乘搬到 Vector 上还保留全部 MIX 握手开销 |
| `al.scope(core_mode)` 手写 CV 流水 | **`bishengir-compile` LLVM stack dump**（910B2+CANN 8.5.1）。投入前先用最简两段 kernel 做 10 分钟门槛测试 |
| `al.multibuffer` 按张量开双缓冲 | **产生 NaN**，40/50，失败集恰为全部多块 case |
| `al.parallel` 用于 KV 循环 | **40/50 → 33/50 精度崩**。online softmax 有跨迭代依赖（`m`/`l`/`acc` 递推），**语义上不可并行**；用于外层 task 循环语义正确但实测更慢 |
| 手工循环不变量外提 / 手工 buffer 复用 / 手工删常量张量 | **三次实测全部 no-op**。判据：两轮的失败数完全相同（都是 6/50、44 个编译错），而后者比前者多"释放"了一块 `[128,64]` fp32 ⇒ 省下的 UB 是幻觉。**源码层面的内存优化在 triton-ascend 上基本无效，UB 手算模型预测不了真实边界** |
| 批处理 2 个 KV tile 凑重叠 | **−14%**，且串行度 1.69→1.71 纹丝不动 ⇒ 编译器仍逐条串行。**流水要在迭代内部做，不是迭代之间** |
| 手工软件流水（提前发射下一块 QK） | 8/50，多持一个分数块撑爆 UB |
| K 预转置为 `[B,Dh,S]` 消除 `tl.trans` | **回归**。host 侧转置只在 `M*N*K ≥ 1e12` 才划算，本规模下 kernel 内 `tl.trans` 才是正解（注意与 §2.12「K 直接读成转置态 +0.7%」不同：那是 **load 成转置态**，不是 host 预转置） |
| 按 head 分组共享 K/V（`HG=2`） | 不值得做：`HG=2, BLOCK_SQ=64` 与 `HG=1, BLOCK_SQ=128` 的 K/V 扫描遍数**完全相同**、UB 也相同，是同一个杠杆的两种写法 |
| 把 sq tile 拆散做三维并行 `(B,H,S-tile)` | **回归**。K/V 跨 sq 的复用消失、访存随遍数倍增。⚠️ 与 L1.13 是同一个量的两个方向：**每次 K/V 加载服务多少 query 行** |
| 非融合三段式（QK → softmax kernel → PV） | **访存账否决**：物化 `[B*H,S_Q,S_K]` fp32 = 370MB，4 趟 ≈ 1.5GB ≈ 900us，而当前整个 kernel 才 400us |

### §5.3 ⛔ 结构性下限：小 shape 上**绝不要**拆 kernel

任何含 softmax / elementwise 的 kernel 在 triton-ascend 下**必然编成 `MIX_AIC`**，每次发射带 **~4.5us** 固定成本，是工具链的结构性下限，**改写数学表达绕不开**。
⇒ 小 case 的整个 impl 时延也才 6~8us，**每多拆一个 kernel 就多 ~4.5us**。
判别：看 `kernel_details.csv` 的 `Accelerator Core` 列（`MIX_AIC` vs `AI_CORE`），或看 IR 里有没有 `_mix_aiv` 函数。

### §5.4 三个会误导的 profiling 字段

| 字段 | 陷阱 |
|------|------|
| `aic_scalar_ratio` 高 | ❌ **不是标量算术过多**。三个算子上五次改法全落空。真实构成是 113 条同步指令（`set_flag`/`wait_flag`/`pipe_barrier`/`sync_block`）的等待被计入标量管线。**先 dump IR 数同步指令，不要直接改标量代码** |
| `cube_utilization(%)` | ❌ **名字骗人**。同一行 `cube_utilization=88.383%` 与 `aic_mac_ratio=0.084` 并存。判断 Cube 忙不忙**一律用 `aic_mac_ratio`** |
| `aiv_vec_ratio` 高 | ⚠️ **半对**。判据是「这条指令值多少拍」不是「有几条指令」：去掉 1 条 fp32 `vdiv` = −36%，去掉 8 条掩码指令 = 0% |

---

## 失效边界与不触发场景

- **不触发**：算子不属于 FA 形态（无 KV 维度分块循环、无跨迭代滚动的 online
  softmax 状态量 `m`/`l`/`acc`）时，本清单整体不适用，应回到通用优化点库。
- **不触发**：精度闸门未过时不得进入本清单——先过闸门再谈性能，
  见 [FA 精度闸门与测量口径](../../guides/operator_playbooks/flash_attention__precision_gate_and_measurement.md)。
- **失效**：结构性改动（KV 区间收缩、权重重排）之后本清单中基于历史 tile 的
  结论全部失效，必须重扫，不得沿用旧最优配置。
## 失效边界与不触发场景

- **不触发**：算子不属于 FA 形态（无 KV 维度分块循环、无跨迭代滚动的 online
  softmax 状态量 `m`/`l`/`acc`）时，本清单整体不适用，应回到通用优化点库。
- **不触发**：精度闸门未过时不得进入本清单——先过闸门再谈性能，
  见 [FA 精度闸门与测量口径](../../guides/operator_playbooks/flash_attention__precision_gate_and_measurement.md)。
- **失效**：结构性改动（KV 区间收缩、权重重排）之后本清单中基于历史 tile 的
  结论全部失效，必须重扫，不得沿用旧最优配置。
