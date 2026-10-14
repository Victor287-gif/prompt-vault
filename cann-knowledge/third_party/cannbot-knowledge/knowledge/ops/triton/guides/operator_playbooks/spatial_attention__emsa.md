---
type: Guide
title: EMSA 算子（PVT 系空间缩减注意力） 实现手册
description: '典型特征: K/V 生产链含空间缩减下采样（ratio>1 depthwise conv 或 AvgPool2d）＋LayerNorm，scores
  链含跨 head 1×1 transform conv（apply_transform 门控），softmax 后接 instance_norm（affine=False，逐
  (b,h) 平面全平面归约），输入 queries[B,Lq,D]（Lq=H·W）、d_k≠d_v。'
tags: [task.implementation, topic.attention, topic.compute.neighborhood.convolution, topic.precision, topic.compute.normalization]
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

# EMSA 算子（PVT 系空间缩减注意力） 实现手册

> 类别约束见 [../spatial_attention_constraints.md](../constraints/spatial_attention_constraints.md)，
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §2 EMSA 算子（spatial_attention）

**算子类别**: `spatial_attention`　**典型特征**: K/V 生产链含空间缩减下采样（ratio>1 depthwise conv）＋LayerNorm，scores 链含跨 head 1×1 transform conv，softmax 后接 instance_norm（逐 (b,h) 平面全平面归约，affine=False）。
**性能基准**: 50/50 精度通过，几何平均加速比 **2.0973x**（target 2.0 达标；910B2C 20 AI cores / CANN 9.1.0 / triton-ascend 3.2.2；Phase 3 首版 1.1088，Phase 4 七轮 → 2.0973；最快 case 7.19x，最慢 5 个 0.82~0.86x）

### §2.1 Layer 1 设计约束（算子专属）

本算子逐条适用 [../spatial_attention_constraints.md](../constraints/spatial_attention_constraints.md) Layer 1 的 L1.1~L1.8，precheck 必须整节原文摘录。算子级要点：

- **L1.1 ★★★ 必须 fp32 全链路**：golden `.float()` ⇒ `input_precision="ieee"` 全程 + 禁中间降精度，唯一舍入点在最终输出 store。
- **L1.2 ★★★ 必须 depthwise conv 权重 host 预转置 `[K*K,D]`**（该 kernel 8.5×，本算子头号杠杆之一；AvgPool 型下采样无权重，本条跳过）。
- **L1.3 ★ 禁止 邻域循环持大 tile** / **L1.4 ★ TOK×BLOCK_D≤4K** / **L1.5 ★ tile 档位化**（生死线）。
- **L1.6 ★ 必须 3D tile head 维最后 `[BQ,BLK,HP]`** / **L1.7 ★ pad 行统计 `tl.where` 防 NaN** / **L1.8 全局常量 `tl.constexpr`**。

### §2.2 Layer 2 算法骨架（四阶段：空间缩减 → 投影 → attention 主链 → 输出投影）

阶段职责与任务分解是**参考方向**——子类不同（`sa-sr` 无 transform 段、`sa-t` 无缩减段）、shape
不同，最优 kernel 数在 4~8 之间浮动。照抄**阶段职责与合并判据**才有意义；kernel 命名与数量是具体
算子的终态产物，重新设计（四层隔离：Layer 2 只给骨架方向，输出必须是全新草图）。

```
阶段 A（ratio>1）空间缩减   queries[b,Lq,D] --depthwise conv＋LN--> reduced[b,Lk,D]
    · Lk = ho·wo 由 H×W 网格几何推导（K=ratio+1、pad=ratio//2），逐字复刻参考（判别卡 §0.2 Q2）
    · conv 任务 = (out_token × d_block)：邻域一次 [K_PAD, DBLK] 2D gather，
      权重 host 预转置 [K²,D] 行主序（L1.2），向量 FMA 无 dot（depthwise 无 Cube 需求）
    · 邻域循环禁止持大 tile（L1.3）；LN 与 conv 默认分离两个 kernel——LN 需整行 D 归约，
      TOK×BLOCK_D 预算（L1.4）与 conv 的 DBLK 档位互相牵制（§3.3 判据）

阶段 B          投影          Q/K/V = src @ W{q,k,v}，src = ratio>1 ? reduced : keys/values
    · 投影权重全部 host 预转置 [in,out]（FA 卡 L1.12）；tile 按 shape 分档
      （大 128×128×128 / 中 64×128×128 / 微 16~32，判别卡 S2 档位化）
    · K 投影产出转置布局 [B,H·dk,Lk]：直接算转置 GEMM Wk@xᵀ（权重保持原始 [out,in] 免转置），
      读侧连续、免散射 store（§2.6 转置 GEMM）
    · d_k≠d_v ⇒ QK/PV 两套 head_dim 档独立分（判别卡 §0.2 Q4）

阶段 C          attention 主链   host 条件 launch 分派（禁 kernel 内 constexpr 分支，§2.5）
    · 面积预算管两处 tile：S tile（BQ·BLK）与累加器 tile（BQ·BLOCK_D）都 ≤8192，预算内 BQ 优先放大
    · transform=False → flash 单 kernel：online softmax＋PV（FA 卡骨架），dot 链=2，S 永不物化；
      Lk ≤ BLK 一块装下时按 constexpr 特化单趟精确 softmax（免 online rescale——ratio 缩减后 Lk 普遍小）
    · transform=True → S 物化一次（IN 是 softmax 值的全平面 (b,h) 归约，结构上无法 online 融合）：
      ① QK·scale 与跨 head 1×1 conv：小 S 融合单 kernel（3D 累加器 acc[BQ,BLK,HP] += s[:,:,None]·wcol，
        h 循环每迭代 1 dot；HP 取 head 的 2 幂档 16/32，权重 host pad 补零）；大 S 分离两个 kernel
      ② softmax 两遍法：pass1 行 max；pass2 存未归一 e ＋ 行统计 Z_i=Σe、q2_i=Σe²（pad 行 tl.where，L1.7）
      ③ IN 平面统计：两级确定性归约（部分和 [B·H·n_it,2] → 归约 kernel → [B·H,2]；禁 fp32 atomic 累加）；
        mean ≡ 1/Lk 恒等免统计遍；sumsq = Σ_i q2_i/Z_i²（§2.6 平面统计）
      ④ PV：读取时归一 p = e·(rstd/Z_row) − mean·rstd，dot(p,V)（归一遍消失）

阶段 D          输出投影      O @ Wo ＋ bo → cast orig dtype（全链路唯一舍入点，L1.1）

host
    · ⚠️ 权重经 forward 传参（非 __init__ 持有）⇒ 首次用时预转置＋按 (data_ptr,shape,dtype)
      跨调用缓存（第二次起零开销，缓存超限整清防泄漏）；小权重（如 conv 的 D×K²）现转开销 µs 级
    · apply_transform and num_heads>1 决定阶段 C 分派；S workspace 仅物化路径分配
```

### §2.3 S 矩阵流量模型（transform 路径的目标函数）

S = `[B,H,Lq,Lk]` fp32。基线（逐算子分离实现）**8 遍全量流量**，终态 **5 遍**：

| 削减 | 手段 | 省几遍 |
|---|---|---|
| softmax 两遍法（存未归一 e + 行统计 Z/q²） | 统计并入 pass2，免第三遍读 | −2 |
| 平面统计数学分解（§2.6） | mean≡1/Lk 免统计遍、sumsq 由行统计组装 | （含上行） |
| IN 归一并入 PV 读取（`p=e·rstd/Z − mean·rstd`） | 免独立归一 kernel 的读写 | −1 |

最大 case（B=2,H=20,Lq=Lk=1258）S≈253MB → 8 遍 ≈ 2.0GB → 5 遍 ≈ 1.27GB，直接决定该 case 档的
延迟下限（MFU 3.12% 属结构性预期，非实现问题）。

### §2.4 阶段合并判据（Phase 2 草图期逐条回答）

| 判据 | 融合 | 分离 |
|---|---|---|
| **QK 与 transform conv** | S ≤ **500K**（全局 B·H·Lq·Lk，判别卡 Q5）：融合单 kernel（QK 与 conv 同体），3D 累加器 `acc[BQ,BLK,HP] += s[:,:,None]·wcol`，h 循环每迭代 1 dot | S > 500K：h 循环重复 load q/kt + `[16,dk]×[dk,32]` 小 dot，大 T case 实测退化 30-40% |
| **缩减 conv + LN** | D 小（≤256）时可试融合 | 本类默认分离：LN 需整行 D 归约，TOK×BLOCK_D 预算（L1.4）与 conv 的 DBLK 档位互相牵制 |
| **k 投影与 Kt 布局** | 转置 GEMM 直接产 `[B,H·dk,Lk]`（§2.6） | 自然 GEMM + 转置 store = 多一个 scatter 通道，实测恒劣 |

### §2.5 双路径分派（判别卡 Q3 的落点）

`transform=False` → flash 单 kernel（S 永不物化，Lk 缩减收益直接体现）；
`transform=True` → 物化链（S 物化一次，流量按 §2.3 账本优化）。
**分派在 host forward 做条件 launch，不在 kernel 内 constexpr 分支**（4 段 + attention 合并进
同一 `@triton.jit` 是 mha 卡 §2 明令禁止的形态）。

### §2.6 Layer 3 关键技巧（可参考不可照抄）

- **★★★ 平面统计数学分解**（instance_norm 两遍扫描终态，S 8→5 遍的核心）：
  `mean(p) ≡ Σ_i(Σ_j p_ij)/(Lq·Lk) = 1/Lk`（每行和为 1，免统计遍）；`Σp² = Σ_i q2_i/Z_i²`（行局部量
  组装）。归一挪到 PV 读取时一次完成 `p = e·(rstd/Z_row) − mean·rstd`（两系数先在 stats 归约 kernel
  算成行向量）。实测单轮 +2.5% 几何平均，大 S case 受益最大。
- **★★ conv 权重预转置 8.5×**：完整 kernel 1345us → 157us（L1.2；load-only 对照 138us 证明瓶颈是
  权重 load 寻址，不是 conv 算本身）。
- **★ 转置 GEMM 免散射**：需要 `[B,H·dk,Lk]` 转置布局（KV 循环读侧连续）时直接算 `Wk @ xᵀ`
  （M=H·dk, N=Lk），权重保持原始 `[out,in]` 行主序**免预转置**，store 自然落转置布局，读写两侧连续。
- **★ softmax 两遍法**：pass1 行 max → pass2 `e=exp(s−m)` **不除 Z 直接 store**，同时累行统计
  `Z_i=Σe`、`q2_i=Σe²`（供平面统计组装）；pad 行 `tl.where(r_ok, q2·invz, 0)`（L1.7）；下游 PV 才归一。
- **★ 跨 head 1×1 conv 融进 QK（小 S 专属）**：h 循环内 `acc3[BQ,BLK,HP] += s[:,:,None]·wcol[None,None,:]`，
  每迭代 1 个 QK dot（M4 合规）；大 S 的 h 循环重复 load + 退化 dot，不做。
- **诊断方法论**：load-only 对照（同寻址、只 load + 平凡累加的对照 kernel，纯 gather 138us vs 完整
  1345us 一行定位访存瓶颈）；单 kernel 20 次平均微基准 A/B，**不要用分段 sync 剖析**（冷启动毛刺
  96ms vs 真实 1.3ms，差 70×）。

### §2.7 性能基准 / 优化记录表（50-case 几何平均判定，七轮实测轨迹）

| # | 方向 | 实测增益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ 投影 shape 分档 tile（微 tile → 128×128×128）+ attention 面积预算 4096→8192 + softmax 多行/program | 1.1088 → **1.5302**（+38%，单轮最大） | 总是 |
| 2 | ★★★ 缩减 conv 权重预转置（该 kernel 8.5×）+ softmax/stats 融合 | → 1.8073（+18%） | ratio>1 |
| 3 | ★★ softmax 两遍法 + 平面统计数学分解（S 8→5 遍） | → 1.9425（+2.5%，大 S case 更多） | transform 路径 |
| 4 | ★★ tconv tile 放大 + softmax BLK 512 档 + flash BQ 优先 | → 1.8944（+5%） | 总是 |
| 5 | ★ 小 S 的 qk+tconv 融合分派（S≤500K）+ BLK 256 档 | → 1.9784（+1.8%） | transform 且 S 小 |
| 6 | ★ 融合阈值 100K→500K + tconv 加宽 + TI 放大 + pv BLK 256 | → **2.0973**（+6%） | 同上 |
| 7 | tile 档位化 bucket（判别卡 S2/L1.5） | 编译爆炸 → 可运行 | 首版必须（非收益项） |

最终分布：≥2x 共 29 / 1.5~2x 共 7 / 1~1.5x 共 8 / <1x 共 6；最快 case 7.19x（小 Lq + 大 D 投影主导），
最慢 5 个 0.82~0.86x（超大 Lq/D 的 transform 或大 flash 形态，结构域上限）。

**天花板估算**（先标定、后算账，数字随 NPU 型号/集群状态变化，禁止直接搬用）：① S 流量下限 =
遍数×S 字节÷实测 HBM 带宽（任何 tile 微调都越不过，优化只剩减遍数）；② cube 吞吐上限现场标定、
判据用百分比——投影段微基准 <40% ⇒ 剩余空间在 tile/流水调度，≥50% ⇒ 已近结构域上限转攻流量侧
（*910B2C 示例：fp32 实测 85 TFLOPS，投影段 27~31 = 35%*）；③ 噪声带自测双跑：per-case 均值差 >5%
判优化无效，geomean 达标判定必须双跑。

### §2.8 陷阱 / 证伪方向

| 现象 | 根因 | 处理 |
|---|---|---|
| verify 900s 超时、0 用例启动 | tile 唯一组合编译爆炸 | 判别卡 S2/L1.5 档位化 |
| `PlanMemory Failed` / UB 溢出 17.2Mb | 邻域循环持大 tile 静态物化 K² 份 | L1.3 |
| LN kernel UB 溢出 231KB | TOK×BLOCK_D 超 4K | L1.4 反比档 |
| 输出大面积/全部 NaN（0/50） | pad 行 `0×inf` 统计污染 | L1.7 `tl.where(r_ok,…)` |
| `make_shape_compatible ... 32 and 16`（报错行距真因远） | 3D tile head 维在 index 0 广播 | L1.6 `[BQ,BLK,HP]` |
| `NameError: Cannot access global variable` | host 全局常量未 constexpr | L1.8 |
| 融合版大 case 退化 30-40% | h 循环小 dot + 重复 load | §2.4 S≤500K 才融合 |
| 权重 load 单项 8.5× 慢 | conv 权重列主序碎段 load | L1.2 预转置 `[K²,D]` |
| Lk 推导错一格，全盘 shape 崩 | 未逐字复刻几何推导 | 判别卡 §0.2 Q2 |

**精度判定顺序**（错一步会把 bug 归错类）：① 先看是否全 NaN（L1.7 pad 污染典型形态是 0/50 全 NaN，
先查行统计 `tl.where`，不要调精度）；② 再看覆盖率（`torch.empty` workspace + 部分路径未写 =
未初始化内存）；③ 最后才是精度契约（判别卡 S1 探测结论核对每处 dot 的 `input_precision` 与唯一
cast 点，L1.1）。舍入点契约：全部中间量无舍入（load 后 .to(f32) + ieee dot），最终输出 store 唯一
一次 `.to(orig_dtype)`。
