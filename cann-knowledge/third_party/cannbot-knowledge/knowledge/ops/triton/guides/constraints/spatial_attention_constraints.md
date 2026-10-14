---
type: Guide
title: spatial_attention 类算子设计约束（CV-Attn-空间自注意力）
description: 特征图 token 化（[B,Lq,D]，Lq=H·W）的空间 self-attention 带插段（空间缩减下采样＋LN 产 K/V、跨
  head 1×1 transform conv、softmax 后 instance_norm，PVT/SR-Attention 系）的判别特征、形态识别五问、通用经验
  S1~S3 与 Layer 1 硬约束 L1.1~L1.8（fp32 全链路 / conv 权重预转置 / tile 档位化）及证伪方向。
tags: [task.design, topic.attention, topic.compute.neighborhood.convolution, topic.precision, topic.compute.normalization, topic.data_movement.layout_transform]
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

# spatial_attention 类算子设计约束（CV-Attn-空间自注意力）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；
> 同属 attention 家族：attention 主链约束继承 [flash_attention_constraints.md](flash_attention_constraints.md)
> （L1.1/L1.3/L1.13、online softmax、面积预算），投影段约束继承 [mha_constraints.md](mha_constraints.md)
> （M1~M4、FA 卡 L1.11/L1.12 权重预转置、§4.5 分档），家族分类路由见
> [attention_writing_classification.md](attention_writing_classification.md)。三份都要读，冲突时以本卡为准。

适用范围：**特征图 token 化（`[B,Lq,D]`，Lq=H·W）的空间 self-attention，且 K/V 生产链或
scores 链带插段**——4 个 `nn.Linear` 投影 + depthwise 空间缩减 conv+LN（ratio>1 压缩 KV）+
`softmax(QKᵀ/√d)V` + 跨 head 1×1 transform conv + instance_norm，PVT/SR-Attention 系。
**核心优化哲学**：目标函数是投影段 fp32 GEMM 算力（占比常 >70%）与 transform 段 S 矩阵
`[B,H,Lq,Lk]` 显存流量（基线 8 遍、最大 case 253MB/遍）的加权和——第一动作永远是
① 投影/conv 权重全部 host 预转置到"读侧连续"布局；② 把 S 遍数从 8 压到 5（数学恒等式消统计遍）。
attention 侧 tile 微调放最后。

## 本类别的算子级实现手册

- [`operators/spatial_attention__emsa.md`](../operator_playbooks/spatial_attention__emsa.md) — EMSA（PVT 系空间缩减注意力）实现手册：Layer 2 四阶段骨架、S 流量模型（8→5 遍）、阶段合并判据、双路径分派、Layer 3 关键技巧与性能基准（50/50、2.0973×）。

## §0 适用范围与算子分类

形态由**入参与写法**识别，不按算子名匹配。

| 子类标签 | 入参/写法特征（可模式匹配） | 优化重心 |
|---|---|---|
| `sa-emsa` | 全插段：ratio>1 空间缩减下采样＋LN 产 K/V（实测形态 depthwise conv），**且** scores 链有跨 head 1×1 conv + softmax 后 instance_norm（apply_transform 门控） | 本卡全章节；S 流量模型是主目标函数 |
| `sa-sr` | 仅空间缩减插段：K/V = LN(下采样(queries))，dwconv / `AvgPool2d(ratio)` / stride-conv 皆可；无 transform/IN | 阶段 A（有权重时邻域 gather＋权重预转置 L1.2；AvgPool 退化纯均值下采样）；attention 回落 FA 卡 |
| `sa-t` | 仅 transform 插段：ratio=1（K/V 直接投影），但 scores 过跨 head conv + IN | 阶段 C 物化链；无阶段 A |
| `sa-plain` | 纯三段式（`x.flatten(2).transpose(1,2)` 后无插段） | **不走本卡**，走 [flash_attention_constraints.md](flash_attention_constraints.md) |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. K/V 生产链含**空间缩减下采样层**＋`nn.LayerNorm(D)`。下采样写法不限——depthwise
   conv（`nn.Conv2d(D,D,kernel_size=ratio+1,stride=ratio,padding=ratio//2,groups=D)`，
   本类实测形态）、`AvgPool2d(ratio)`、stride-conv 皆属本类；共同特征 `Lk < Lq` 且 Lk
   由 H×W 网格几何推导（§0.2 Q2，不同下采样公式不同，逐字复刻参考）；
2. scores 链含跨 head 1×1 transform conv：`nn.Conv2d(num_heads, num_heads, 1)` 作用在
   `[B,H,Lq,Lk]` 上（等价逐平面 `S @ W[H,H]` + bias），常由 `apply_transform` 布尔门控；
3. softmax 后接 `F.instance_norm`（对每个 `(b,h)` 平面 over `Lq×Lk` 归约，`affine=False`）；
4. 输入 queries 形如 `[B, Lq, D]` 且 Lq = H·W（特征图 token 化），投影段 d_k/d_v 可不等。

### §0.2 ★ 形态识别五问（Phase 2 第一步必须回答）

| # | 问题 | 影响 |
|---|---|---|
| **Q1** | golden 是否把全部输入/权重 `.float()` 后计算、仅末尾 cast 回 orig dtype？ | 决定 L1.1 **fp32 全链路契约**是否生效（探针法见 S1）。成立 ⇒ `input_precision="ieee"` 全程 + 禁一切中间降精度；不成立 ⇒ 回落 FA 卡分 dtype 配方 |
| **Q2** | ratio>1？K/V 是否经空间缩减下采样？何种写法？ | 决定阶段 A 存在与 **Lk 几何推导**——dwconv `ho=(H+2·pad−K)//ratio+1`（K=ratio+1, pad=ratio//2）；`AvgPool2d(ratio)` 是 `ho=H//ratio`；ratio=1 时 `Lk=keys.shape[1]`。**Lk 必须逐字复刻参考推导，错一格整盘 shape 崩** |
| **Q3** | `apply_transform` 且 `num_heads>1`？ | 决定双路径分派（playbook §3.4）：False → flash 单 kernel（online softmax，S 不物化）；True → **S 必须完整物化一次**（instance_norm 是 softmax 值的全平面归约，结构上无法 online 融合），走阶段 C 物化链 |
| **Q4** | d_k ≠ d_v / head_dim 非 2 的幂？ | dot 操作数 2 幂封顶 + 分维 mask（QK 用 dk 维、PV 用 dv 维，两套 BLOCK_D 独立分档）；本类实测 D∈[48,1536]、H∈[2,24]，head_dim 普遍非 2 幂 |
| **Q5** | S 规模多大（B·H·Lq·Lk 元素数）？ | 决定 qk+transform 融合分派阈值（实测 **S≤500K 融合**）与 S 物化显存预算（最大 case 253MB fp32，HBM 可容纳）；flash 路径 UB 预算 BQ·BLK≤8192 |

## §1 通用经验（跨形态，首次生成必须遵守）

### S1 ★ 精度契约探测：先钉死计算域再写任何 kernel

host 逐行 replay 参考 forward，找两处证据：① 输入/权重是否 `.float()`；② 输出是否仅末尾
`.to(orig_dtype)`。两者都成立 ⇒ fp32 全链路（L1.1）。**探测结论写进 sketch 头部注释**，
Phase 3/4 全程不可推翻——实测 rel_thr 1.2e-4 下任何一处中间降精度直接越阈。

### S2 ★ tile 参数档位化（bucket 量化）是本类生死线

50 case 的 `(Lq, Lk, D, H, dk, dv, ratio, dtype)` 唯一组合数 → 每组合编译一个 kernel 版本。
**tile 必须量化到固定档位集合**（如 BLK∈{128,256,512}、TOK∈{4,8,16}、投影 tile 小/中/大三档），
且**档位再按实际维度封顶**（`tile = min(档位, bucket(ceil16(维)))`，防小 case 拿大 tile 整块 mask
浪费），编译版本数收敛到 ~30 个且磁盘缓存跨 case 复用。不档位化 ⇒ verify 900s 超时、0 用例启动。

### S3 核数动态读取 + 双核数分流

`NUM_CORES` 模块导入时 `get_device_properties()["num_aicore"]` 读一次（910B2C=20），进程内保持
常量（与 mha 卡 M1 同源）。depthwise conv / LN / gather 类纯 Vector 段可用 vector 核数上限（40），
GEMM/attention 用 cube 核数（20），grid 分别取 min。

## Layer 1 设计约束（首次生成就要全部满足）

> 共 8 条（L1.1~L1.8），全部在本类算子上实测。L1.1/L1.2 决定成败量级，L1.3~L1.5 是编译崩溃/超时类
> （极难从输出反推），L1.6~L1.8 是静默错值类。precheck 必须整节摘录。

- **L1.1 ★★★ 必须 fp32 全链路精度契约**：golden `.float()` ⇒ 实现 `input_precision="ieee"` 全程，唯一舍入点在最终输出 store。
  - **Why:** golden 计算域与输入 dtype 无关；任何中间降精度（输入是 fp16/bf16 也不行）在 rel_thr 1.2e-4 下直接越阈。FA 卡 L1.16（三档 cast / half 路径 hi-lo 双 dot）在本类**不适用**——hi/lo 拆 3 dot 同时违反 mha 卡 M4（dot 链 ≤2）。**"fp32 cube 慢、必须降精度"是误传**：fp32 ieee cube 实测吞吐可达当前硬件标称峰值的 1/4 量级，单 kernel 投影微基准能拿 fp32 实测上限的 ~1/3（*910B2C：27~31 / 85 TFLOPS*）；1.9 TFLOPS 级读数全部来自分段 sync 剖析毛刺。
  - **How to apply:**
    ```python
    acc = tl.dot(a, b, input_precision="ieee")   # 投影 / QK / PV / transform 全部如此
    # ❌ 任何中间降精度（输入是 fp16/bf16 也不行——golden 在 fp32 上算）
    acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    ```

- **L1.2 ★★★ 必须 depthwise conv 权重 host 预转置成 `[K*K, D]` 行主序**
  - **Why:** kernel 内按 `[D,K,K]` 原始布局加载，列主序地址 `d*KKW+kk` 碎成 16 元素宽离散段，该 load 单项拖慢 **8.5×**（完整 kernel 1345us → 157us；load-only 对照同寻址、只 gather + 平凡累加 138us，证明瓶颈就是权重 load 的寻址模式，不是 conv 算本身）。与投影权重预转置（FA 卡 L1.12）同一原理在 conv 权重上的镜像。AvgPool 型下采样无权重，天然无此瓶颈，本条跳过。
  - **How to apply:**
    ```python
    # host: w.reshape(D, K*K).t().contiguous() → [K*K, D]，kernel 行连续 load
    w_tile = tl.load(wt_ptr + kk_offs[:, None] * D + d_offs[None, :])   # [K_PAD, DBLK]
    ```

- **L1.3 ★ 禁止 邻域循环持大 tile**：循环体内的 `[TOK, BLOCK_D]` 会被静态物化 K² 份。
  - **Why:** `for kk in range(K*K): acc += load(x[...]) * w` 使 tile 编译期展开 K² 份，实测 UB 需求 17.2Mb vs 预算 1.57Mb，直接溢出。
  - **How to apply:** 任务切到 (token × d_block)，邻域一次 `[K_PAD, DBLK]` 2D gather（K_PAD=ceil16(K²)），循环体只持一份 tile。

- **L1.4 ★ 必须 LN / 逐 token 类 kernel 的 `TOK × BLOCK_D` 乘积封顶 ~4K 元素**（fp32 多临时）
  - **Why:** 16×1024 组合必溢出（实测 231KB > 192KB UB）。
  - **How to apply:** TOK 按 BLOCK_D **反比**取档：BLOCK_D≤256→TOK=16，≤512→8，≥1024→4。

- **L1.5 ★ 必须 tile 档位化**（见 S2）——违反表现是 verify 全量超时而非报错。

- **L1.6 ★ 必须 3D tile 的 head 维放最后一维 `[BQ, BLK, HP]`**
  - **Why:** `[HP,BQ,BLK]` 布局下 `offs_hp[:,None,None]` 与 `rows[:,None,None]` 在 index 0 广播，HP==BQ 时**静默融合成错误形状**（报 `make_shape_compatible ... 32 and 16`，且报错行距真因 200+ 行）。
  - **How to apply:**
    ```python
    acc3 = tl.zeros((BQ, BLK, HP), tl.float32)
    acc3 += s[:, :, None] * wcol[None, None, :]
    ```

- **L1.7 ★ 必须 pad 行统计污染防护**：任何行级统计 buffer 落盘前 `tl.where(r_ok, x, 0.0)` 逐处包住。
  - **Why:** tile 行 `TI > Lq` 的 pad 行在 softmax 统计中产生 `s_sum=0 → inv=inf → e=0×inf=NaN`。S 本体有 store mask 幸免，**行统计 buffer（Z/q²/part）会全 NaN** 并经 stats 归约传染整个输出（实测 0/50 全 NaN 形态）。
  - **How to apply:** 如 `Z_i = tl.where(r_ok, tl.sum(e, axis=1), 0.0)`，逐处包住再参与归约。

- **L1.8 必须 全局命名常量（`_NEG` 等）加 `tl.constexpr` 注解**
  - **Why:** kernel 内引用 host 全局时 `NameError: Cannot access global variable`。

## §N 证伪方向 / 常见陷阱

### §5.2 ⛔ 证伪方向全表（这一节比优化清单更值钱）

| 方向 | 结果 |
|---|---|
| 邻域 `tl.static_range` 展开 + 逐 tap 1D load | ⛔ 11.8ms，比 2D gather 版差 **8×**（串行小 load 无 tile 化，且踩 L1.3 物化） |
| 缩减 conv 的 DBLK 放大到 512/1024 | ⛔ 无收益（1451 vs 1345us）——行段长度不是瓶颈，**权重寻址模式才是**（L1.2） |
| qk+tconv 无条件全量融合 | ⛔ 大 T case 退化 30-40%（h 循环小 dot + 重复 load）；仅 S≤500K 融合 |
| 分段 sync 剖析定位瓶颈 | ⛔ 冷启动毛刺 70×（96ms vs 真实 1.3ms），结论全错；用单 kernel 微基准 A/B |
| dot 降精度（bf16/fp16 输入 case） | ⛔ 精度契约禁止（L1.1，rel_thr 1.2e-4）；hi/lo 拆 3 dot 违反 M4 |
| `a % b` 取模寻址 | ⛔ checklist 违规 + 本后端代码生成劣化——改 `a - a//b*b`（实测 11 处全改后通过） |
