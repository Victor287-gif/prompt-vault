---
type: Guide
title: Add 算子（elementwise 逐元素加法）实现手册
description: '算子类别: elementwise。out = x + y，逐元素二元加法，输入支持 scalar/row/跨轴广播（1D-4D），dtype
  fp32/fp16/bf16；单 kernel 统一 persistent no-autotune。'
tags: [task.implementation, topic.compute.elementwise.arithmetic, topic.data_movement.layout_transform, topic.parallelism, topic.memory, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/add.md
  role: primary
created_at: '2026-09-01T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# Add 算子（elementwise 逐元素加法）实现手册

> 类别约束见 [elementwise_constraints.md](../constraints/elementwise_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §1 Add 算子（elementwise）

**算子类别**: `elementwise`（逐元素二元加法）
**典型特征**: `out = x + y`，输入支持 scalar / row / 跨轴广播（shape 1D-4D，最大 `[32,64,128,256]` 67M 元素）；50 case 覆盖 **21 fp32 / 19 fp16 / 10 bf16**，dtype 混合（含全半宽大 case）。torch 侧 5 个广播 case 拆成 Mul+Add 两个 launch。
**性能基准**: 50/50 verify pass。官方 benchmark 轨迹 **0.129→0.352→0.3785→0.4110→0.4455→0.4360→0.5737→0.5951→0.5951→0.5951→0.6015**（round-11 全局最优，kernel `096e7ed8`，msprof 18.63us）。目标 5x 未达成——elementwise add 在 910B2 上 **HBM 带宽 bound**，torch.add 本就不慢。

> **统一尺度重测（2026-09-01，post-stop）**: msprof Task Duration kernel-only、50 case、NPU 1 串行；torch 50-case per-case-total geomean **12.903us**（55-group harness 10.796us secondary）。统一 speedup：r7=0.663（19.462us）、r8/r9/r10=0.655（19.699us）、**r11=0.690（18.705us）**。全局最优保持 round-11。详见 §1.5。
>
> **硬件绑定**: Ascend 910B2（24 cube / 48 vector）/ CANN 9.1.0。UB 192KB。

### §1.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 ★★★ 广播索引 div/mod 是首要标量瓶颈
- broadcast elementwise 逐元素索引若含 `div`/`mod`（如按 `tl.arange` 平铺后的坐标反解），`aiv_scalar_ratio` 可到 **0.634**，直接主导整 kernel。
- **必须**用 **2D grid（外层行 / 内层连续块）+ scalar-broadcast** 消除逐元素 div/mod（G010/G011，round-2 2.7x）。跨轴 3D broadcast 参照 [case_elemwise_broadcast_3d.md](../constraints/case_elemwise_broadcast_3d.md)（先 broadcast 展开+reshape 为 2D）。

#### L1.2 ★★★ tiny-inner 必须保留完整并行 2D grid
- tiny-inner（最后一维很小，如 `[17,31]`、`[100,200]`）**禁止**无条件 program-count cap——cap 会剥夺完整 2D grid 并行度，聚合回归（G019 aggressive cap 已删除）。
- tiny-inner 用**自适应 lanes 隔离**（autotune-set SHAPING：tiny-inner 集 `{256w1,128w1,256w2}` 与主集隔离，G014），round-4 +11.8%。
- tiny-outer（如 case30 `o0=14, n_out=50176`）加 **block-floor>=32** 护栏（round-10 portable 结论，188→156us）。

#### L1.3 ★★★ 半宽输出必须以原生 dtype store
- fp16/bf16 case 的输出**必须按半宽 store**，禁止 fp32 store-promote（写字节减半）。冠军统一 kernel 已天然满足；若出现 promote 是缺陷（G020，round-10 证伪后确认冠军无此问题）。
- fp32 case 禁止 dtype 降宽（容差不可突破）。精度门：fp16 atol=9e-2 / bf16 atol=1e-1 / fp32 atol=1e-3 rtol=1.22e-4。

#### L1.4 large HBM-bound case 的 tiling 杠杆有限
- 大 broadcast case（如 `[32,64,128,256]` bf16 67M 元素）在 910B2 上钉在 **HBM 带宽下限**（`aiv_mte2` 0.98-0.99），tiling 杠杆耗尽——大 tile hoisting（G018）只在未饱和前有效；饱和后唯一杠杆是 **HBM 调度**（读侧 scalar-load 消除 G021 / 写侧 write-coalescing）。

### §1.2 Layer 2: 算法骨架（round-11 冠军统一形态）

1. **单 kernel、单 launch、no-autotune**（G017）：统一 persistent work-item loop，固定小 grid，kernel 内 work-item 迭代覆盖全部 shape 类；消除 per-case host-launch / autotune 开销（round-7 +28.8% 的最大单杠杆）。
2. **2D grid**：外层 = 行 program，内层 = 连续块；tiny-inner 保留完整并行度（L1.2）。
3. **读路径 vector tile-load + register-broadcast**（G021）：SC1 行广播用 vector tile-load 后寄存器广播，消除逐元素 scalar gather。
4. **大 tile hoisting**（G018）：`max_contiguous`/`multiple_of` hint + `num_warps=8` + `BLOCK_SC0=2048`（num_warps=4 回退）。
5. **SC1 条件 program-cap**（G019 窄版本）：仅 SC1 用 `MIN_INNER_SC1=512` cap，宽泛无条件 cap 禁止。
6. **行 tiling hoisting**（G016 窄版）：row-const tile hoisting 对 case38/49 ~3x。

### §1.3 Layer 3: 关键技巧

- **div/mod 消除**：2D grid + scalar-broadcast（G010/G011）——广播 elementwise 的头号标量瓶颈。
- **outer-row batching** RPO=4（G012）：小 launch case（48/49）0.004→0.135。
- **核内 shape 分发**（G013）：kernel 内 constexpr dispatch 替代多 kernel host 分发（G009 的"做对"版本）；SC3 flat-periodic dispatch 修 case30 161.8→25.8us。
- **strided-row persistent**（G015）：tiny-outer 正确形态；flat grid-stride persistent 无效（0.3562）。
- **大 tile hint**：`max_contiguous`/`multiple_of` + `num_warps=8` + `BLOCK_SC0=2048`（G018）。
- **调试方法**：dtype-aware 容差；`aiv_scalar_ratio` / `aiv_mte2` / `aiv_mte3` 定位标量 / 读写带宽瓶颈。

### §1.4 常见陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 逐元素 div/mod 索引 | broadcast 坐标反解 | 2D grid + scalar-broadcast（G010/G011, L1.1） |
| flat grid-stride persistent | case30 形态误判 | strided-row persistent（G015） |
| 无条件 program-count cap | tiny-inner 并行度剥夺 | 仅 SC1 条件 cap（G019 窄版, L1.2） |
| fp32 store-promote | 直觉写 fp32 | 半宽输出原生 dtype store（G020, L1.3） |
| num_warps=4 大 tile | 并行度不足 | num_warps=8 + BLOCK_SC0=2048（G018） |

### §1.5 融合搜索基因经验（fusion-search，2026-09-01）

来源：`triton-optimizer-multi-agent` fusion-search 11 轮进化（基因池 G001–G021），round-8 后 MCTS 扩展；最优 kernel `096e7ed8`（round-11，0.6015x official / **0.690x 统一尺度**）。核心叙事：**scalar 瓶颈（div/mod）→ host 开销（autotune/launch）→ HBM 带宽饱和后的调度杠杆**，三层逐级攻破。

#### G017 ★★★ VALIDATED（R7，+28.8% → 0.5737x，kernel `887aaeb9`，route-2）
**Unified Persistent Work-Item Loop，no-autotune single-launch**：固定小 grid + kernel 内 work-item 循环覆盖全部 shape 类，消除 per-case host-launch / autotune 开销。elementwise 家族在 0.44 plateau 后唯一能破局的**结构性**杠杆（gen-5/6 plateau 的答案）。前置：G010/G011（scalar）、G012（batching）、G014（tiny-inner lanes）为统一 kernel 提供 per-class 形状处理。

#### G021 ★★★ VALIDATED（R11，+1.07% → 0.6015x，kernel `096e7ed8`，route-3）
**Read-Side Row-Broadcast Scalar-Load Elimination / HBM scheduling**：SC1 读路径上 vector tile-load + register-broadcast 消除逐元素 scalar gather（读侧冗余在 HBM 带宽 floor 前的最后杠杆）。结构路线中尺寸广播 case +20-46%（idx18 +100%、idx46 +46%、idx42 +24%、idx39 +20%）；探索路线 case47 -27%、case49 -0.9%。**结论：HBM 饱和后（`aiv_mte2` 0.98-0.99）tiling 无效，读侧调度是正杠杆**（gen-9 G019 + gen-11 G021 双重确认 persistent-read scheduling）。

#### G018 ✓ VALIDATED（R8，→ 0.5951x，kernel `335e5ca3`，route-2）
**Bandwidth-Saturation Large-Tile Hoisting**：`max_contiguous`/`multiple_of` hint + `num_warps=8` + `BLOCK_SC0=2048`，大 case 带宽攻击。num_warps=4 回退。前提是未达 HBM floor；饱和后非杠杆（R9 确认 `aiv_mte2` 0.98-0.99 无 headroom）。

#### G019 ✓ VALIDATED-NARROW（R9，case47 155.8→118.0us -24.2%，promoted active）
**Adaptive Program-Count（SC1 条件 cap）**：`MIN_INNER_SC1=512` 仅对 SC1 cap，窄验证有效。**宽泛无条件 cap 删除**——大 case 回归（case49 1450.2us）且 tiny-inner 必须保留完整 2D grid（L1.2）。round-11 再次确认 aggressive cap 不可用。

#### G014 ✓ VALIDATED（R4 +11.8% → 0.4110x；R10 block-floor>=32 portable）
**Tiny-Inner Adaptive Lanes**：autotune-set SHAPING 隔离（tiny-inner 集 `{256w1,128w1,256w2}` 与主集隔离）。R10 的 `block-floor>=32`（case30 188→156us）为 portable per-case finding。

#### G016 ✓ VALIDATED-NARROW（R6，case46/47 254→163us 1.56x；R7 row-const hoisting）
**General-Broadcast Index Decomposition**：`MIN_INNER=512` 门控的 general-broadcast div/mod 兜底消除；row-const tile hoisting 对 case38/49 ~3x。窄适用（保持 candidate 而非 unconditional active）。

#### G015 ⚠️ SUPERSEDED（R5 active 0.4252x → R7 deleted）
**Tiny-Outer Launch Attack（STRIDED-ROW persistent kernel）**：case30（`o0=14, n_out=50176`）的正确形态（flat grid-stride 0.3562 无效）。被 G017 统一 persistent 循环取代删除——G017 覆盖其收益。

#### G020 ✗ REJECTED（R10，→ 0.5951x，route-2）—— 勿迁移
**Output-Store Dtype Faithfulness / byte-reduction**：冠军统一 kernel **已按原生半宽 store** 半精度输出（out dtype = x.dtype，无 fp32 store-promote）；alpha compute-cast 正确但中性；case40 可复现性回退（123.4→164-168us）。**结论：对已符合 L1.3 的 kernel 无收益**——先验证是否存在 promote 缺陷，而非假设存在。

#### 关键非显然结论
- **广播 elementwise 的第一瓶颈是标量 div/mod**（`aiv_scalar_ratio` 0.634），不是访存——先 2D grid + scalar-broadcast（G010/G011），再谈带宽。
- **per-case host-launch / autotune 开销**在小中 shape 主导——统一 persistent no-autotune 单 launch（G017，+28.8%）是 elementwise 家族最大单杠杆。
- **HBM 饱和后 tiling 失效**（`aiv_mte2` 0.98-0.99）——唯一杠杆是调度（读侧 scalar-load 消除 G021；写侧 write-coalescing 未攻，case49 残留 1.74x gap）。
- **半宽输出原生 dtype store**是正确默认（G020 证伪了"存在 promote 缺陷"的假设）。
- tiny-inner 的完整 2D grid 并行度不可剥夺（G019 宽 cap 已删除）；`block-floor>=32` 是 tiny-outer 的 portable 护栏。

### §1.6 标准 triton-op-generator 复验（2026-09-01，dual-path 双 kernel 形态）

独立于 fusion-search 的标准生成流程（`triton-op-generator`，50 case 同数据）复验了 Add 的 HBM-bound 结论并确认 G014/G015/G017：

- **形态**：双路径 host 分派（GC-14）——无广播 18 case 走 flat 连续 kernel；32 个广播 case 走**行分组 2D persistent kernel**（`[BLOCK_ROWS, BLOCK_N]` tile），标量行解码（`(row//y_div) % y_mod`），逐元素索引纯加法（L1.1 合规）。
- **结果**：50/50 精度通过（max_diff=0）；几何平均 **0.4854 → 0.6682×**（`speedup_vs_baseline` 1.377）。接近 fusion-search 冠军统一尺度 0.690×，确认该 op 天花板在 ~0.69×。
- **关键杠杆确认**：广播 kernel 由「每 work-item 1 行」改「2D tile `[BLOCK_ROWS, BLOCK_N]`」（target ~2048 元素、BLOCK_ROWS≤32）对 23/50 case 大幅提升（小/中 2D 广播 case 2-4×，如 case12 0.53→2.11×）；但收紧 tile（target 1024、BLOCK_ROWS≤16）反而回退到 0.6259× → 本任务局部最优为 `2048/br≤32`。
- **新陷阱**：host 侧 `col_stride`(0/1) 数值误当 `COL_BC` 布尔传入 kernel 会**反转列广播逻辑**（32/50 广播 case 精度失败，diff 1-6）。必须显式 `COL_BC=(col_stride==0)` 转换。详见 `general_debug/exp_debug_add_colbc_inversion_20260901.yaml`。

> **2026-09-03 复验更新（triton-op-generator rerun，并列记录）**：双路径同构实现（flat grid-stride + broadcast **2D 行 tile [BR, BN]**，BR×BN≈2048、BR≤32、BN=min(next_pow2(N),2048)，每 tile 内对 BR 行做标量行解码 `(rows//y_div)%y_mod`）50/50 PASS，几何平均 **0.4856 → 0.6916×**（speedup_vs_baseline 1.424）。2D 行 tile（替代逐 work-item 单行）是 broadcast 侧关键杠杆：多数 broadcast case 从 <0.4 抬到 0.5-0.7+（idx42 0.27→0.73、idx49 0.31→0.79）；微小内维 channel/plane broadcast（N≤64）仍偏慢（~0.04-0.17，launch/scheduling 主导）。flat 大 case 增掩码消除/大 BLOCK 是剩余低价值杠杆。与 HBM-bound 天花板 ~0.69 结论一致。
