---
type: Guide
title: GroupedMatmulSwigluQuantV2 算子（grouped-matmul-swiglu-quant-v2） 实现手册
description: '算子类别: grouped matmul + activation + quantization。x [M,K] int8 按 group_list
  切 E 个 expert 区间 @ weight [E,K,N] int8 + per-channel scale → SwiGLU → per-token int8
  quant；expert 边界对齐 BLOCK_M 合并 + 扁平 (m_tile,n_tile) 并行 + partial row_max 免 atomic。'
tags: [task.implementation, topic.compute.composite, topic.compute.quantization, topic.compute.elementwise.activation, topic.tiling, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-31T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# GroupedMatmulSwigluQuantV2 算子（grouped-matmul-swiglu-quant-v2） 实现手册

> 类别约束见 [matrix_compute_constraints.md](../constraints/matrix_compute_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §7 GroupedMatmulSwigluQuantV2 算子（grouped-matmul-swiglu-quant-v2）

**算子类别**: `grouped matmul` + `activation` + `quantization`
**典型特征**: `x [M,K] int8` 按 `group_list`（cumsum=type 0 / count=type 1）切分为 E 个 expert 行区间，各 expert 配 `weight [E,K,N] int8` + per-channel weight_scale；matmul 后 SwiGLU（SiLU gate）再 per-token int8 量化。`dequant_mode=0/1` 框架实际均为 per-channel weight scaling，但 **mode=1 必须走 fp16 预缩放 dot 路径**才能对齐 NPU 数值（int8 dot + fp32 后缩放会产生 off-by-N 量化误差）；`quant_mode=0/1` 框架行为一致均为 per-token scale。
**性能基准**: 50/50 verify 通过；几何平均 **0.7738x** vs torch_npu（19 case ≥1.0x，最高 1.68x）；优化过程 implementation 平均延迟 **0.4129 → 0.0459 ms（约 9x）**

### §7.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 ★★★ 行 tile 必须按 expert 边界对齐，禁止跨界
per-expert 行区间 `[start_e, end_e)` 内切 BLOCK_M tile，`mask_m = offs_m < end_e`；kernel 内标量 E 循环（found-flag + `tl.where(hit,...)`)定位 expert_id/row_base/end_sel/m_local。越界行 x_scale 用 `other=1.0`，workspace 与 partial max **必须同样 mask**，否则以错误 expert 的垃圾值污染下一 expert 的行。

#### L1.2 ★★ 禁止 host 读 group_list（D2H 禁令）
tile 规划只用 M/K/N/E 元数据；grid 用上界 `(cdiv(M,BM)+E)*num_n_tiles`，多余 program 由空 stride 循环跳过。

#### L1.3 ★★ dot kernel grid ≤ num_aicore（910b1=20），纯 vector kernel ≤ num_vectorcore（40）
含 `tl.dot` 的 mix kernel 用 aicore 数；核数查询按 device index 缓存到 `ModelNew._core_cache`。

#### L1.4 ★★★ dot 的 M 维必须 ≥16（核心，latency-bound 修复）
BLOCK_M ∈ {16,32,64} 自适应（取最大 BM 使 `(M//BM)*num_n_tiles >= num_aicore`，链式三元表达式实现，**禁 while**）；连续 `[BLOCK_M, BLOCK_N]` 单 tile。BLOCK_M=1 逐行是 0.079x 的根因；BM 上限 64（UB：acc 2×64×128×4=64KB 主导，峰值 ~180KB，BM=128 必溢出）。

#### L1.5 ★★ 禁 atomic_max，row_max 用 per-n_tile partial + Pass2 归约
partial 布局 `[num_n_tiles, M]`；归约是顺序无关 max，数值 bit-exact。

#### L1.6 ★★ fp16 预缩放路径（dequant_mode=1）BLOCK_K 锁死 256
该路径需物化 w fp16 tile（BK×BN×2B），BK=512 时仅 w_fp16=128KB 必溢出 UB；int8 dot 路径（mode=0）BK 可到 512（dot 数减半，+12% geomean）。host 按 `dequant_mode`（python 标量参数）分档。

#### L1.7 多 case verify 三件套
verify_dir 内 `{op}_torch.py` + `{op}_triton_<impl>.py` + `{op}.json` 同名共存。

### §7.2 Layer 2: 算法骨架（要点）

1. **Pass1**：扁平 tile 空间 `(m_tile, n_tile)`，`m_tile = flat // NUM_N_TILES`、`n_tile = flat - m_tile * NUM_N_TILES`（禁 `%`）；stride 循环 `range(pid, total_flat, NUM_CORES)`；每 program 处理同一 expert 内连续 BLOCK_M 行 × 单 n_tile，K 循环左右半各一次 dot，SwiGLU 后写 fp32 workspace + partial max。
2. **Pass2**：行块循环，先标量循环归约 partial max，再 `inv_scale` 向量量化（round-half-away-from-zero + int32 clip + int8）。
3. **手写 sigmoid**：`gate = acc_left / (1.0 + tl.exp(-acc_left))`，禁 `tl.sigmoid`。

### §7.3 Layer 3: 关键技巧

- **BLOCK_K 分档**：`512 if (dequant_mode == 0 and K % 512 == 0) else 256`
- **count 模式边界累计**：`end_e = boundary + load(offsets+e)`，与 cumsum 模式统一在同一 E 循环（`group_list_type` 作 constexpr 分支）
- **冒烟测试先行**：改 tiling 后先用 5 个代表 case（最大 K mode-0/mode-1、count 模式、最小 M、最大 E）验 int8 maxdiff ≤ 1，再进全量；粗略 python 计时被 host 开销淹没，不可用于判优
- **收益递减判据**：BM 轴 +599%→+19%→+4.4%，BK 轴 +12%；轴收益 <5% 或被 UB 硬锁时停止

### §7.4 性能历程与陷阱

| 版本 | 改动 | geomean vs torch_npu |
|------|------|---------------------|
| 初始 | BLOCK_M=1 行循环 | 0.0793x |
| opt1 | expert 对齐 BM=16 + 扁平并行 + partial max + grid 修正 | 0.5539x |
| opt2/opt3 | BM 32→64 | 0.6600x / 0.6889x |
| opt4 | mode-0 BK 256→512 | **0.7738x** |

| 陷阱 | 修复 |
|------|------|
| BLOCK_M=1 行循环（0.079x） | 行维并入 dot M 维（L1.4） |
| tile 跨 expert 边界 | per-expert 对齐 + end_sel mask（L1.1） |
| atomic_max | partial + 二阶段归约（L1.5） |
| fp16 预缩放路径硬上 BK=512 | UB 锁死 BK=256（L1.6），差距属结构性 |
| forward 内 while 自适应 | AST 校验 Type-3 拦截，改链式三元 |
| kernel 内 `%` | `flat - m_tile * NUM_N_TILES` |
| mix kernel grid 用 vectorcore | dot kernel 用 num_aicore（L1.3） |

**关键结论**:
1. grouped matmul 的首要瓶颈是"行循环锁死 dot M 维=1 + weight 整份重复加载"，expert 边界对齐的 BLOCK_M 合并是 9x 级修复。
2. mode=1（fp16 预缩放）与 mode=0 的 BLOCK_K 上限不同，本质是按数值路径分裂。
3. 剩余差距（偶数 case 全部 mode=1 大 K）是 fp16 物化副本 vs torch_npu 专用融合 kernel 的结构性差距，不要再调参。
