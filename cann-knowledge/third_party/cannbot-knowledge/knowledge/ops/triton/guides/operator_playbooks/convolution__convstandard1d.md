---
type: Guide
title: ConvStandard1d 算子（1D 标准卷积） 实现手册
description: 算子类别：convolution1d　典型特征：(N, C_in, L) × weight C_out, C_in//groups, K
  → (N, C_out, L_out)。
tags: [task.implementation, topic.compute.neighborhood.convolution, topic.compute.composite, topic.compute.indexing, topic.data_movement.memory_access, topic.data_movement.layout_transform]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# ConvStandard1d 算子（1D 标准卷积） 实现手册

> 类别约束见 [convolution_constraints.md](../constraints/convolution_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 ConvStandard1d 算子（1D 标准卷积）

**算子类别**：`convolution-1d`　**典型特征**：`(N, C_in, L)` × weight `[C_out, C_in//groups, K]` → `(N, C_out, L_out)`
**性能基准**：50/50 pass，geomean **~0.84x** vs torch（target 0.8x，达标）。约束前上限 ~1.06x（使用 `unfold`/`permute`）；合规后 Path B/D 改 Triton gather，降至 ~0.84x。

### §2.1 Layer 1：特化约束

- **fan_in**（G1）：`fan_in = (C_in // groups) * K`。
- **C1d-NO-LAYOUT-OPS ⚠️ `forward()` 禁止 `torch.unfold` / `torch.permute` / `F.pad` 等布局接口**：padding、im2col、weight 重排必须改用 AST 白名单纯内存操作（`zeros`/`fill_`/`copy_`/`view`/`reshape`）或专用 Triton kernel。违反即使更快也不得保留（1.06x 的非合规实现已因此回退）。
- **G14 ⚠️ `tl.dot` 的 b 操作数必须连续加载**：stride>1 的 strided 2D `tl.load` 喂 `tl.dot` 在 Cube 上 mis-lower（stride 2/3 全错，stride 1 正常；同 b 喂 `tl.sum` 却正确）。**stride>1 输入必须先物化连续 im2col 再 GEMM**，不可 strided 直喂 Cube（隔离脚本 `repro_dot.py`）。
- **C1d-PAD ⚠️ 1D padding 用 `torch.zeros` + `copy_`**：独立 Triton pad kernel 即使 1D grid 全 tile 并行仍占实现耗时 ~80%；`x_pad = torch.zeros((N, C_in, L_pad), ...); x_pad[:, :, padding:padding+L_in].copy_(x_contig)` 在 AST 白名单内（纯内存/形状操作），是跨越 target 的核心杠杆。
- 其余通用约束见 §1 G1–G13。

### §2.2 Layer 2：路径选择

| 条件 | 路径 | 说明 |
|------|------|------|
| `groups==1 && stride==1` | **shifted 1×1 block_ptr GEMM**（A1，首选） | K-tap 分解为 K 个 shifted 1×1 dot，reduction 仅 C_in；block_ptr 输入；消除 host im2col |
| `stride==1 && groups>1` | **general shifted kernel**（opt_iter_2 新增） | scalar 连续输入加载（stride==1 连续，可喂 dot）+ block_ptr group-major 权重；消除 host im2col |
| `stride>1 && groups==1` | im2col + GEMM **直写 + bias 融合**（A3，opt_iter_4） | stride>1 输入非连续（G14），物化 im2col；GEMM 直接写 `out[N,Cout,Lout]` 连续布局并融合 bias，消除 out_2d/reshape/permute/copy/bias-kernel |
| `stride>1 && groups>1` | grouped im2col + GEMM **直写 + bias 融合**（opt_iter_4） | 同上，tile 解码 `(n_batch, g, co_b, l_b)`，按 group 独立 GEMM 直写 `out` |

**统一归约维**（im2col 路径）：`K_dim = (C_in // groups) * K`；权重按 `(C_out_per_group, K_dim)` 扁平连续；`x_col` reduction 顺序 `(C_in_per_group, K)` 须与 weight 展开一致。

### §2.3 Layer 3：关键技巧

- **T1/T8 禁布局接口后的 device 侧重排**（C1d-NO-LAYOUT-OPS）：
  - **im2col（stride>1）**：Triton gather kernel 把 strided 输入列收成连续 workspace；每 program 处理一个 `(n, ci, k)`，对 `l=0..L_out-1` 取 `x_pad[n, ci, l*stride + k*dilation]` 连续 store。groups==1 布局 `(C_in, K, N, L_out)`；groups>1 布局 `(groups, C_in_pg, K, N, L_out)`。
  - **group weight 打包**：`weight_pack_grouped_kernel` 逐元素重排 `[C_out, C_in_pg, K]` → `[GROUPS, C_out_pg, K, C_in_pg]`，缓存 key=weight `data_ptr()`（G9）。
  - ⚠️ device gather 明显慢于原 host `unfold/permute`（受约束不得不采用）；stride==1 路径仍优先 shifted block_ptr GEMM。gather 是**指令吞吐 bound（非带宽 bound）**——见 T9 放大 BLOCK_L。
- **T5 shifted 1×1 分解**（groups==1 && stride==1 主力）：K-tap 拆成 K 个 shifted 1×1 dot，reduction 仅 C_in。输入每 tap 一次 block_ptr load 取整块 `[l+k*D, l+k*D+BN)`；权重 device 转置为 `[C_out, K, C_in]`（G8）后 per-tap 读 `w_ptr + k*C_in`，`strides=(K*C_in, 1)`。**动态 BLOCK 分档**：`_gemm_blocks(c_out, c_in_red, n_dim)` 按 `BLOCK_N`(L_out: 128/64/32) / `BLOCK_M`(C_out: 16/32/64) / `BLOCK_K`(reduction: 16/32/64) 分档，受寄存器上限约束（acc[BM,BN]≤2048、b[BK,BN]≤4096、a[BM,BK]≤2048），block_ptr+boundary_check 天然 tile-size 无关、分档无精度风险。
- **T6 general shifted kernel**（stride==1 && groups>1）：T5 推广到 groups，tile 解码 `(n, g, co_b, l_b)`，scalar 加载 `x_pad[n, g*Ci_pg+ci, l+k*DIL]`（stride==1 连续，可喂 dot），权重 block_ptr 读 group-major 打包 `[GROUPS, Cout_pg, K, Ci_pg]`，tap-k base `w_ptr + g*(Cout_pg*K*Ci_pg) + k*Ci_pg`、`strides=(K*Ci_pg, 1)`（单位 stride reduction 轴，G8）。⚠️ **per-k 偏移不可漏**：布局 `[g,co,k,ci]` 下 base 必须含 `+k*Ci_pg`，否则所有 k 读同一 slice（首版全错）。
- **T7 im2col GEMM 直写最终布局 + bias 融合**（stride>1 两路径）：tile 解码增加 `n_batch`（groups 路径再加 `g`，`g = tile_idx // (m_blocks*n_blocks)` 仅读本组 w/x_col），GEMM 直接按 `out[N,Cout,Lout]` 连续布局写入（`out_ptr + nb*(Cout*Lout)`、`strides=(Lout,1)`），store 前融合 `bias[co]`。消除 `out_2d`/reshape/permute/copy/独立 bias kernel。边界：越界 l_tile 读到下一 batch x_col（合法内存）被 store 的 `boundary_check` 丢弃。
- **T2 K_dim 对齐 C0=16**（G10）：`x_col` 与 weight 的 `K_dim` pad 到 16 倍数。
- **T9 im2col BLOCK_L 自适应**（opt_iter_16 验证，stride>1 路径）：im2col gather 的 `tl.load(x_base + l*STRIDE)` 是指令吞吐 bound（3MB 流量有效仅 ~2GB/s，远低于 HBM 1.2TB/s），故更宽的 BLOCK_L 减指令数有效。`_im2col_block_l(L_out)` 取最小覆盖 L_out 的 2 的幂（≤256，floor 64），大 L_out 单次 masked 迭代。实测 case 25（L_out=256）im2col 1.39→1.09ms（−22%），全量 geomean +1.4%。
  - ⚠️ **已证伪的同类方向**（勿重试）：(1) `tl.make_block_ptr(strides=(STRIDE,))` strided load 与 pointer-tensor gather **同速**（0.98x，maxerr=0），Ascend 不把 1D strided block_ptr lower 成更快的 strided DMA；(2) 大 GEMM tile 调优（放宽 acc 上限到 8192）对 case 25 **无效**——分量 profile 显示 im2col gather 占 impl 97%，GEMM <3%，GEMM tile 不是瓶颈。

### §2.4 性能基准

- **里程碑**：scalar gather 0.016x → im2col+GEMM + shifted 1×1 分解（0.43x）→ general shifted kernel 覆盖 groups>1 stride==1（+9%）→ im2col GEMM 直写+bias 融合（+3%）→ **padding 改 `torch.zeros`+`copy_` 跨越 target（0.73→1.06x，核心杠杆）** → 移除 `unfold`/`permute` 合规后 0.83x → opt_iter_15 大 GEMM tile（证伪，GEMM 占比 <3%）→ **opt_iter_16 im2col BLOCK_L 自适应 0.84x**（case 25 gather −22%）。

---

### §2.5 近期验证记录（2026-08-04）

- **任务**: `6_ConvStandard1d` 50-case 多 shape 评测，架构 `ascend910B1`，target_speedup=0.8。
- **结果**: 50/50 verify pass，geomean **0.8348x** vs torch，平均实现延迟 0.0434 ms，达到目标。
- **Phase 3 关键修复（从 iter_0 38/50 → iter_1 50/50）**:
  1. Path A/B `shifted_gemm_kernel` / `general_shifted_kernel` 内补充 `for ci0 in range(0, C_IN, BLOCK_K)`，保证 `BLOCK_K < C_IN` 时完整规约。
  2. Path C/D `im2col_gemm_kernel` / `grouped_im2col_gemm_kernel` 将 `while k0 < K_DIM_PAD:` 改为 `for k0 in range(0, K_DIM_PAD, BLOCK_K):`，规避 Ascend MLIR 编译器 segfault。
  3. `x_col` buffer 改为扁平零填充布局 `(K_DIM_PAD, N*L_OUT)` / `(groups, K_DIM_PAD, N*L_OUT)`，避免 K_DIM_PAD 越界。
- **Phase 4**: `triton-latency-optimizer` 未给出可落地的结构性优化点；本轮复测与 Phase 3 基线一致（0.8294x → 0.8348x，测量噪声范围）。

---
