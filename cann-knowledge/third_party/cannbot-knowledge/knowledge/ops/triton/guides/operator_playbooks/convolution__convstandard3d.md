---
type: Guide
title: ConvStandard3d 算子（3D 标准 / grouped，源 benchmark 实际 `kernel_size=(k,k,1)`） 实现手册
description: "算子类别：convolution3dstandardgrouped\t典型特征：(N, C, D, H, W) × weight C_out,\
  \ C_in//groups, Kd, Kh, 1。"
tags: [task.implementation, topic.compute.neighborhood.convolution, topic.data_movement.layout_transform, topic.alignment, topic.implementation.masking, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# ConvStandard3d 算子（3D 标准 / grouped，源 benchmark 实际 `kernel_size=(k,k,1)`） 实现手册

> 类别约束见 [convolution_constraints.md](../constraints/convolution_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §5 ConvStandard3d 算子（3D 标准 / grouped，源 benchmark 实际 `kernel_size=(k,k,1)`）

**算子类别**：`convolution-3d-standard-grouped`	**典型特征**：`(N, C, D, H, W)` × weight `[C_out, C_in//groups, Kd, Kh, 1]`；源 benchmark 固定 W 维 kernel 为 1，因此等价于在 D-H 平面做 2D 卷积并对 W 切片独立计算。
**性能基准**：60/60 pass，geomean **1.1332x** vs torch（target 0.8x，**达标**）。主力提升来自 P3 改为 no-permute 布局 + HW 展平 dot（opt_iter_26，见 §5.3 T20 / §5.5）。

### §5.1 Layer 1：特化约束

- **fan_in**（G1）：`fan_in = (C_in // groups) * Kd * Kh`（`Kw=1`，不显式乘）。
- **kernel_size 退化维显式处理**：任一维为 1 时坐标直接按 `ox*stride - pad` 计算，**禁止**写该维 `for k in range(1)` 循环。
- **batch 偏移必须用每 batch 平面体积**（⚠️致命）：`base = x_ptr + n * CDHW + ...`，其中 `CDHW = C*D*H*W`。**禁止**用 `n * NCDHW`（会跳到第 N 个 batch 之后，batch>1 时后续 batch 全零/越界）。
- groups>1 按组处理（G12）：`g = co // C_out_per_group`，权重 shape `[C_out, C_in//groups, Kd, Kh, 1]`。
- **forward() 禁止 torch/F 布局/计算接口**：padding、pack、permute、copy 切片赋值等必须封装进 `@triton.jit` kernel（G6）。

### §5.2 Layer 2：路径选择优先级

1. `C_in_per_group==1 and C_out_per_group==1` → **scalar depthwise fallback**（4D block_ptr depthwise 在该 workload 上已证伪，必须回退）。
2. `Cout_pg < 16` 且 groups 足够大、`GROUPS_PER_M * Cin_pg ≤ 16` → **groups-batched M=16 path**。
3. `stride==1`、`C_in_per_group≤16`、`C_out_per_group≤64`、`K≤5`、`H_out≥2` → **P3 no-permute 布局 + HW 展平 dot**（主力，详见 §5.3 T20）。
4. `stride==1` → **no-permute NC1DHW0 path**（保留 `[N,g,C1,C0,D,H,W]`，W 内层连续）。
5. 其余 → **标准 NC1DHW0 pack + 1×1 卷积分解**（A1+A2，主路径；x 打包 `[N,g,C1,D,H,W,C0]`）。

### §5.3 Layer 3：关键技巧

- **T7 NC1DHW0 + 1×1 分解**（A1 算法级突破，主路径）：`Kd·Kh` 卷积拆为 `Kd·Kh` 个 shifted 1×1 dot，reduction 维降到 C0=16。x 打包 `[N, groups, C1_pg, D, H, W, C0]`；weight 打包 `[groups, Cout_pg, C1_pg, Kd, Kh, C0]`；padding 做进 x，kernel 内用 `boundary_check` 处理输出边界。
  - **`c1_inner` 循环必须用 `tl.static_range`**（普通 `range` 导致结果错误）。
  - **weight c1_inner stride = `Kd*Kh*C0`**（**非** `C1_pg*Kd*Kh*C0`；C1_pg=1 会掩盖 bug，C1_pg>1 全失败）。
- **T14 no-permute（stride==1）**：当不进入 HW-flat 门控时，可直接从 padded `[N,C,D,H,W]` 以 strides `(D_pad*H_pad*W_pad, 1)` load `(C0, W)` block，与 packed weight 做 `tl.dot([BLOCK_COUT,C0], [C0,BLOCK_W])`，跳过 `pack_input` transpose。stride>1 必须 permute 到 `[N,g,C1,D,H,W,C0]`，否则 block_ptr 非法 / gather 回退（-26%~-50%）。
- **T9 groups-batched M=16 path**（`Cout_pg ≤ 8`）：多个 groups×Cout_pg 批量进 M=16 做饱满 Cube dot；x 重打包 `[N, M_BATCH, D, H, W, C0]`，weight 重打包 block-diagonal `[M_BATCH, 16, Kd, Kh, C0]`；`M_FLAT = GROUPS_PER_M * Cout_pg` 必须作为 `tl.constexpr` kernel 参数；**gp=4 不启用**（Python packing 开销高反而退化）。
- **T15 depthwise 路径**：4D block_ptr depthwise 在该 60-case 集合上 geomean 从 0.2976x 跌至 ~0.2745x，必须回退到 scalar direct-conv fallback；用 `BLOCK_W=32/16` 让 W 单块化，`BLOCK_C=min(16, pow2_ceil(C))`，`BLOCK_D/H=4`，K 循环 `tl.static_range`。
- **T20 P3 no-permute 布局 + HW 展平 dot（stride==1 主力路径，达标 1.1332x）**：仅当 §5.2 条件启用（必须严格门控，否则 aicore timeout / 编译极慢）。
  - **输入布局 no-permute** `[N, g, C1_pg, C0, D_pad, H_pad, W_pad]`（C0 stride=`Dp·Hp·Wp`，**W stride=1** 最内层）。由 `_pack_x_nc1dhw0_grouped` 产出：`torch.zeros` 预置 padded workspace（spatial pad 区 + channel 对齐到 C0=16 的多余通道天然为零）+ `pad_copy_4d_kernel` 连续 4D block-copy（只写 interior `[BLOCK_H, BLOCK_W]`，无 F.pad / permute / slice 赋值，G6）。pack 是纯连续拷贝（非 gather），效率高于旧 C0-HW 交错布局。
  - **HW 展平 dot**：`Kw=1 且 stride==1` ⇒ `W_out==W_pad`，(kd,kh) halo 在展平 H·W 轴上是常数（`id_pad·HW + kh·dil·W_pad`，其中 `id_pad=od·stride+kd·dil`），整个 (oh,ow) tiling 折叠为 1D flat HW 块 `(od, hw_block)`。每 tap：block_ptr load `x_tile[C0, BLOCK_HW]`（`shape=(C0, HW_out)`, `strides=(DHW, 1)`, `order=(1,0)`, `boundary_check=(0,1)`）+ `w_tile[BLOCK_COUT, C0]`（wpk `[groups,Cout_pg,C1,K,K,C0]`，c0 stride=1，G8）+ `tl.dot(ieee)` 累加；末尾融合 bias + flat-HW store。
  - **关键（二者缺一不可）**：① no-permute 物理布局（pack 连续拷贝，C0 大 stride 但 W/H 连续）；② flat-HW dot（保留大 `BLOCK_HW` ≤256 连续 Cube load，tile 数少）。**仅做 no-permute 但改用分离 W 分块**（`BLOCK_W≤64`、tile 数 `Hout×` 增加、丢失大块 load）→ 0.4632x 回退（opt_iter_25 已证伪，见 §8.1）。
  - **超参**：`BLOCK_COUT` = Cout_pg≤16→16 / ≤32→32 / 否则 64；`BLOCK_HW = max(16, min(256, next_pow2(H_out·W_out)))`（`=512` 必 aicore timeout）；`hw_blocks=ceil(HW_out/BLOCK_HW)`；`total_tiles = N·groups·co_blocks·D_out·hw_blocks`；`NUM_CORES = min(total_tiles, CUBE_CORE_NUM)`（Cube dot 路径）。
  - **FULL_COUT store 分支**（跨 group co 溢出）：store 用绝对地址 `shape=(C_out, HW_out)`（截总 C_out 不截 group 边界），故仅当 `BLOCK_COUT==Cout_pg`（`FULL_COUT=True`）才 block store；否则 masked scalar store（block store 会写零污染下一 group）。masked store 不加 `tl.where` 钳位 pointer（依赖 mask 语义）。
  - **stride 不可省**：kernel 形参含 `stride: tl.constexpr`（`id_pad = od·stride + kd·dil`），但 P3 门控恒 stride==1，host **必须显式传 `1`**（旧 hwflat kernel 隐式假设 stride==1 不接 stride 参数；迁移到带 stride 的 kernel 时漏传 → `NameError: name 'stride' is not defined`，34 case 全 fail）。
  - **被取代的旧方案**：旧 P3 用 C0-HW 交错布局 `[N,g,C1,Dp,C0,HWpad]`（`pack_input_hwflat_kernel` + `conv3d_hwflat_kernel`），geomean 0.6745x；no-permute+HW-flat dot 把 pack 从 gather 改为连续拷贝且 compute 保留大块 load，P3 per-path geomean 0.7585x→1.9182x，整体 0.6745x→1.1332x 达标。
- **block_ptr store 跨 group 溢出**（groups>1 且 `C_out_per_group < BLOCK_COUT`）：`boundary_check` 只截 C_out 不截 group 边界，masked 行越过 group 边界会写零污染下一组。**解法**：constexpr 分支——满 tile（`BLOCK_COUT==C_out_per_group`）才用 block store，否则 masked scalar store；**masked scalar store 不要额外加 `tl.where` 钳位 pointer**（groups 小 Cout case 严重衰退 0.2x 级），依赖 mask 语义。
- **Pack + Padding 融合**：对标准 / HW-flat / M=16 路径，用 `torch.zeros` 初始化 packed workspace，`pack_input_kernel` 只读取原始 `x` 的有效 interior 并写到 `x_packed` 的 `[P:P+W]` 列；padding 位置天然保持零。这样省掉 `x_pad = torch.zeros(...) → copy_(x)` 的完整 memory pass。
- **T21 Pack + Padding 融合的进一步源侧迭代**：在 Pack+Padding 融合基础上，将 pack kernel 的迭代空间从 padded 输出空间 `N·groups·C1_pg·D_pad·H_pad` 改为**源侧有效区域** `N·groups·C1_pg·D·H`，目标 padded 坐标直接由源坐标推导：`pd = sd + padding`、`ph = sh + padding`（无需边界判断）。W 维因 stride 映射不可逆，仍保留目标侧 `W_out` 循环，通过 `sw = ow*stride - padding` + `sw_ok` mask 处理。效果：消除 D/H 维度的 `d_ok/h_ok` 分支与坐标钳位，grid 缩小，对 pad>0 case 有额外 3-8% 提升。
  - **P3 no-permute 路径的标准实例**：`_pack_x_nc1dhw0_grouped` + `pad_copy_4d_kernel`（见 §5.3 T20）即本技巧的 4D block-copy 版本——源侧 tile `(ng, c, sd, sh_blk, w_blk)`，目标 `out[ng, c, sd+pad, sh+pad, sw+pad]`，pad 区由 `torch.zeros` 预置。
- **T13 weight packing 缓存**（G9）：key 含完整 conv_key + variant tag；仅对计算重路径（mgrouped / NC1DHW0 / im2col weight_pad）缓存，cheap 路径缓存得不偿失。

### §5.4 性能基准与优化轨迹

- **当前最佳（应复现方案）**：opt_iter_26，60/60 pass，geomean **1.1332x**（**达标** target 0.8x），平均实现延迟 0.0597 ms，框架延迟 0.0464 ms，vs Phase 3 基线 0.0531x = **21.34×**。
- **Phase 3 baseline**：im2col + 分组 GEMM，60/60 pass，geomean **0.0531x**。
- **关键里程碑**：
  - opt_iter_1：NC1DHW0 pack + 1×1 分解（T7 主路径骨架）。
  - opt_iter_5/6/7：Pack+Padding 融合（T21）+ 各路径 packed block_ptr load 补齐 → 五路径完整（旧基线下 0.7789x）。
  - opt_iter_19（2026-08-07，0.6745x）：新验证基线下补齐 P2/P3/P4/P5 packed 路径 + 调参，target 0.8x 未达。P3 当时用 **C0-HW 交错布局** `[N,g,C1,Dp,C0,HWpad]`（已被取代）。
  - **opt_iter_26（2026-08-08，1.1332x，达标）**：P3 改 **no-permute 布局 + HW 展平 dot**（§5.3 T20），P3 per-path geomean 0.7585x→1.9182x，整体达标。
- **沿用至今的有效改动**（opt_iter_19 起，opt_iter_26 未变）：
  1. **depthwise 并入 P2**（`Cin_pg==Cout_pg==1` 走 P2 而非 P1 scalar）：6 个 depthwise case geomean ~0.28x→~0.76x；P1 门控置 `False`。
  2. **P2 `M_FLAT=32`（`Cout_pg==4`）**：`GROUPS_PER_M=8`，case 20/45 等提升。
  3. **P4 no-permute `BLOCK_COUT` 上限 128**：`Cout_pg>64` 取 128。

### §5.5 近期验证记录（2026-08-08）

#### opt_iter_26（2026-08-08，P3 改 no-permute + HW 展平 dot → 达标 1.1332x，当前最佳）

- **任务**: `8_ConvStandard3d` 60-case 多 shape，架构 ascend910B1，target_speedup=0.8。
- **结果**: 60/60 verify pass，geomean **1.1332x** vs torch（**达标** `target_reached=true`），平均实现延迟 0.0597 ms，框架延迟 0.0464 ms，vs Phase 3 基线 0.0531x = 21.34×，vs opt_iter_19 0.6745x = +68%。
- **per-path geomean**: P3 **1.9182x**（34 case，主力）/ P2 mgrouped **0.7541x**（12 case）/ P5 NC1DHW0 **0.5285x**（9 case）/ P4 no-permute **0.3315x**（5 case）。
- **变更（P3 重构）**：
  - packer：`_pack_x_nc1dhw0_grouped` → `[N,g,C1,C0,Dp,Hp,Wp]`（torch.zeros + `pad_copy_4d_kernel` 连续 4D block-copy，T21 实例）。
  - compute：`conv3d_nc1dhw0_hwflat_kernel`（读 no-permute 布局 + flat-HW dot，详见 §5.3 T20）。
  - 旧 `pack_input_hwflat_kernel` + `conv3d_hwflat_kernel`（C0-HW 交错布局）已移除。
- **最终门控**（沿用 opt_iter_19，仅 P3 内部实现替换）：
  ```python
  if False and Cin_pg == 1 and Cout_pg == 1:        # P1 disabled (depthwise 并入 P2)
      _launch_depthwise(...)
  elif Cout_pg < 16 and (16 // Cout_pg) * Cin_pg <= 16 and groups != 4 and stride == 1:
      _launch_mgrouped(...)                         # P2
  elif stride == 1 and Cin_pg <= 16 and Cout_pg <= 64 and K <= 5 and Hout >= 2:
      _launch_hwflat(...)                           # P3 = no-permute + HW-flat dot
  elif stride == 1:
      _launch_nopermute(...)                        # P4
  else:
      _launch_nc1dhw0(...)                          # P5
  ```
- **关键超参（必须按此取值才能复现 1.1332x）**：
  - P2 `M_FLAT`：`Cout_pg==4 → GROUPS_PER_M=8, M_FLAT=32`；否则 `GROUPS_PER_M=16//Cout_pg, M_FLAT=GROUPS_PER_M*Cout_pg`。
  - P3 `BLOCK_COUT`：Cout_pg≤16→16 / ≤32→32 / 否则 64；`BLOCK_HW=max(16,min(256,next_pow2(H_out·W_out)))`；`FULL_COUT=(BLOCK_COUT==Cout_pg)`；`NUM_CORES=min(total_tiles, CUBE_CORE_NUM)`；host **传 stride=1**（门控恒为 1）。
  - P4 `BLOCK_COUT`：≤16→16 / ≤32→32 / ≤64→64 / >64→128。
  - P5 `BLOCK_COUT`：≤16→16 / ≤32→32 / 否则 64。
- **迁移要点**：
  - P3 kernel 形参含 `stride`/`dilation`（tl.constexpr），但 `_launch_hwflat` 不接 stride（P3 门控恒 stride==1）→ host 须传 `1`，否则 `NameError`（34 case 全 fail）。
  - wpk 布局 `[groups,Cout_pg,C1,K,K,C0]` 复用 `pack_weight_kernel`（不变）；x 用新 packer；out 标准 `[N,Cout,Dout,Hout,Wout]`。

---
