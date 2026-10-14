---
type: Guide
title: GroupedMatmulSwigluQuant 算子（grouped-matmul-swiglu-quant） 实现手册
description: '算子类别: groupedmatmulswigluquant。'
tags: [task.implementation, topic.compute.composite, topic.compute.quantization, topic.compute.elementwise.activation, topic.implementation.memory_management, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# GroupedMatmulSwigluQuant 算子（grouped-matmul-swiglu-quant） 实现手册

> 类别约束见 [matrix_compute_constraints.md](../constraints/matrix_compute_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 GroupedMatmulSwigluQuant 算子（grouped-matmul-swiglu-quant）

**算子类别**: `grouped-matmul` + `quantization`
**典型特征**: `x` int8 `[M,K]`，`weight` 为长度 E 的 tensor list（每 expert 独立 int8 `[K,N]` + fp32 `weight_scale[e]` `[N]`），fp32 per-row `x_scale` `[M]`，`group_list` `[E]` 变长行分组（前缀和定界，允许空 expert）；计算链 = int8 grouped GEMM（int32 acc）→ 两级反量化（×ws → fp16 → ×xs → fp16）→ SwiGLU → per-token 动态量化（int8 + f32 scale 输出）
**性能基准**: 60/60 verify pass，几何平均加速比 **1.3202x** vs torch（相对 Phase 3 基线 0.7978x 提升 1.6548x；impl avg 0.1010ms；E=2~8 / M=100~4096 / K=1024~8192 / N=256~4096；终态为 opt_iter_11）

> **硬件 / 工具链绑定**: Ascend **910B2C**（24 cube / 48 vector cores）/ CANN 8.5.1。L0C 128KB 档位、UB 192KB、MTE pipe 画像均与该 SKU 绑定，跨芯片外推前必须重验。
>
> **方案演进说明**: 本算子 2026-08-12 曾归档一版 3-kernel 串行方案（gmm_dot → swiglu_fp16 → dquant，1.0283x，见 §3.6 历史方案）。终态优化（2026-08，opt_iter_11）将架构收敛为 **persistent 双 kernel**（K1 Cube gmm_dequant + K2 AIV swiglu_dquant，1.3202x）。**下文 §3.1-§3.5 为终态方案，是当前唯一有效架构；历史方案的 Layer 1 约束已被取代，仅供复盘，新实现不得按 §3.6 生成**。
>
> **与 quant 类卡 §3 / §4 的关系**: 本算子是 §4（int8 weight GEMM + 反量化）的 grouped 变长分组形态 + §3（SwiGLU + 动态量化）的尾部融合。**K2 量化段的数值链约束与 §3 完全同源**（手写 sigmoid / round-half-away-from-zero / int32 阶段 clip / vector scale），可复用；**K1 GEMM 段的调参与解耦经验不通用**——§4 的 dequant workspace 解耦 / N-strip chunking 在 grouped 变长分组下不适用（见 §3.4 失败方向表）。

### §3.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 ★★★ 禁止 per-expert 循环 launch，必须 persistent 单 launch 跨 expert 全局编号 tile
- **禁止** `for e in range(E): kernel[grid](...)` —— 每次迭代隐含同步与调度开销，speedup 随 E 单调恶化（初始实现 E=2 geo 1.104 → E=8 geo 0.472）。
- **必须** tile 跨 expert 全局连续编号，`grid = (min(cube_cores, total_tiles),)`，kernel 内 `for tile_idx in range(pid, num_tiles, NUM_CORES)`。
- **Why:** E 次 launch 的调度/同步开销在多 case 几何平均下是最大单一损失源；persistent 化实测 0.7978→1.0965（+37.4%，本算子头号杠杆）。
- **How to apply:** 见 L3.1。

#### L1.2 ★★★ 禁止逐 expert `.item()`，group_list 只做一次 D2H
- **必须** host 侧一次 `gl_cpu = group_list.cpu().tolist()`，后续全部用 CPU list。
- **禁止** kernel 启动前逐个 `group_list[e].item()`（E 次设备→主机同步）。
- **Why:** 每次 `.item()` 是一次完整 D2H 同步，与 L1.1 叠加构成 host 调度瓶颈。

#### L1.3 ★★ 跨 kernel 元数据必须单次 H2D 打包为 int64 tensor
- **必须** 把全部 per-expert 元数据（指针 delta / tile 边界 / 行边界）打包为单个 `torch.tensor(meta, dtype=torch.int64)` 一次上传，kernel 内按 constexpr 偏移切分。
- **推荐布局**: `[w_delta(E) | wsd_delta(E) | tile_lo(E+1) | row_lo(E) | row_hi(E)]`，空 expert 剔除后 `E_eff` 作为 constexpr 传入。
- ⚠️ `tile_lo` 比 E **多一个末位哨兵**（= total_tiles，供前缀扫描比较），因此其后 `row_lo` / `row_hi` 段的首地址偏移是 `3*E_eff+1` / `4*E_eff+1`——这个 **+1** 推导易错，偏移错一位全盘访存错位（精度灾难）。
- **Why:** 逐项 `torch.tensor(...).to(device)` 是 E+ 次小 H2D；meta 常驻 L2，kernel 内标量 load 开销可忽略。

#### L1.4 指针差必须按元素单位换算
- **必须** host 侧 `(ptr_a - ptr_b) // element_size()` 后再入 meta（kernel 内指针算术以**元素**为单位）。
- **Why:** int8 weight 与 fp32 weight_scale 的 element_size 不同，字节差直接传会错位访存（精度灾难，不是性能问题）。

#### L1.5 BM×BN 不得跨 L0C 32768 元素档
- **禁止** `BM×BN = 32768`（如 128×256）——L0C 档位硬失败（编译/运行错误），实证 BM≤64（BN=256 时）。
- **安全档**: 64×256 = 128×128 = 16384 元素（int32 acc 64KB）。
- **Why:** L0C 128KB 物理上限按 acc 元素数分档，16384×4B=64KB 安全，32768×4B=128KB 恰好顶满被拒。

#### L1.6 双缓冲 tile 必须核算 UB 192KB
- **必须** `2*(BM*BK + BK*BN) ≤ 192KB`（int8 输入按 1B/elt）。BLOCK_K=256、BM=64、BN=256 时 2*(16KB+64KB)=160KB 可行；BK=512 溢出。
- **Why:** 编译器自动双缓冲 A/B tile，超 UB 直接编译失败或静默拆小 tile。

#### L1.7 数值链冻结（bit 级对齐）
- **必须**保持反量化链 op 顺序：`acc(i32).to(f32) × ws → fp16 → f32 × xs → fp16`（两次 fp16 舍入点）；sigmoid 手写 `g/(1.0+tl.exp(-g))` 在 f32（同 §3 L1.4）；sw → fp16 → f32；round-half-away-from-zero（`tl.where(q>=0, q+0.5, q-0.5)`，同 §3 L1.10 思路）；clamp 在 int32 阶段 `tl.maximum/tl.minimum`（同 §3 L1.7）。
- **禁止** libdevice sigmoid 替换（硬件 sigmoid 指令舍入行为不同 → fp16 舍入点漂移）、禁止融合/移动 fp16 舍入点。
- **Why:** 60/60 全过的精度是 bit 级对齐的结果；K1 epilogue 与 K2 之间通过 GM fp16 中转固化了舍入点位置（de-mix 实验证明移动舍入点位置即使"等价"也会改变实测行为，见 §3.4）。

#### L1.8 expert 定位段保持 constexpr 标量前缀扫描，禁止向量化
- **必须** `for i in range(E_C - 1): e += (tile_idx >= bound).to(tl.int32)`（E_C constexpr → 编译期展开，int32 比较）。
- **禁止** bounds 向量 load + `tl.sum` 聚合的向量化改写（实测 0.9059 显著劣化）。
- **Why:** E≤8 次标量 load 全部 L2 命中，向量化引入 gather + reduce 反而更慢。

#### L1.9 沿用早期约束（与终态方案兼容，不单独编号冲突）
- **int8 dot 原生 int32 累加**: `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)`；`for k: acc += tl.dot(a, b)`，禁止先 `.to(f16/f32)` 再 dot（同 §3.2 语义要点 2）。
- **gmm_dot 内只乘 `weight_scale`，不乘 `x_scale`**: `x_scale` 在反量化链后段（×xs）应用，与 `npu_grouped_matmul(scale=ws_list)` 语义一致（同 L1.7）。
- **K2 dquant 不得使用 `tl.isnan` 标量调用**: `isnan` 会被 lower 为标量循环；本算子输入经 swiglu 后无 NaN/Inf，直接 `max_abs = tl.max(tl.abs(swig), axis=1)` → `row_scale = max_abs / 127.0` → `tl.maximum(row_scale, 1e-10)` → `inv_scale = 1.0 / row_scale`。

### §3.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 双 kernel 拆分（K1 Cube / K2 AIV）

```
K1 gmm_dequant_persistent_kernel（Cube, persistent 单 launch）:
  grid = min(cube_cores, total_tiles)，tile 跨 expert 全局连续编号
  for tile_idx in range(pid, num_tiles, NUM_CORES):
      e = constexpr 前缀扫描定位 expert（L1.8）
      w_ptr = w_base + w_delta[e]（元素单位）
      local_tile → (bm, bn) → m_start = row_lo[e] + bm*BLOCK_M
      k 循环: int8 tl.dot(a, b, acc_i32)（BLOCK_K by K 整除性）
      epilogue: 两级 dequant + fp16 舍入（L1.7）→ store y(fp16)

K2 swiglu_dquant_kernel（AIV, 行循环）:
  grid = min(M, 4*vec_cores)，行交织 for row_idx in range(pid, M, num_programs)
  load gate/up(fp16) → sigmoid 链（f32）→ sw → fp16 → f32
  → 行 max_abs → scale = 127/max(mabs, 1e-10)
  → round-half-away-from-zero + clamp[-128,127] → store int8 + f32 scale
```

**语义要点（文中不展开代码，但写错任一条 verify 必挂）**：

1. **K1 行界 mask 用 `row_hi[e]` 而非全局 M** —— grouped 下 m_e 非 BLOCK_M 倍数时，用 M 会越过当前 expert 读到下一 expert 的行；`x` / `xs` 的 load 与 `y` 的 store 三处 mask 同源。
2. **K1 dot 必须原生 int8 操作数 + int32 acc** —— 禁止先 `.to(f16/f32)` 再 dot（改变累加语义且损失吞吐）；仅 weight 非 int8 的备用分支（`W_IS_I32`）才转 f32。
3. **K2 gate/up 拆分方向是语义** —— gate 取前半 `[:, :HALF_N]`、up 取后半 `[:, HALF_N:]`；搞反则 sig×up 颠倒必挂。两段 load 各自按半宽 mask。
   **sw 公式 = `sigmoid(gate) × up`，不含 gate 因子** —— 标杆 `npu_swiglu(dim=-1)` 的实证口径是 σ(前半)×后半，**不是**通识 SwiGLU 的 silu 变体 `gate·σ(gate)·up`；多乘 gate 会 60/60 全挂（sw 分布变 → per-row scale 全变，|diff|≤1 容差下 81% 元素超限、max_abs_diff=127 满量程）。2026-08-22 从零生成首版实测踩中。
4. **K2 BLOCK_N 按 `next_power_of_2(HALF_N)` 取**，不沿用 K1 的 128/256 —— HALF_N 可达 2048，行内单 tile 覆盖全半宽。
5. **out_s 存的是 `max_abs / 127`（scale 的倒数）**，不是 scale 本身 —— 与标杆反量化约定一致，见 L3.3。
6. **dequant 广播方向** —— `ws [N]` 按 N 列广播、`xs [M]` 按 M 行广播，方向写反是维度错误。

#### L2.2 Host 侧 BLOCK 门控分派表

| 条件 | BLOCK_M | BLOCK_N | BLOCK_K | 说明 |
|------|---------|---------|---------|------|
| N % 256 == 0（默认大 N） | 64 / 32 / 16 by avg_m≥48/32 | **256** | 256 / 128 / 64 by K%256/%128 | 主路径，UB 160KB |
| N % 256 != 0 | 64 / 32 / 16 by avg_m≥48/32 | 128 | 同上 | 次路径 |

- `avg_m = M // E`（行均摊）；BLOCK_M 按 avg_m 分档避免小 expert padding 浪费；BM=64 需 `E*ceil(avg_m/64)*num_tiles_n ≥ cube_cores`（24 core 满负载守卫，小 N shape 退回 BM=32 防并行度损失）。
- ⛔ 已证伪：`avg_m≥128 时 BM=128/BN=128` 方形 tile（见 §3.4 失败表）。

#### L2.3 元数据构建（host，每 forward 一次）

1. `gl_cpu = group_list.cpu().tolist()`（唯一 D2H，L1.2）；
2. 遍历 E 个 expert：空 expert（m_e==0）剔除，非空的累积 `w_delta`（元素单位，L1.4）/ `wsd_delta` / `row_lo` / `row_hi` / `tile_lo`（`tiles_e = ceil(m_e/BM) * num_tiles_n`）；
3. `meta = w_deltas + wsd_deltas + tile_lo + row_lo_l + row_hi_l` → 单 tensor H2D（L1.3）。

### §3.3 Layer 3: 关键技巧（Agent 可参考但不可复制代码结构）

#### L3.1 ★★★ persistent tile 循环 + constexpr 前缀扫描 expert 定位

```python
# E_C: tl.constexpr（剔空后的 expert 数）→ 编译期完全展开，int32 比较
for tile_idx in range(pid, num_tiles, NUM_CORES):
    e = 0
    for i in range(E_C - 1):
        bound = tl.load(tile_lo_ptr + i + 1).to(tl.int32)
        e += (tile_idx >= bound).to(tl.int32)
    w_ptr = w_base_ptr + tl.load(w_delta_ptr + e)   # 元素单位 delta
    lo_e = tl.load(tile_lo_ptr + e).to(tl.int32)
    local_tile = tile_idx - lo_e
    bm = local_tile // num_tiles_n
    m_start = tl.load(row_lo_ptr + e).to(tl.int32) + bm * BLOCK_M
```

**可替代方向**: 按行分组（row_hi 定界）也可定位，但 tile 粒度才能配合 BLOCK_M 分档；向量化前缀扫描已证伪（L1.8）。

#### L3.2 meta 打包 + 单次 H2D / 单次 D2H

```python
gl_cpu = group_list.cpu().tolist()        # 唯一一次同步（替代 E 次 .item()）
meta_t = torch.tensor(
    w_deltas + wsd_deltas + tile_lo + row_lo_l + row_hi_l,
    dtype=torch.int64, device=device)     # 唯一一次小 H2D
```

**注意**: M==0 边界（全部 expert 空）时 meta_t 仍要构建（K2 行循环零迭代，但 kernel 签名需要指针）。

#### L3.3 K2 标量 scale store 的 broadcast trick

**问题**: 每行输出一个 f32 scale，标量 `tl.store` 会降级。

**解决**:
```python
r1 = tl.arange(0, 1)
tl.store(out_s_ptr + row_idx + r1, (max_abs / 127.0) + r1 * 0.0)
```

**可替代方向**: 把 scale 打包进 int8 输出尾部的融合布局——会改变输出语义，禁止。

### §3.4 性能基准与优化尝试记录

| 版本 | cases | geomean | 相对上版 | 说明 |
|------|-------|---------|---------|------|
| Phase 3 基线（per-expert launch + 逐次 .item()） | 60/60 | 0.7978x | — | 双 kernel，E 次同步调度 |
| **opt_iter_0（#12 persistent + meta 打包）** | 60/60 | **1.0965x** | +37.4% | 头号杠杆（L1.1-L1.3） |
| opt_iter_1（#1 constexpr mask 折叠） | 60/60 | 1.0647x | ↓ | 无增量，回退 |
| **opt_iter_2（#2 BLOCK_K=256）** | 60/60 | **1.1516x** | +5.0% | 当时 best（L1.6 UB 核算） |
| opt_iter_3（#5 expert 定位向量化） | 60/60 | 0.9059x | ↓↓ | 回退（L1.8 证伪依据） |
| opt_iter_4（#13 autotune） | 60/60 | 1.0840x | ↓ | 回退 |
| opt_iter_5（#29 K1/K2 CV 融合） | 60/60 | 0.5596x | ↓↓↓ | 回退 |
| opt_iter_6（#31 de-mix：int32 acc 直存，dequant 挪 K2） | 60/60 | 1.0495x | ↓ | bit 级等价但劣化，见失败表 |
| opt_iter_7_ir_2（#31 IR 轮 2：K2 分析） | — | — | — | 无新建议，IR 耗尽 |
| opt_iter_8（#2 变体 BM=128/BN=128 方形 tile） | 60/60 | 0.9301x | ↓ | 追加轮证伪，见失败表 |
| **opt_iter_9（#11 Load 重排序：weight load 先于 x load 发射）** | 60/60 | **1.2235x** | ↑ | 掩盖 L1→L0 2D transpose 慢搬运链延迟，搬运瓶颈 case 38 0.529→0.605 |
| opt_iter_10（框架外 weight-stationary N_M_SUB=2 双 kernel） | 60/60 | 0.8604x | ↓ | 修复 constexpr 假分支竞态后正确，但 B 搬运无实际节省 + int8 fractal 利用下降，见失败表 |
| **opt_iter_11（#2 下探 BLOCK_M=64 到 avg_m≥48 带）** | 60/60 | **1.3202x** | ↑ | **终态 best**：tile 数≥cube_cores 守卫下减半中型 shape weight 搬运量（case 48/46/45/58/39 -9~36%），0 回归超 10% |
| opt_iter_12（框架外 tile 顺序 m_group 主序 → weight L2 复用） | 60/60 | 0.9770x | ↓ | 破坏 x 复用 + weight 超 L2 不命中 + MTE2 ND2NZ 粒度下降，见失败表 |
| opt_iter_13（KKB exp_perf_012：存活 mask i32→fp32 比较转换） | 60/60 | 不可用 | ↓↓↓ | K1 fp32 行 mask aicore 挂死；K2 fp32 half mask 非 2 幂 case 14× 变慢，见失败表 |

**有效优化点**:
1. **persistent 单 launch + meta int64 打包单次 H2D + group_list 单次 D2H**（0.7978→1.0965，+37.4%）
2. **BLOCK_K=256 tiling**（K%256==0 时 k 循环迭代数减半，→1.1516，+5.0%）
3. K2 标量 scale store broadcast trick（L3.3，随主版本带入）
4. **#11 Load 重排序（weight load 提前发射）**（→1.2235，+6.2%）：weight 搬运链是唯一主导 pipe（MTE1+MTE2=68.1%），把慢搬运 load 提到 x load 前发射掩盖延迟，数值 bit 级等价
5. **#2 下探 BLOCK_M=64（avg_m≥48 带 + tile 数≥cube_cores 守卫）**（→1.3202，+7.9%）：减半中型 shape 的 weight 搬运量；守卫防小 N shape 并行度损失

**已验证无效/失败的优化方向**:

| 优化方向 | 结果 | 原因 |
|----------|------|------|
| constexpr mask 折叠（#1） | 1.0647x ↓ | mask 已被编译器处理，手动折叠无增量 |
| expert 定位向量化（bounds gather + tl.sum） | 0.9059x ↓↓ | E≤8 标量 load L2 命中；向量化引入 gather+reduce 更慢 |
| autotune（#13） | 1.0840x ↓ | 编译/搜索开销吃掉收益，60 case 多 shape 场景尤甚 |
| K1/K2 CV 融合单 kernel（#29） | 0.5596x ↓↓↓ | Cube persistent tile 循环与 AIV 行循环粒度不匹配，互相拖累 |
| K1 de-mix（int32 acc 直存去 lock-step，dequant 链挪 K2，#31） | 1.0495x ↓ | bit 级等价仍劣化：IR 上 mix lock-step（每 tile 两次跨核握手 + 64KB workspace）真实代价仅 FLOWCTRL 2.3%（simulator 证实），编译器 mix 流水已有效重叠 AIC/AIV；而 K2 读 int32(4B/elt) 比读 fp16(2B/elt) 带宽翻倍 + 新增 dequant 计算，大 case 劣化 5~17%。**IR 结构性瓶颈必须配分段流量账核算（GM 总量账 ≠ 分段带宽账）** |
| BM=128/BN=128 方形 tile（avg_m≥128 门控） | 0.9301x ↓ | 纸面装载/计算比 -20% + weight 流量减半，实测 58/60 case 劣化：int8 fractal 装载下 BN 256→128 使 B 的 N 向装载趟数翻倍（碎片/启动开销升），ND2NZ 大块搬运更高效，avg_m≥128 不保证每 expert m_e≥128（padding 浪费）。**纸面矩形 tile 推演必须经 fractal 对齐与 padding 校正** |
| BM=128/BN=256（增大 tile） | 编译硬失败 | L0C 32768 元素档溢出（L1.5） |
| BLOCK_K=512 | 编译硬失败 | UB 192KB 溢出（L1.6） |
| weight-stationary（M 方向 weight 复用，N_M_SUB=2 双 kernel） | 0.8604x ↓↓ | 修复 constexpr 假分支竞态后正确，但 M 复用权重搬运无实际节省 + 每 program 变稀疏的 int8 fractal 装载效率下降，B 搬运仍是瓶颈 |
| tile 遍历 m_group 主序（相邻 program 共享 weight 走 L2） | 0.9770x ↓ | 破坏原 n_block 主序的 x 复用（x 流量与 weight 同量级）；weight 470MB 远超 L2，grid-stride 使 program 跨 bn 处理不连续列 → MTE2 ND2NZ 大块转换效率下降（同 opt_iter_8 BN 粒度教训） |
| **i32→fp32 存活 mask 转换（KKB exp_perf_012，Cube row_hi mask）** | aicore 挂死 | triton_ascend 后端对 fp32 派生 load/store mask 生成损坏 MTE 任务描述符：K1 Cube kernel 永不完成（507014 watchdog），进程死后 aicore 仍 99% 空转卡死整块 device |
| **i32→fp32 存活 mask 转换（KKB exp_perf_012，AIV half-width mask）** | 14× 变慢 | K2 非 2 幂 HALF_N case（13/60）speedup 塌到 0.04-0.07（K2 单次 forward ~4s）；i32 比较虽"标量降级"却是后端处理良好的路径。结论：fp32 mask 转换在本算子双向失效，勿在 mask 上使用 exp_perf_012 |

**关键结论**:
1. **host 调度是 grouped 变长分组的第一瓶颈**：per-expert launch + 逐次 D2H 使 speedup 随 E 单调恶化（E=8 时 0.472x）；persistent + meta 打包一步修复（+37.4%）。
2. **修复后瓶颈立即转移到 weight 搬运链**：simulator 实测（真实配置 BN256）K1 Cube 核 **MTE1 44.2%（L1→L0 2D transpose）+ MTE2 23.9%（GM→L1 ND2NZ）= 68.1%**，MMAD 仅 6.4%（三组 shape 稳定 5.5~6.4%）。非计算 bound，但 #7/#21/#10 核对不适用（单 pass GEMM / 无重复 gather / 无循环不变量），增大 tile 与 bf16 化不可行 → 属 int8 grouped GEMM 结构性搬运代价。**终态 1.3202x（opt_iter_11）已达理论上限 1.3995 的 94.4%**；向量化/autotune/融合/de-mix/方形 tile/weight-stationary/tile 序/fp32 mask 全部证伪。**后续同类算子不要按"计算/调度"方向继续硬攻**。
3. **IR 结构瓶颈必须配分段流量账**：见失败表 de-mix 行。simulator 小 shape 会高估 SCALAR / 低估搬运（K64 时 SCALAR 60.2%，K1024 真实配置下 21%）——诊断必须用接近线上的 shape + BLOCK 配置扫描。
4. **§4 的解耦/N-strip 经验对本算子不适用**：dequant 已在 K1 epilogue 单次执行（无跨 M-tile 重复 dequant 问题）；workspace/N-strip 与 expert 变长分组边界冲突，K1/K2 融合类方向全部实测劣化。
5. **benchmark 环境污染判定法（共享 NPU 服务器必读）**：framework（sha256 冻结的 PyTorch 参考）延迟漂移 >50% 即为争用铁证（framework 代码不变，波动只能来自环境）；impl 绝对延迟 per-case 对比不受 framework 噪声影响，可作交叉验证。被污染的 run 必须重测，不可用修正值充当官方结果。
6. **数值链 bit 级对齐是 60/60 的前提**：两级 fp16 舍入点 + 手写 sigmoid + round-half-away-from-zero + int32 clip（与 §3 同源）；任何"等价改写"（libdevice sigmoid、移动舍入点）都会破坏对齐。
7. **fp32 mask 转换（exp_perf_012）在本算子双向失效**：Cube load/store mask 转 fp32 → aicore 挂死（损坏 MTE 任务描述符，kernel 永不完成，507014 watchdog，进程死后 aicore 仍 99% 空转卡死整块 device）；AIV mask 转 fp32 → 非 2 幂 case 14× 变慢（speedup 0.04-0.07）。i32 比较虽"标量降级"却是后端处理良好的路径。**该 KKB 技术仅适用于纯向量算术路径，不适用于 mask 生成**。
8. **挂死 kernel 的隔离方法论**：verify 60/60 全灭且报 `aicore timeout 507014` 时，先隔离 impl 单 kernel 复现（impl forward → framework forward → sync 的 verify 忠实驱动）；挂死是 Cube 还是 AIV 用单点变体二分（`iso/k1only.py` / `iso/k2only.py`）。挂死会污染 device 整块卡死（进程死但 aicore 空转），**必须换空闲 device（`ASCEND_RT_VISIBLE_DEVICES=<id>`，verify.py 固定 `torch.device("npu")` 取可见首卡）重测**，否则后续所有 benchmark 全灭误判为代码 bug。

### §3.5 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| per-expert 循环 launch / 逐 expert `.item()` | E 次隐式同步与调度，speedup 随 E 单调恶化（E=8 时 0.472x） | persistent 单 launch + group_list 单次 `.cpu().tolist()` + meta 单次 H2D (L1.1-L1.3) |
| 指针差按字节传 meta | kernel 内指针算术以元素为单位，int8 与 fp32 element_size 不同 | host 侧 `(ptr_a - ptr_b) // element_size()` 换算 (L1.4) |
| meta 段偏移漏掉 tile_lo 的 +1 | tile_lo 比 E 多一个末位哨兵，`row_lo`/`row_hi` 段首偏移是 `3E+1`/`4E+1` 而非 `3E`/`4E` | 按 L1.3 布局字符串逐段推偏移，错一位全盘访存错位 |
| K1 行界 mask 误用全局 M | grouped 下尾部 tile 会越过当前 expert 读到下一 expert 的行 | mask 用 `row_hi[e]`，x/xs load 与 y store 三处同源 (§3.2 语义要点 1) |
| K2 gate/up 搞反（前半当 up） | 拆分方向是算子语义，sig×up 颠倒后数值全错 | gate=前半、up=后半，各自按半宽 mask (§3.2 语义要点 3) |
| sw 写成通识 SwiGLU `gate·σ(gate)·up`（silu 变体） | 标杆 `npu_swiglu` 实证口径是 σ(gate)×up，多乘 gate 使 sw 分布与 per-row scale 全变 | 公式冻结为 `sigmoid(gate) × up`（无 gate 因子），60 case 无一幸免、max_abs_diff=127 (§3.2 语义要点 3) |
| K2 BLOCK_N 沿用 K1 的 128/256 | HALF_N 可达 2048，固定小块导致行内多趟或 mask 越界 | `next_power_of_2(HALF_N)` 单 tile 覆盖全半宽 (§3.2 语义要点 4) |
| BM×BN 顶到 32768 档（如 128×256） | L0C 128KB 按元素数分档，32768×4B 恰好顶满被拒 | 保持 16384 档内（64×256 或 128×128 acc 均为 64KB）(L1.5) |
| BK=512 双缓冲溢出 | 2*(BM*BK+BK*BN) 超 192KB UB | BLOCK_K≤256 并核算 UB 预算 (L1.6) |
| expert 定位段向量化 | E≤8 次标量 load 全 L2 命中，向量化引入 gather+reduce 反而慢（0.9059x） | 保持 constexpr 标量前缀扫描 (L1.8) |
| K1/K2 融合或 de-mix 改写 | Cube tile 循环与 AIV 行循环粒度不匹配；mix lock-step 真实代价仅 2.3%，去 mix 省的同步抵不过中间缓冲 dtype 翻倍的带宽 | 保持双 kernel + fp16 GM 中转；IR 结构改动前先算分段流量账 (§3.4) |
| BM=128/BN=128 方形 tile（纸面装载比更优） | int8 fractal 装载下 BN 减半使 B 装载趟数翻倍；avg_m 门控挡不住 m_e 不均的 padding 浪费（0.9301x） | 保持 BN=256 宽 tile + BM≤64 分档 (L2.2) |
| libdevice sigmoid / 移动 fp16 舍入点 | 硬件 sigmoid 舍入行为不同；舍入点位置变化破坏 bit 级对齐 | 数值链冻结 (L1.7，同 §3) |
| 把共享服务器 benchmark 的病态 case 当代码 bug | 环境争用使 1-5 个 case 病态（framework 也同步漂移） | framework（冻结参考）漂移 >50% 即判污染，重测；impl 绝对延迟交叉验证 (§3.4 结论 5) |
| 存活 mask 转 fp32（i32→fp32 比较） | Cube mask aicore 挂死 / AIV mask 14× 变慢 | mask 保持 i32（后端良好路径）；exp_perf_012 仅限纯向量算术 (§3.4 结论 7) |
| device 挂死后继续在同一 device 上 benchmark | 挂死 kernel 污染整块 device（aicore 99% 空转），后续全灭误判代码 bug | 换空闲 device（`ASCEND_RT_VISIBLE_DEVICES=<id>`）重测 (§3.4 结论 8) |

### §3.6 历史方案（早期 3-kernel 串行，已被 §3.1-§3.5 取代，仅供复盘）

> 2026-08-12 归档的初版方案（1.0283x）。终态 persistent 双 kernel 已取代其架构；此处保留原约束与记录以免信息丢失。**新实现请勿按本节生成，仅作方向复盘参考**。

**算子类别**: `grouped-matmul-swiglu-quant`
**典型特征**: 分组 int8 矩阵乘 + SwiGLU 激活 + per-row 动态量化到 int8
**性能基准**: 60/60 verify pass，几何平均 **1.0283x** vs torch；相对 Phase 3 基线 0.9102x 提升 **+12.95%**

> 与纯 SwigluQuant 不同，本算子**主要瓶颈在 Grouped MatMul cube 利用率**，其次才是 dquant 的 vector 效率。

#### L1.1 必须保持 3-kernel 结构

**禁止**融合 SwiGLU 与 dquant。即使按原 `swiglu_fp16_kernel` 做 fp16 rtne cast 模拟，kernel 内浮点执行顺序/规约路径与分步实现仍存在系统性差异，导致 out_scale 的 MERE 超 fp32 阈值（实测 0/60 fail）。

正确流程：
```
gmm_dot → mm_out (fp16)
swiglu_fp16 → swig_tmp (fp16)
dquant → out_q (int8) + out_s (fp32)
```

#### L1.2 禁止在 gmm_dot 中应用 `x_scale`

- 只乘 `weight_scale`，不乘 `x_scale`，与 `npu_grouped_matmul(scale=ws_list)` 语义一致。

#### L1.3 int8 matmul 必须使用 int32 累加

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
for k in range(0, K, BLOCK_K):
    acc += tl.dot(a, b)
```

#### L1.4 dquant 不得使用 `tl.isnan` 标量调用

- `isnan` 会被 lower 为标量循环，显著拖慢 dquant。本算子输入经 swiglu 后无 NaN/Inf，直接：
  ```python
  max_abs = tl.max(tl.abs(swig), axis=1)
  row_scale = max_abs / 127.0
  row_scale = tl.maximum(row_scale, 1e-10)
  inv_scale = 1.0 / row_scale
  ```
- 仅此改动 geomean +5.4%。

#### L1.5 BLOCK_N=512 必须配合 BLOCK_K=256 及以下

- `BLOCK_M=48, BLOCK_K=512, BLOCK_N=512` 在 K≥7168 时直接报 `cbuf overflow`；必须降 BLOCK_K。

#### L2.1 Host 侧调度

1. grouped matmul 按 expert 拆分串行调用 `gmm_dot_kernel`。
2. `swiglu_fp16_kernel` 读 `mm_out` 写 `swig_tmp`。
3. dquant 混合策略：
   ```python
   if HALF_N <= 1280:
       # 直接用 HALF_N 作 BLOCK_N，不要用 next_power_of_2
       swiglu_dquant_single_kernel(..., BLOCK_N=HALF_N)
   else:
       swiglu_dquant_kernel(..., BLOCK_N=512)
   ```

#### L2.2 Matmul Tiling

| N 范围 | BLOCK_M | BLOCK_K | BLOCK_N | 说明 |
|--------|---------|---------|---------|------|
| N ≥ 4096 | 64/32 | 256 | 512 | 必须降 BLOCK_K 防 UB 溢出 |
| 2048 ≤ N < 4096 | 64/32 | 512 | 256 | 基线参数，最稳定 |
| N < 2048 | 48/32 | 512 | 128 | 小 N 场景 |

> 对 60 cases，固定 `BLOCK_M=48, BLOCK_K=512, BLOCK_N=256` 已达 0.9102x；动态 tiling 收益有限且易劣化，首次生成建议以基线参数为主。

#### L3.1 扩展 single-pass dquant 阈值到 HALF_N ≤ 1280

- 原阈值 `HALF_N <= 1024` 扩展到 `HALF_N <= 1280`，geomean 从 0.9877x → **1.0283x**（+4.1%）。
- 用 `BLOCK_N=HALF_N`，**不要** `next_power_of_2`（1280 会变成 2048，触发 UB overflow）。

#### L3.2 Host 缓存 driver/stream

见 §1 G3，实测 geomean +3%。

#### L3.3 消除 dquant NaN/Inf 标量链

见 §3.1 L1.4，实测 geomean +5.4%。

#### 历史优化尝试记录

| 优化点 | 结果 | 原因 |
|--------|------|------|
| 融合 swiglu + dquant | ❌ 0/60 精度失败 | kernel 内数值链与分步实现存在系统性差异 |
| 动态 GMM tiling（BLOCK_N=512, BLOCK_K=512） | ❌ 9/60 编译失败 | UB/L0 cbuf overflow |
| 动态 GMM tiling（BLOCK_N=512, BLOCK_K=256） | ⚠️ 0.9293x | 慢 case 提升但大量快 case 下降，整体劣化 |
| Host 缓存 | ✅ +3% | 减少小 shape enqueue 开销 |
| 消除 dquant NaN/Inf 链 | ✅ +5.4% | 移除标量循环 |
| single-pass 阈值 1024→1280 | ✅ +4.1% | 减少一次 HBM 读 |

#### 历史关键结论

1. **3-kernel 结构是精度基础**：SwiGLU 与 dquant 不可融合。
2. **dquant 的 NaN/Inf 标量链是主要性能陷阱**：参考实现无异常处理，直接移除。
3. **single-pass 阈值扩展是跨过 1.0x 的关键**：1024→1280，但需用 `BLOCK_N=HALF_N` 而非 `next_power_of_2`。
4. **GMM tiling 谨慎调整**：基线 `(48,512,256)` 已很稳，激进改大 BLOCK 易 UB overflow 或整体劣化。
5. **Host 缓存有小但稳定收益**：尤其在多 expert、小 shape 场景。

---
