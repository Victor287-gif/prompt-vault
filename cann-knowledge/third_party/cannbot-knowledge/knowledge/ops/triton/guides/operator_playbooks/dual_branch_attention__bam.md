---
type: Guide
title: BAM 算子（bam） 实现手册
description: '典型特征: BAM：通道 GAP→MLP(C→C/r→C)→sigmoid 支与空间 k×k 膨胀 conv 支加性门控 out=x·(1+sigmoid(ch+sp))，k×k
  conv 走 shifted-1×1 dot（Cube 路径）。'
tags: [task.implementation, topic.attention, topic.compute.neighborhood.convolution, topic.compute.reduction, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:23:23Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# BAM 算子（bam） 实现手册

> 类别约束见 [../dual_branch_attention_constraints.md](../constraints/dual_branch_attention_constraints.md)，
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §1 BAM 算子（bam）

**算子类别**: `dual_branch_attention`　**典型特征**: BAM：通道 GAP→MLP(C→C/r→C)→sigmoid 支与空间 k×k 膨胀 conv 支加性门控 `out = x·(1+sigmoid(ch+sp))`；k×k conv 走 shifted-1×1 dot（Cube 路径）。　**性能基准**: 50/50 verify，几何平均加速比 **2.9430**（target 2.0 达标；ascend910_93 / NPU4）

### Layer 1 设计约束（算子专属）

全部继承类别卡 L1.1~L1.12（见 [../dual_branch_attention_constraints.md](../constraints/dual_branch_attention_constraints.md)）。本算子特判：

- **L1.2 命中 Cube 路径**：k×k 膨胀 conv（cr→cr，真实跨通道 GEMM）→ shifted-1×1 GEMM + `tl.dot`；
- **通道支 MLP** 是 `[B,C]→[B,C]` 小 GEMV，走 vector（M=16 pad，b≤4 冗余重算 gemm1）；
- L1.8/L1.9 的 conv 平面化、halo_zero 三段式在本算子有完整实锤（§3）。

### Layer 2 算法骨架（8 kernel 管线 + 平面化变体）

```text
通道支: K_gap（per-(b,cblk) 行块 GAP）→ K_mlp（双 GEMM，M=16 pad，b≤4 冗余重算 gemm1）
空间支: K_conv1（1×1, c→cr，写 ws1 interior）→ K_conv2a/b（k×k dilated, cr→cr，
        halo workspace 串联 ws1→ws2）→ K_conv3（cr→1 向量归约）→ K_pool（adaptive，仅 Hout≠h 时启动）
融合:   K_combine: out = x·(1+sigmoid(ch[b,c] + sp[b,hw]))（广播逐元素）
```

- **zeros workspace halo**：`ws[b,cr,h+2d,w+2d]` 一次清零，interior 由上一级 conv 写入，
  同时是下一级的 padded 输入——pad 语义由 halo 零承担，**禁 host 侧 F.pad/unfold/切片赋值**。
- **safe-adjust**：`eff = dil·(k-1)+1 > min(h,w)` 时收缩 (k,dil)——host 逐行复刻参考实现。
- 平面化变体（Phase 4 主收益，§3）：k=3 帧同尺寸时 `halo_zero → plane → compact` 三件套；
  k=1 恒等平面 + `_expand_kernel` 逆变换（§3.2）。

### Layer 3 关键技巧（可参考不可照抄）

#### §3.1 ★ conv 帧平面化（主收益，2.17 → 2.94）

- HOUT 行循环并入 dot 的 N 维：tile 数 `b·co_blocks·HOUT → b·co_blocks·⌈FP/BN⌉`
  （idx15 实测 84 → 4，dot 调用 756 → 36）。
- tap 输入列 = `pn + t_off` **纯加法常量偏移，无 div/mod**；输出 halo 列写 garbage，
  下游读侧 `IN_OFF/IN_CORE` constexpr 屏蔽。
- 前置：帧平面含 halo 断带——**断带转换不可消除，只能集中到最便宜的纯搬运环节**（compact/expand）。

#### §3.2 halo_zero 三段式纯连续（44x 提升）

```python
# 段选择 = 每 tile 一次标量 3 路分支，段内整块同构 mask（绝不逐元素 tl.where）
seg = (py < d) - (py >= HOUT + d)          # -1: top 带, 0: 核心行, 1: bottom 带
if seg != 0:    tl.store(top_bot_2d_block, 0.0)      # 整带全行连续 2D store
else:           tl.store(core_row_side_steps, 0.0)    # 核心行两侧步进连续段
```

- 行循环版 206-445µs（每行 2 个微型 store 摧毁 MTE 流水，~0.4-1.5µs/行）；
  2D `tl.where` 混合 mask 版 89-143µs（混合 mask 使 store 走逐元素 scatter 慢路 ~0.8GB/s）；
  **三段式** 4.7-6.4µs。
- 通则：**清零/搬运类 kernel 的 mask 必须整段同构**，混合 mask 必然走 scatter 慢路。

#### §3.3 k=1 恒等平面 + expand 逆变换（k=1 占 10/50 case）

- k=1 conv = 逐像素 GEMM：恒等 flat 无偏移映射（OFF0=0）+ 双侧 `pn < FP` mask，padding lane 天然安全；
- 偏移/断带由独立 `_expand_kernel`（compact 逆变换）承担——把最容易错的偏移逻辑隔离在最便宜的纯搬运里。

#### §3.4 分核 BN 收缩（case 级 +5~7%）

- 大帧（FP≥1024）欠并行时 BN 减半提升 grid 占用，仅 BM>16（cap=256）场景启用——
  BM=16/cap=512 时宽 tile 的 MTE 效率优于并行度，**收缩不是万能**。

### 性能基准 / 优化记录表

| 指标 | 值 |
|------|-----|
| 算子 | 36_BAM |
| 架构 / NPU | ascend910_93 / NPU4 |
| geomean | **2.9430**（target 2.0 达标；conv 平面化四步 2.1697 → 2.9430） |
| verify | **50/50** |
| 轮次 | Phase 3×1 + Phase 4×6 |
| 关键路径 | conv 平面化三件套 + k=1 恒等平面 + BN 收缩 |

### 陷阱 / 证伪方向

- **清零 kernel 极慢（百 µs 级）**：行循环微型 store / 混合 mask → §3.2 三段式纯连续。
- **标量 bool 与向量 mask 组合再触发 507015**：谓词组合编译缺陷（conv2b `kmask & row_ok & col_ok`）→
  行判定改标量 `if row_in:` 直接跳过（顺带省 tap load）。
- **平面化后下游读入 garbage**：输出 halo 列未清 → halo_zero 三段式 / `IN_CORE` constexpr 屏蔽。
- 证伪：block_ptr 全量替换 masked load **-24~-29%**（小 tile 下固定开销超 gather 节省）；
  safe-adjust 边界逐行复刻参考实现，不得自行近似。
