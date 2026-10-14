---
type: Guide
title: TripletAttention 算子（triplet_attention） 实现手册
description: '典型特征: CH/CW/HW 三分支两两交互 zpool(mean/max)→门控 conv→BN→sigmoid，out=(o1+o2+o3)/3，permute
  全消解为索引映射。'
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

# TripletAttention 算子（triplet_attention） 实现手册

> 类别约束见 [../dual_branch_attention_constraints.md](../constraints/dual_branch_attention_constraints.md)，
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §1 TripletAttention 算子（triplet_attention）

**算子类别**: `dual_branch_attention`　**典型特征**: CH/CW/HW 三分支两两交互：zpool(mean/max) → 门控 conv → BN → sigmoid，`out = (o1+o2+o3)/3`；permute 全消解为索引映射（forward 零 host 布局操作）。　**性能基准**: 50/50 verify，几何平均 **2.6433**（ascend910_93 / NPU13）

### Layer 1 设计约束（算子专属）

全部继承类别卡 L1.1~L1.12（见 [../dual_branch_attention_constraints.md](../constraints/dual_branch_attention_constraints.md)）。本算子特判：

- **L1.2 → 全 vector 路径禁 `tl.dot`**：门控 conv C_in=2、C_out=1，用 Cube 16×16 阵列浪费 15/16；
- L1.10 / L1.11 / L1.12 在本算子均有实测实锤（store 行内仿射 / max `other=-inf` / z 平面 plane0=mean、plane1=max）。

### Layer 2 算法骨架（8 kernel / 13 launch）

```text
三分支 permute 全部消解为索引映射（forward 零 host 布局操作）：
  CH 分支归约 W（stride=1）｜CW 分支归约 H（stride=W）｜HW 分支归约 C（stride=H·W）
K1 zp_outer ×2（cw,hw: 二维 tile 双平面 mean/max）+ K2 zp_rowreduce ×1（ch: 行矩阵 [C·H, W] block_ptr）
→ K3 gate_conv ×3（tap static_range，权重标量加载 2·K²≤98，读 zp padded 区）
→ K4 bn_stats ×3（grid=(1,) 双趟）→ K5 ap_plane / K6 ap_row / K7 ap_scalar（按 attn 形状选：行向量/逐 h 向量/逐 h 标量）
→ K8 combine3（(o_ch+o_cw+o_hw)·(1/3)）
依赖: {K1,K2} → K3 → K4 → {K5,K6,K7} → K8（分支间无依赖，launch 顺序无关）
```

- 每分支 (D1,D2,R,S,S1) 参数化：ch=(C,H,W,1,W)、cw=(C,W,H,W,HW)、hw=(H,W,C,HW,W)。
- K=kernel_size ∈{3,5,7} 由 `tl.constexpr` 特化，单一架构覆盖全部 case（无 host 门控分派）。

### Layer 3 关键技巧（可参考不可照抄）

#### §3.1 ★★ store 行内仿射化（IR 轮，+13.0%，全算子最大单点）

```python
# ❌ dst 含整除/取模，跨行跳跃非仿射 → IR 判 DiscreteMemAccess → scf.for 逐元素标量 store
dst = (rows // H + PAD) * D2P + (rows % H + PAD)
tl.store(zp_ptr + dst, val)                       # 每迭代 1 float + 4 次 wait/set_flag
# ✅ 重构 program=(n, c-block) 逐 c 归约（RPB=2 行 static 展开），dst 行内仿射连续
dst = (c + PAD) * D2P + hv + PAD                  # c 为 static_range 循环变量
tl.store(zp_ptr + dst, val)                       # 1D 向量 store
```

- IR 证据：`last_pass.mlir` 中 `scf.for` 标量循环 + `ExtractedLoadOrStore` + `DiscreteMemAccess`，
  每迭代 UB 标量 load + `store_ubuf_to_gm_1d`（1 float）+ 4 次 wait/set_flag——64 值 × 2 通道全部标量 store。
- 修法落点两处（rowreduce / creduce），合计 **+13.0%**。
- **方法论**：Phase 4 常规点耗尽后 dump IR 搜 `DiscreteMemAccess`——凡 dst 含 `//` `%` 跨行跳跃的
  store 都是嫌疑；源码层面看不出来（写法合法，编译后才知道退化）。

#### §3.2 tap 分组两阶段（+6.5%）

- `total < 16`（欠并行）时 `TG = min(2k, max(2, cdiv(40, total)))` 自适应分组：
  小 B 小 C 的门控 conv 从 ≤16 program 补足 ~40 program；kernel 体零改动，模块级统一三分支分派。
- tap 阈值 40（放宽分组门控）证伪：-1.7%。

#### §3.3 宽度自适应 `_bw()` 8 档（+2.9%）

- 小维度（W/H ≤ 23）按 8 档宽度函数选 BLOCK，替代定宽 32；配 `BLOCK_H = next_pow2(H)`。

#### §3.4 host 缓冲复用（+4.5%）

- zp 的 pad 环零填充一次（workspace 缓存），每轮只写 interior——省掉每轮整张 zeros 重分配重清零。

### 性能基准 / 优化记录表

| 指标 | 值 |
|------|-----|
| 算子 | 64_TripletAttention |
| 架构 / NPU | ascend910_93 / NPU13 |
| geomean | **2.6433**（轨迹 1.0580 → 2.6433） |
| verify | **50/50** |
| 轮次 | Phase 3×1 + Phase 4×11（含 IR×2 + sim×1） |
| 关键路径 | tap 两阶段 +6.5% → IR store 行内仿射化 +13.0% |

### 陷阱 / 证伪方向

- **某 kernel 比同规模慢 10 倍**：store dst 含 div/mod → DiscreteMemAccess 标量循环（L1.10/§3.1）。
- **max 归约结果错（负值输入）**：mask `other=0` 参与 max → 改 `other=-inf`（L1.11）。
- **三分支输出互换/颠倒**：z 平面 cat 顺序与 conv 权重 c2 维不对应（L1.12 plane0=mean / plane1=max）。
- 证伪：BPR=128 行块翻倍 -1.4%；masked load 形态改造 -2.1%；rowreduce BLOCK_H 自适应为噪声；
  creduce BHW2 分档 64/32/16 +4~5% 有效但被 store 行内仿射化**整体取代**——同瓶颈的参数级与结构级解法，先试结构级。
