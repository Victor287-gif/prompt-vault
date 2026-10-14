---
type: Guide
title: FusedRecurrentGatedDeltaRule 算子（Gated Delta Rule） 实现手册
description: '典型特征: 每 (b,vh) 一套 [K,V] fp32 状态，T 步串行递推，delta rule rank-1 修正，状态驻留 UB
  单融合 persistent kernel'
tags: [task.implementation, topic.compute.composite, topic.precision, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-26T08:27:20Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# FusedRecurrentGatedDeltaRule 算子（Gated Delta Rule） 实现手册

> 类别约束见 [../linear_recurrent_constraints.md](../constraints/linear_recurrent_constraints.md)（L1.1~L1.9 硬性边界），
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## FusedRecurrentGatedDeltaRule 算子（fused_recurrent_gated_delta_rule）

**算子类别**: `linear_recurrent`
**典型特征**: 每 (b,vh) 一套 [K,V] fp32 状态，T 步串行递推，delta rule rank-1 修正
`S += k⊗((v−Sᵀk)·β)`，状态驻留 UB 单融合 persistent kernel。
**性能基准**: **50/50** 精度通过（全部 dtype × flag 组合，2026-08-24 独立复验）；性能 geomean
**0.143×** vs torch（原 5.7527× 声明已撤回，测量口径事故）；对照同一 benchmark 上另一会话
13 轮迭代的最优实现 **1.3231×**。

> 锚定算子：`92_FusedRecurrentGatedDeltaRule`（naive/default，fla 系 fused_recurrent gated delta
> rule）。探索期 2026-08-21 ~ 2026-08-24。硬件 Ascend910_9372（DAV_2201，910B 家族）。
> 任务约束（本算子用户指定，同类任务可沿用）：forward() 计算路径禁止调用 torch_npu 专有接口，
> 全部计算由 Triton kernel 完成（torch 仅限布局/视图/空分配等宿主操作）。

### Layer 2 算法骨架（单融合 persistent 递推 kernel）

```text
grid = (B * VH,)                         # 每 program: (b, vh)；GQA: h = vh // GQA_R
状态 S[BK, BV] fp32 = zeros / load(h0)   # 驻留 UB；h0 按 state_v_first 决定加载索引交换
for t in range(T):                       # 串行数据依赖，禁 break（L1.1）
    1) 门控衰减（flag 组合）:
         S *= exp(-exp(A_log)·softplus(g + dt_bias))     # gate_mode & 1 且 use_gate_in_kernel
       或 S *= exp(g)                                     # 标量 gate 路径
       或 S *= exp(gk)[:,None] · exp(gv)[None,:]          # 逐维 gate（&2 / &4）
    2) 读 q/k/v/beta（GQA: q/k 用 h，v/beta 用 vh）
       可选: l2norm(q), l2norm(k), sigmoid(beta)（allow_neg_eigval 时 ×2）
    3) delta rule rank-1 修正:
         kv    = Sᵀ k                    # [BV] = sum_k k[k]·S[k,v]
         resid = (v − kv) · β
         S    += k ⊗ resid
    4) 读出: o_t = (q·scale)ᵀ S          # [BV]，store 时 cast 回目标 dtype
末状态: S 按 1D flat [BK*BV] 连续写回（L1.3），host 侧 reshape/transpose 还原
```

host 侧要点：非 contiguous 输入防御性 `.contiguous()`；flag 全部转 bool 以 constexpr 传入（L1.2）；
可选输出用占位张量（L1.7）；返回前 `synchronize()`（L1.6）；每 program 常量（如 `A_log[vh]`、
`dt_bias[vh]`）在 t 循环外预取（LICM）。

### Layer 3 关键代码片段（技巧可参考，变量名/结构须重设计）

**L1.3 修复：状态 1D flat 写回 + host 侧还原**
```python
# kernel 内：[BK,BV] 状态按 1D 连续布局写，元素 [k,v] 的 flat index = k*BV + v
flat_offs = offs_k[:, None] * BV + offs_v[None, :]
tl.store(ht_ptr + (b * VH + vh) * BK * BV + flat_offs, s_tile, mask=bound_mask)
```
```python
# host 侧：view -> 裁剪 -> 按目标布局转置
final = buf.view(B, VH, BK, BV)[:, :, :K, :V]
if state_v_first:
    final = final.transpose(-2, -1).contiguous()   # 目标 [B, VH, V, K]
```

**L1.2 constexpr flag 特化（调用侧）**
```python
kernel[(B * VH,)](
    q, k, v, ..., output, final_state,
    T, K, V, VH, H, GQA_R, scale,
    BK=next_pow2(K), BV=next_pow2(V),
    HAS_G=bool(gate_mode & 1), HAS_GK=bool(gate_mode & 2), HAS_GV=bool(gate_mode & 4),
    USE_INITIAL_STATE=bool(...), OUTPUT_FINAL_STATE=bool(...), ...,
)
```

**L1.4 fp32 边界 mask / L1.5 稳定 softplus**
```python
m_k = offs_k.to(tl.float32) < K.to(tl.float32)
sp  = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))   # F.softplus(threshold=20)
```

### 精度与性能实测

- **精度 50/50**（全部 dtype × flag 组合；2026-08-24 独立复验确认）。头号精度教训：**masked 2D
  state store 静默丢格**——output 全对（~9.5e-7）而 final_state 错，20/25 失败 case 源于此（L1.3）。
- **性能 geomean 0.143×** vs torch（50/50 精度通过）——比 torch 基线慢约 7 倍，未达任何加速目标。
  原声明的 **5.7527× 已撤回**（同环境配方独立重测不可复现，根因是原测量 framework 基线逐 case
  慢 ~3-12×，见 [../linear_recurrent_constraints.md](../constraints/linear_recurrent_constraints.md) §5.1）。
- 单融合 persistent kernel（状态驻留 UB + flag constexpr 特化 + device 内串行 t 循环）仍是本类形态
  的正确**架构方向**——精度 50/50 可达、结构合理；但**架构正确 ≠ 性能达标**：本实现 kernel 体
  （逐 token 标量 load + rank-1 更新）在大多数 case 上仍慢于优化过的 torch eager 参考实现，
  性能优化空间仍大（对照：同一 benchmark 上另一会话 13 轮迭代的最优实现为 **1.3231×**）。
- host 侧清理类微调（输入 contiguous 化精简等）无额外收益（见约束卡 §5），不值得做。

### 陷阱 / 证伪方向

- masked 2D store 写状态 → L1.3 flat 1D store + host reshape（本算子 20/25 失败 case 根因）。
- BK=64 PlanMemory 编译失败 → L1.8 先 UB 记账再选 tile。
- 未设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 双后端冲突大面积伪失败 → L1.9。
- state_v_first 初态/末态错位：`[V,K]` vs `[K,V]` 索引交换必须加载与写回两侧都按 flag 交换
  （Layer 2 骨注）。
- 已排除：标量头基准指针状态写回（Variant A 无效）；row-wise store 首版（编译错误，修正后 = L1.3
  方案 2）；host 侧清理迭代 #17（无提升）。
