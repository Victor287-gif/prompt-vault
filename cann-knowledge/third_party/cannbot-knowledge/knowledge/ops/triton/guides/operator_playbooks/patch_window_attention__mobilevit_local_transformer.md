---
type: Guide
title: MobileViT 系局部 transformer（patch_window_attention） 实现手册
description: '典型特征: 卷积-块化窗口 transformer——conv 局部表示（k×k+1×1）→ NCHW→(b,p,n,D) patch
  块化 unfold → 窗口内稠密多头 attention → fold → 尾部 conv(cat([x,y])) 双输入融合卷积；49 case 从 0.024
  → 1.08 的完整轨迹。'
tags: [task.implementation, topic.attention, topic.compute.neighborhood.convolution, topic.compute.composite, topic.data_movement.layout_transform, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-31T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# MobileViT 系局部 transformer（patch_window_attention） 实现手册

> 类别约束见 [patch_window_attention_constraints.md](../constraints/patch_window_attention_constraints.md)（L1.1~L1.10、S1~S4、§5 优化点/证伪表），
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 Layer 2: 算法骨架

### §3.1 主骨架（前向，分阶段多 kernel）

```
x[b,c,h,w] (fp32 内部计算)
 ├─ im2col(x): [b, hw, c·k²] 紧凑缓冲（NHWC 一次转置 + per-tap 纯拷贝，§4.1）
 ├─ conv1 GEMM: x_col @ W1ᵀ[c·k², c] + bias          → y1 [b·hw, c]
 ├─ conv2 GEMM: y1 @ W2ᵀ[c, dim] + bias               → y2 [b·hw, dim]
 ├─ unfold（布局视图）: y2.view(b,nh,ph,nw,pw,D).permute(0,2,4,1,3,5) → [b, p, n, D]
 ├─ transformer × depth（每层 7 kernel，无 host sync）:
 │    LN → qkv GEMM → attention(按 n 分派) → out GEMM(+残差) → LN → ff1 GEMM → ff2 GEMM(SiLU+残差)
 ├─ fold（布局视图）: 逆 permute 回 [b, hw, D]
 ├─ conv3 GEMM: y_fold @ W3ᵀ[dim, c] + bias           → y3 [b·hw, c]
 ├─ im2col(y3): 同款 NHWC 路径                          → y_col [b, hw, c·k²]
 ├─ conv4 两段链式:  x_col @ W4aᵀ → tmp;  y_col @ W4bᵀ + tmp + bias → out[b·hw, c]
 │    （双输入融合卷积，L1.1/L1.5 的链式写法）
 └─ 输出布局 [b,hw,c] → NCHW permute（torch 视图）
```

要点：
- **conv / 投影全部走同一个 GEMM kernel 的 constexpr 变体**（HAS_BIAS / ADD_RESIDUAL / APPLY_SILU_A），一份 kernel 覆盖全算子的矩阵乘段；
- unfold/fold 只做 permute+reshape 视图（torch 形状操作），**不物化**（permute 后 reshape 会拷贝一次，量级 ~数百 MB/case，可接受；进一步消除会破坏 GEMM 连续性，见 §5.2 证伪表）；
- attention 的 grid = min(核数, 任务数)，任务 = (b, patch)，head 在 kernel 内循环。

### §3.2 ★★ attention 的 n 分派判据

| 窗口规模 | 路径 | tile | 说明 |
|---|---|---|---|
| np2(n) ≤ 128 | **KV 单块特化** | BQ=128（n>64），BKV=np2(n) | 无 m/l 滚动、无 alpha、EVEN 时无边界 where；dot 次数 = ceil(n/BQ) |
| n > 128 | 多块 flash | BQ=128, BKV=128 | 标准 online softmax，m 初始 -1e30 |

单块特化对小 n 的收益 1.2~2.0×；对 n≤64 的极小窗口，BQ=BKV=np2(n)（≥16）。

### §3.3 GEMM 配置档位（fp32 ieee 实测）

| 变体 | BLOCK 配置 | 实测吞吐（910B2） |
|---|---|---|
| plain（无 residual/silu） | BM=256 / BN=128 / BK=128 | ~67.5 TFLOPS |
| residual / silu 变体 | BM=256 / BN=128 / BK=128（与 plain 同档） | ~67 |
| LN | BLOCK_M = max(4, 8192//BLOCK_D) | — |

BN=256 与 BK=256 均不可编或更慢；grid 收缩 `min(核数, cdiv(M,BM)·cdiv(N,BN))`。

## §4 Layer 3: 关键技巧

### §4.1 ★★★ im2col 的正解：NHWC 一次转置 + per-tap 纯连续拷贝

per-tap 做 `NCHW 切片 → permute(0,2,3,1) → reshape → 写入`（每个 tap 一次 4D 转置拷贝）是隐性大头。正解分两步：

```python
srcn = src.permute(0, 2, 3, 1).contiguous()          # 第一步：全图一次 NHWC 转置
xp = torch.zeros((b, h + 2*pad, w + 2*pad, c), ...)  # 第二步：pad 后按 tap 纯拷贝
xp[:, pad:pad+h, pad:pad+w, :] = srcn
for t in range(k*k):                                  # 每 tap 连续 reshape 拷贝，无 permute
    dst[:, :, t*c:(t+1)*c] = xp[:, di:di+h, dj:dj+w, :].reshape(b, h*w, c)
```

实测大 case **18.9ms → 10.5ms（1.8×）**，数值逐位等价。行主序源（token 布局 [b,hw,c]）侧同理已是 NHWC 等价路径，无需再动。

### §4.2 ★ 双输入融合卷积的两段链式（替代 host cat 与双 B 指针）

第一段 GEMM 出 tmp（无 bias），第二段 `ADD_RESIDUAL` 在 epilogue 读 tmp 相加 + bias。
tmp 的跨布局读取（如转置世界 [c, b·hw]）用独立 stride 参数（rs_m/rs_n）寻址，不要在 host 侧再转置一次。

### §4.3 ★ GEMM 变体同档复用：epilogue 代价远小于降档损失

bias / residual / silu 变体的 epilogue 只在 K 循环外多一次 load + 一次加法（residual tile 仅在收尾驻留），
UB 预算足够支撑与 plain 相同的 BM=256/BK=128——同档实测 ~67 TFLOPS。把变体降档到 BM=128/BK=64 会损失
~30% 吞吐，是本类算子最常见的"自我设限"。唯一例外：APPLY_SILU_A 在 K 循环**内**对 A tile 做激活时，
多一份激活中间量，若编译报 UB 溢出再单独对该变体降 BK（先降 BK 再降 BM）。

### §4.4 ★ 诊断方法论：GEMM "吞吐上限"先问搬运还是计算

GEMM 卡在某个 TFLOPS 时，用 `msprof op simulator` 采 per-instruction pipe 分布（单 kernel 小脚本仿真，
勿整 forward）：若 **MTE2/MTE1 合计 >70% 而 CUBE <20%**，瓶颈是搬运未流水
（`MOV_OUT_TO_L1*_ND2NZ` + `SET_FLAG` 等待），不是计算极限——此时调 BLOCK 无益，见证伪表的流水掩盖行。
IR 侧旁证：`npuir.mlir` 里 plain GEMM 应只有 1×mmad/1×sync；attention 的 38×sync_block 属 mix 结构固有。

## §5 证据基础（49 case，2026-08，910B2C / CANN 9.1.0 / triton-ascend 3.2.2）

- 覆盖：B∈[1,8]、C∈[4,1024]、H/W∈[2,406]、窗口 n∈[1,725]、patch∈[2,14]、dim∈{256..1024}、fp16/bf16/fp32 混合。
- **精度全 50/50**；几何平均从基线 **0.024 → 1.08**（三轮优化，先修双 B 指针写法崩溃，再 im2col NHWC，最后 attn 单块特化 + GEMM 同档）。
- 本类算子无投影段 GEMM 可吃时，收益全部在**结构**（L1.1 复合卷积写法 + im2col 布局 + attn 分派 + GEMM 同档），tile 级微调无普适赢面（§5.2 证伪表）。
