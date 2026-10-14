---
type: Guide
title: SparseFlashAttention 算子（sparse-attention） 实现手册
description: '算子类别: sparseattention。'
tags: [task.implementation, topic.attention, topic.compute.indexing, topic.implementation.masking, topic.implementation.memory_management, topic.compute.reduction]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# SparseFlashAttention 算子（sparse-attention） 实现手册

> 类别约束见 [transformer_inference_constraints.md](../constraints/transformer_inference_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §7 SparseFlashAttention 算子（sparse-attention）

**算子类别**: `sparse-attention`
**典型特征**: MLA-absorb (attention_mode=2)、MQA/N2=1、按 `sparse_indices` gather KV、`rightDownCausal` mask
**性能基准**: ascend910_9362 / 72-case 全量，几何平均 **0.9251x**，72/72 pass；37+/72 shape 快于 PyTorch

> ⚠️ **关键区分**：SFA 的核心瓶颈是 **sparse gather 的 MTE2/MTE3 流量**，而非 Cube 算力。以下 Layer 2/3 采用 gen3 route2 骨架；旧架构 gpack 路径已证劣化，不再保留。

### §7.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 精度基准必须用纯 PyTorch golden

- 禁止直接调用 `torch_npu.npu_sparse_flash_attention` 或 CANN 原生 op 作为 golden。
- native SFA 的 bf16 累加与纯 PyTorch 存在 ~1% 系统误差，在 MERE 容差下会导致正确实现也被判失败。
- 参考：[[sfa-golden-native-vs-purepytorch]]

#### L1.2 布局转换与连续化

- Q/K 必须先 pack 成 contiguous `[.., d_qk]`；V 转成 `[t2, n2, d_v]` 连续。
- BSND: `permute(0,2,1,3).reshape(t1, n2, g, d).contiguous()`
- TND: `reshape(t1, n2, g, d).contiguous()`
- PA_BSND: 按 `block_table` 物理块顺序展平

#### L1.3 预计算 per-(b,s1) 标量

- 预计算 `b_idx`、`s1_idx`、`cur_s2`、`cur_s1`、`s2_start`、`threshold` 等 per-t1 标量。
- 避免在 compute kernel 里做 cumsum 搜索或 Python `.item()` 同步循环。

#### L1.4 统一双 kernel 骨架

- **必须**采用 **gather kernel + compute kernel** 结构。
- 禁止使用 `partial_kernel + reduce_kernel` 的 HBM part_m/part_l/part_out 往返方案。
- base_par < 32 时走 split-KV 分支；否则走 direct online-softmax。

#### L1.5 gather 向量化 + span-select 去重广播

- 禁止用逐行 masked gather 遍历 sparse row（同地址 gather 会被硬件串行化，实测 11x 退化）。
- 用 `BLOCK_N=64` 批量 gather；tile 内不同 src 地址数 ≤ BS 时走 span-select：只加载 `[src_lo, src_lo+BS]` 窗口内连续行，再等值选择广播。
- **窗口必须钳制到 `min(BS, 4)`**：BS>4 时仅 span<4 走 span-select，否则回退逐行 masked gather。BS≥16 时窗口>4 会导致窗口外行静默取零（case 33/34/35/67/68 AccuracyError）。

#### L1.6 双源 rope 免拼接

- 禁止在 host 侧把 `q_nope + q_rope`、`k_nope + k_rope` 拼接。
- kernel 内 main 和 rope 两个独立 tensor：`QK = Σ dot(q_main, k_main) + dot(q_rope, k_rope)`，fp32 累加。
- rope 主 tile 固定 `BLOCK_DR=64`，chunk 顺序 main 在前 rope 在后。

#### L1.7 Tiling 参数硬性边界

- `M_TILE = 16` 固定，更大值在 D≤128 时容易 UB 溢出。
- `BLOCK_G = 32` 固定（combine kernel 中 BLOCK_G=64 会 UB 溢出）。
- `BLOCK_ROW` 在 direct 路径可选手动调度：g>=64 时推荐 `BLOCK_G=64, BLOCK_ROW=128`；否则 `BLOCK_G=32, BLOCK_ROW=256`。
- `BLOCK_D_GATHER` 按 BS 分档：BS==1 用 256；BS>1 用 128（span-select 分支需驻留 BS 份 fp32 广播临时量，256 时 UB 溢出）。
- `BLOCK_DV=256` 仅用于 br≤128 的路径；br=256 的 split 路径保持 128。
- 大 T1 场景必须按 `BLOCK_T1=16` 切 chunk；多 chunk 下禁止 split-KV。

#### L1.8 加法掩码替代 bool 2D where

- 禁止 softmax 关键路径上的 `tl.where(row_ok[None,:], ..., -2e35)`。
- 改为 1D f32 bias：`bias = tl.where(row_ok, 0.0, -2e35)`，`scores = acc * scale + bias[None, :]`。
- invalid 行精确舍入到 `-2e35`，`exp=0`；全无效行边界 `m_ij == -2e35` 时亦得 `exp(-2e35)=0`。

#### L1.9 fp32 累加全链（又见 §1 T2）

- `tl.dot` 输出 dtype 必须为 fp32；`m_i`、`l_i`、`acc` 全程 fp32，仅在 store 时 cast 回 bf16/fp16。
- 禁止 p 量化 fp16/bf16，禁止 QK K 维合并（会改变 fp32 归约分组，verify 0 容差失败）。

#### L1.10 环境修复

- 文件头部必须插入 sys.path 修复代码，优先使用 `megablock_env_extract` 或本地 `pylibs/` 的 triton-ascend。
- 参考：L3B.1 代码片段。

### §7.2 Layer 2: 算法骨架（gen3 route2）

#### L2.1 整体控制流

```
for each T1 chunk (BLOCK_T1=16):
    1. gather_kv_kernel  → 连续 workspace (pos_ws, k_ws, kr_ws, v_ws)
    2. 路由选择：
       if base_par < 32:
           split_attn_kernel (split-KV, ×NUM_SPLITS) + split_combine_kernel
       elif t1<=BLOCK_T1 且 direct programs<64 且 温和 split 条件满足:
           split_attn_kernel (2 段 split) + split_combine_kernel
       else:
           gathered_attn_kernel (direct, 单 kernel)
```

#### L2.2 Host 侧预处理

1. 布局转换（L1.2）
2. 预计算 per-t1 标量（L1.3）
3. 分配 workspace：pos_ws、k_ws、kr_ws、v_ws
4. 按 L2.1 路由分派 kernel

#### L2.3 路由参数

- `g_blocks = ceil(g / BLOCK_G)`
- `base_par = t1c * n2 * g_blocks`
- `NUM_SPLITS = min(ceil(sel_rows / BLOCK_ROW), ceil(64 / base_par))`，上限建议 ≤8
- 温和 split 条件：`t1 <= BLOCK_T1` 且 `t1c * n2 * ceil(g / bg_dir) < 64` 且 `sel_rows >= 4 * br_dir`

### §7.3 Layer 3: 关键代码模式

#### L3.1 环境修复 sys.path 插入

```python
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
for _cand in ("/home/l30083907/megablock_env_extract",
              _os.path.join(_here, "..", "pylibs"),
              _os.path.join(_here, "pylibs")):
    _cand = _os.path.abspath(_cand)
    if _os.path.isdir(_os.path.join(_cand, "triton", "backends", "ascend")):
        if _cand not in _sys.path:
            _sys.path.insert(0, _cand)
        break
```

#### L3.2 gather 向量化 + span-select

```python
# grid: (t1c * n2, ceil(SEL_ROWS, BLOCK_N))
pid_t = tl.program_id(0)
pid_n = tl.program_id(1)
n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

idx = tl.load(IDX + base + block_idx, mask=row_in, other=-1)
src = kv_start + idx * BS + offset
valid = row_in & (idx != -1) & (src >= kv_start) & (src < kv_end)
src_safe = tl.where(valid, src, kv_start)

src_lo = tl.min(src_safe)
src_hi = tl.max(src_safe)
span_ok = (src_hi - src_lo) < (BS if BS <= 4 else 4)

if span_ok:
    # 加载窗口内 BS 行，等值选择广播
    k_local = tl.load(K + src_lo + tl.arange(0, BS)[:, None] * stride + offs_d[None, :])
    k_val = tl.where((src_safe == src_lo + 0)[:, None], k_local[0:1, :], 0.0)
    if BS > 1:
        k_val = tl.where((src_safe == src_lo + 1)[:, None], k_local[1:2, :], k_val)
    # ...
else:
    k_val = tl.load(K + src_safe[:, None] * stride + offs_d[None, :],
                    mask=valid[:, None], other=0.0)
```

#### L3.3 双源 rope dot 累加

```python
# Main: BLOCK_D=128 chunk
acc = tl.zeros([BLOCK_G, BLOCK_ROW], dtype=tl.float32)
for c in range(NUM_DM):
    qc = tl.load(q + offs_g[:, None] * D_MAIN + c * BLOCK_D + offs_d[None, :])
    kc = tl.load(k_ws + offs_k[:, None] * D_MAIN + c * BLOCK_D + offs_d[None, :])
    acc += tl.dot(qc, tl.trans(kc))

# Rope: BLOCK_DR=64 窄 tile
if HAS_ROPE:
    for c in range(NUM_DR):
        qr = tl.load(qr + offs_g[:, None] * DR + c * BLOCK_DR + offs_dr[None, :])
        kr = tl.load(kr_ws + offs_k[:, None] * DR + c * BLOCK_DR + offs_dr[None, :])
        acc += tl.dot(qr, tl.trans(kr))
```

#### L3.4 加法掩码

```python
pos = tl.load(POS_WS + ws_base + rows, mask=row_in, other=-1.0)
row_ok = row_in & (pos >= 0.0) & (pos < thresh)
bias = tl.where(row_ok, 0.0, -2e35)
scores = acc * SCALE + bias[None, :]

m_ij = tl.maximum(m_i, tl.max(scores, 1))
p = tl.exp(scores - m_ij[:, None] + bias[None, :])
alpha = tl.exp(m_i - m_ij)
l_i = l_i * alpha + tl.sum(p, 1)
# PV ...
m_i = m_ij
```

#### L3.5 split-KV combine

```python
# 第一遍：全局 max
m_tot = tl.full([BLOCK_G], -2e35, dtype=tl.float32)
for s in range(NUM_SPLITS):
    m_s = tl.load(P_M + s * tng + pg, mask=g_mask, other=-2e35)
    m_tot = tl.maximum(m_tot, m_s)

# 第二遍：logsumexp 加权合并
l_tot = tl.zeros([BLOCK_G], dtype=tl.float32)
acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)
for s in range(NUM_SPLITS):
    m_s = tl.load(P_M + s * tng + pg, mask=g_mask, other=-2e35)
    l_s = tl.load(P_L + s * tng + pg, mask=g_mask, other=0.0)
    w = tl.exp(m_s - m_tot)
    l_tot += l_s * w
    acc += w[:, None] * tl.load(P_ACC + (s * tng + pg)[:, None] * D_V + offs_d[None, :], ...)

out = tl.where(l_tot[:, None] > 0, acc / l_tot[:, None], 0.0)
```

### §7.4 性能基准与已验证死路

#### 当前基准

- **ascend910_9362 / 72-case 全量**: 几何平均 **0.9251x**，72/72 pass。
- Framework 平均延迟 ~0.168 ms，Triton 实现 ~0.182 ms。
- 37+/72 shape 快于 PyTorch。

#### 低速比 case 特征

- 主要集中在 **小 S1**（S1=1,3,5,8,13,21,34,55,89）且 S2 较大的场景。
- 瓶颈：kernel 启动/固定开销占比高，加上后端固有的 MIX GM 往返与 fp32 dot 发射率限制。

#### 已证死路（禁止尝试）

| 方向 | 结果 | 原因 |
|------|------|------|
| bf16 workspace | -30% | compute 读 bf16 转 fp32 的 convert 开销超过流量收益 |
| BLOCK_T1=32 | 严重回退 | workspace 峰值爆炸，L2 压力 |
| QK K 维合并 | verify 失败+负收益 | 改变 fp32 归约分组，0 容差失败；UB 压力 |
| 全局 BLOCK_N=64 | case 回归 | 必须条件化启用 |
| 全局 32/256 direct tiling | 0.9106x | 对当前 workload 不如 64/128 |
| 无约束 autotune | 不采纳 | 扫描确认当前配置即局部最优 |
| fused gather-attention 单 kernel | 25-60x 劣化 | 逐行 gather 破坏向量化 |
| partial_kernel + reduce_kernel | 严重劣化 | HBM 中间态往返 |

#### 残余差距根因

- CANN 原生 SFA 使用 DataCopyPad/srcGap/Preload/Vector-issue 等硬件 gather 优化，**Triton 用户层无法表达**。
- 0.9251x 之后，进一步提升需要突破 Triton 表达墙或改变算法结构，常规 tiling/split 优化已扫尽。

---
