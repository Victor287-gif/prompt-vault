---
type: Guide
title: Paged KV Cache Attention 算子（decode / prefill） 实现手册
description: '典型特征: KV 经 block_table 间接寻址的分页注意力，批量小页 decode / 大页 decode / split-q
  prefill 三子形态，跨页不跳块（整段 KV 都参与 softmax）。'
tags: [task.implementation, topic.attention, topic.implementation.masking, topic.compute.normalization, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-26T08:23:10Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---

# Paged KV Cache Attention 算子（decode / prefill） 实现手册

> 类别约束见 [paged_attention_constraints.md](../constraints/paged_attention_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 Paged KV Cache Attention 算子（decode / prefill）

**算子类别**: `paged_attention`
**典型特征**: KV 经 `block_table` 间接寻址、页与页显存不连续但**跨页不跳块**——整段 KV 都参与
softmax；三子形态：批量小页 decode（M=G≤8，每页固定开销受限）、大页 decode（SWA/GQA 单序列，
内存受限）、split-q prefill（计算受限）。
**性能基准**: decode 去 offset-load 向量 gather 后 geomean **251.8→17.6µs**、prefill block_ptr+helper
流水线 **90.7→54.5µs**；批量 decode G==1 向量通路 **+24.3%**、除法外提 **+7.1%**
（Ascend950PR prefill/大页 decode，Ascend910B2 批量小页 decode）。

### Layer 1 设计约束（算子专属）

> L1.1-L1.12 全部硬约束及 Why/How 见 [paged_attention_constraints.md](../constraints/paged_attention_constraints.md)，
> 与类别卡同等生效。本手册按子形态给落地要点：

- **L1.1 加载红线**: 批量小页 decode 页基址是标量（BLOCK_BS ≤ BS 不跨页），affine 2D offset load 与
  block_ptr 均合规；大页 decode / prefill 必须 block_ptr + `boundary_check=(0,1), padding_option="zero"`，
  prefill 的 KV/Q/O 全部 block_ptr 且 KV load 写在 helper 内。
- **L1.2 页界（精度红线）**: 批量小页直接取 `BLOCK_BS = BS`（BS 全 2 幂恒整除）；大页/prefill 取
  `BLOCK_N = min(128, next_pow2(PAGE_SIZE))`，禁止 BLOCK_N > PAGE_SIZE（跨页补零被当真实 key）。
- **L1.3 GQA 映射**: decode 按 kv_head **反推**（ABAB `kv + r*nkv` / AABB `kv*ratio + r`）；prefill 按
  q_head **正推**（ABAB `%` / AABB `//`）；批量 decode `head = kh*G + g`（AABB，g∈[0,G)）。
- **L1.4 任务划分**: decode `grid=(kv_heads, batch)` 或 persistent（min(核数, tasks) + 核内步长）；
  prefill `grid=(cube_num,)` + kernel 内轮询 `(batch, q_block, q_head)`；G==1 走向量通路（不走 dot 通路）。
- **L1.9 舍入链**: golden 带 cast 链（`softmax(...).to(v.dtype)` / `einsum(...).float()`）时**必须两轮
  KV 循环**（批量小页 decode；大页/prefill 同规）；golden 无 cast 链时才可单轮 online。
- **L1.11 标量舍入**: host 侧 `_scale_for_dtype(scale, Q.dtype)`（fp16 RNE 预舍入，bf16/fp32 直传）。

### Layer 2 算法骨架

#### 子形态 A：批量小页 decode（两轮 KV 循环 + 双通路分派，golden 含 cast 链时）

```python
# host（forward，AST 白名单内）:
#   G = H // KH；sc = _scale_for_dtype(scale, Q.dtype)            # L1.11
#   out = torch.empty((NS, H, D), dtype=Q.dtype)                  # 唯一 host 分配
#   if G == 1:    # MHA 向量通路（无 tl.dot，调度全部 vector 核）
#       task = (seq, q_head)；grid = min(num_vec_cores, NS*H)
#   else:         # GQA Cube dot 通路
#       BLOCK_G = 16（tl.dot M 下界；G 个 q head 并成 q tile，行掩码 g<G）
#       task = (seq, kh, gb)；grid = min(num_cube_cores, NS*KH*GB)
#   NUM_CORES = grid[0]（核内步长循环 for task in range(pid, total, NUM_CORES)）

# kernel（每 task，两轮 KV 循环）:
#   kv_len   = min(load(SL + seq), MB*BS)                         # 标量 load + cap
#   kv_start = max(kv_len - sw, 0) if HAS_SW else 0               # L1.6 区间收缩
#   q_tile   = load Q[seq, kh*G + g, :]（行掩码 g<G）
#   q_tile   = (q_tile * sc).to(native)                           # L1.10 同序舍入
#   m = -1.0e30（有限极小值）; l = 0
#   轮 1 for kb in [kv_start//BS, cdiv(kv_len,BS)):
#       phys = load(BT + seq*MB + kb)                             # 标量（L1.1/L1.2）
#       k = load [BS, D]（标量页基址 affine / block_ptr）
#       s = dot(q, kᵀ).to(native).to(f32)                         # L1.10 round-trip
#       s = where(valid[None,:], s, -3.0e38)                      # L1.12 掩在 scores 上
#       m_new = max(m, rowmax(s)); l = l*exp(m-m_new) + rowsum(exp(s-m_new)); m = m_new
#   inv_l = 1.0 / where(l>0, l, 1.0)                              # 倒数预计算（外提）
#   轮 2 for kb in [kb0, kb1):
#       k = load; v = load（v 提前到 dot 前）                      # 早 V-load
#       s = dot(q, kᵀ).to(native).to(f32)                         # 与轮 1 逐位一致
#       p = exp(s - m) * inv_l                                    # ★ 先归一化再舍入
#       acc += dot(p.to(native), v)                               # m 固定 ⇒ 无 rescale
#   store out[seq, head, :] = acc.to(native)，行掩码 g<G          # 出口不除 l
```

**UB 预算（192KB；原生 PV 无 v.to(f32) 大缓冲）**: 最坏 D=256, BS=64 时
k 32KB + trans(k) 32KB + v 32KB + q 8KB + scores/p 8KB + acc 16KB ≈ **130KB** ✓；
D=128,BS=64 ≈ 70KB。`tl.trans(k)` 按一份等大空间记账（`gqa.md` G3）。

#### 子形态 B：大页 decode（单轮 online GQA-grouped FlashDecoding，golden 无 cast 链时）

1. `grid=(num_kv_heads, batch)`；`pid_kv=program_id(0)`, `pid_b=program_id(1)`
2. `kv_seq_len = load(seqlens[pid_b])`；按 L1.3 算 `head_idx[r]`；`qg = load(Q[b, head_idx, :BLOCK_D])`（pad BLOCK_M≥16）
3. `m=-inf, l=0, acc[BLOCK_M,BLOCK_D]=0`
4. **页循环** `for nb in range(max_num_blocks)`: 若 `page_start>=kv_seq_len` 停；L1.6 死区判据 skip；否则 `phys=load(block_table[b,nb])`
   - **页内 tile 循环** `for tn in range(0,PAGE_SIZE,BLOCK_N)`:
     - `k = load_block_ptr(K[phys, kv, tn:tn+BN, :BD])`（L1.1）
     - `qk = dot(qg, trans(k)).fp32 * scale`；apply SWA keep mask（L1.6）；在线 softmax 合并（m_new/alpha/p/l）
     - `v = load_block_ptr(V[…])`；`acc = acc*alpha + dot(p.bf16, v)`
5. `out = where(l>0, acc/l, 0)`；store `O[b, head_idx, :HEAD_DIM]`

> BLOCK_M 用 `tl.dot` 最小形（≥16）；decode ratio 通常 ≤8，pad 到 16。空 seq（l==0）输出 0。
> ⚠️ 前提是 golden 的 p 是 fp32 干净值（无显式 cast）。golden 带 `softmax(...).to(v.dtype)`
> 时单轮 online 不满足舍入链契约，须改子形态 A 两轮骨架——**两轮结构与 G==1 分派对大页形态同样适用**。

#### 子形态 C：prefill（split-q persistent FlashPrefill）

1. `grid=(cube_num,)`；`pid=program_id(0)`, `n_progs=num_programs(0)`
2. 遍历 batch：`q_start/end = load(cu_q_lens[b]/[b+1])`；`kv_seq_len = load(seqlens_kv[b])`；`cur_q_tasks = cdiv(q_seq_len,BLOCK_M)*NUM_Q_HEADS`
3. **轮询领取任务** `for q_task_id in range((prev_q_tasks+pid)%n_progs, cur_q_tasks, n_progs)`:
   - `q_block_id = q_task_id // NUM_Q_HEADS`, `q_head_id = q_task_id % NUM_Q_HEADS`；GQA 映射见 L1.3
   - `Q_bp/O_bp = make_block_ptr(...)`；`q = load(Q_bp)`；`m=-inf, l=0, acc=0`
   - **KV block 循环** `for kv_block_id in range(num_kv_blocks)`: paged 查 phys（L1.2）→ `K_T_bp/V_bp = make_block_ptr(...)` → `mask = causal_mask_fn(aux_mask,…)`（L1.5）→ `acc,l,m = _prefill_flash_block(...)`（L1.1/L1.7 helper）
4. `m_i += log(l_i)`；`out = acc/l_i[:,None]`；`store(O_bp, …)`

> BLOCK_M=128, BLOCK_D=HEAD_DIM, BLOCK_N=min(128,next_pow2(page_size))。online softmax（fp32）。
> 同子形态 B 前提：带 cast 链的 prefill 变体须按 L1.9 契约改造。

#### 骨架选择判据

| 判据 | 选择 |
|---|---|
| query_len==1 且 NS 大、页小（BS≤64）、golden 带 cast 链 | 子形态 A（两轮 + persistent 任务循环） |
| 序列少、页大、golden 无 cast 链 | 子形态 B（单轮 online + 2D grid） |
| query_len>1（prefill） | 子形态 C（split-q persistent） |
| G==1（任意 decode） | 向量通路 kernel 替换 dot 通路 |
| 任务数 < 核数（如 NS=1） | 属结构约束，不要靠改骨架硬撑 |

### Layer 3 关键技巧（可参考不可照抄）

- **block_ptr 加载 + dot（L1.1 杠杆）**:

```python
k_bp = tl.make_block_ptr(base=k_ptr + phys*stride_kb + kvh*stride_kh,
                         shape=(PAGE_SIZE, HEAD_DIM), strides=(stride_ks, stride_kd),
                         offsets=(tn, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
k = tl.load(k_bp, boundary_check=(0, 1), padding_option="zero")
qk = tl.dot(qg, tl.trans(k)).to(tl.float32) * scale
```

- **SWA keep mask（大页 decode；批量小页用 L1.6 区间收缩替代）**:
  `keep = (j < kv_seq_len) & ((j >= local_thr) | (j < global_window))`；`qk = tl.where(keep[None,:], qk, -float("inf"))`。
- **helper 早 V-load + 分离 scale/sub 触发 auto-multi-buffer（L1.7 核心）**:

```python
@triton.jit
def _prefill_flash_block(acc, l_i, m_i, q, K_T_bp, V_bp, softmax_scale, mask, HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, is_fp8):
    # 早 V-load（与下面 QK→softmax 标量链重叠 → 触发 bishengir 双缓冲）
    v = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(q, K_T_bp_load, ...)               # K load 也在 helper 内
    qk = qk * softmax_scale
    qk = tl.where(mask, qk, -float("inf"))
    # 分离的 softmax 标量链（不要 inline 进 dot）
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.math.exp(m_i - m_new)
    p = tl.math.exp(qk - m_new[:, None])
    p = propagate_nan(m_new, p)                     # 处理 -inf 行
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    tl.dot(p.to(v.dtype), v, acc)                   # 3-arg 累加 dot
    return acc, l_i, m_new
```

- **aux_mask 表 load（L1.5，替代算术 mask）**: 下三角 bool 表按 `offset = clip(kv_start - q_abs_start, …)`
  线性偏移 `tl.load(mask_ptr + mask_offs).to(tl.int1)`——load 走 **mte2** 不占向量核；实测 0.84× → 1.00×。
- **★★★ G==1（MHA）向量通路分派（批量 decode 最大杠杆）**: G==1 时 ① KV 分组**无复用收益**；
  ② `tl.dot` 的 M=1 pad 到 16 行 ⇒ **15/16 Cube MAC 浪费**；③ 任务天然可细分为 (seq, q_head)。
  ⇒ 走**无 dot 的纯向量 kernel**（QK/PV 用「升 fp32 逐元素乘 + 树形归约」，`tl.sum(q_f[None,:] * k_tile.to(f32), axis=1)` 后 `.to(native).to(f32)` round-trip），可调度到全部 vector 核（= 2× cube 核数）。
  实测 geomean **+24.3%**。两通路必须共享同一舍入链契约（L1.9/L1.10）。M=G≥2 时恢复 dot 通路
  （KV 访存 1/G 折扣 + 行数填满 dot tile）。分派开销 = host 侧一个 if。
- **★ 循环不变量除法外提（inv_l 预计算）**: 轮 2 每页一次 `[BLOCK_G, BLOCK_BS]` 全 tile 向量除法 →
  循环外一次倒数 + 循环内乘法。实测 geomean **+7.1%**。精度：x/l 与 x*(1/l) 差 ~1ulp fp32，p 舍入
  native 翻转概率 ~2^-13，MERE 余量充足时安全，落地后走 verify 确认。
- **v_tile 提前 load**: 编译器不重排 load 顺序，手动把 `v_tile = tl.load(...)` 提到 QK dot 之前发射，
  MTE2 load 与 Cube 计算重叠。收益小但零风险。
- **grid-split FlashDecoding（仅 grid≤4 的大页 decode）**: BIGPAGE（grid=2）SPLITS=4 → 端到端 **1.25×**；
  GROUP1（grid=4）→ 1.11×；grid≥8 无效甚至有害（LONG1 grid=8 → 0.97×，MBF16-512 grid=16 → 0.78×）。
  hybrid 双路径只多 ~0.06× 且仅 2 case 受益，不值得；保持 single block_ptr kernel。split 要点：
  tile-strided（`for tid in range(pid_split, MAX_TILES, NUM_SPLITS)`）、`page=tid//TILES_PER_PAGE`
  （BN 整除 PS 保证 tile 不跨页）、partial `m/l/acc[BLOCK_M,BLOCK_D]` 写 workspace、reduce kernel
  online-merge（注意 `acc*a[:,None]` broadcast）。⚠️ 本条的 split 是双 kernel 显式 split+reduce；
  kernel 内多页合并/拼页已证伪。

### 性能基准 / 优化记录表

| # | 方向 | 收益 | 适用 |
|---|---|---|---|
| 1 | ★★★ K/V 加载改 block_ptr（去 offset-load 向量 gather） | decode：mac 1.7%→17-27%，geomean **251.8µs→17.6µs**；prefill：重叠 1.12×→**1.72×**（90.7→54.5µs） | 所有形态（L1.1） |
| 2 | ★★★ G==1 向量通路分派 | **+24.3%**（geomean，批量 decode） | G==1 的 decode |
| 3 | ★★ helper 内 op 顺序触发 auto-multi-buffer（早 V-load + 分离 scale/sub） | **82µs → 64.7µs** | prefill（L1.7） |
| 4 | ★★ aux_mask 表 load 替代算术 mask | 0.84× → 1.00× | prefill（L1.5） |
| 5 | ★ 循环不变量除法外提 | **+7.1%**（geomean，批量 decode） | 两轮结构轮 2 / 任何循环不变除法 |
| 6 | ★ SWA 区间收缩 / 死区跳页 | 页迭代数最多减半（sw=kv_len/2 时） | 带滑窗 decode（L1.6） |
| 7 | v_tile 提前 load | 个别百分点 | 总是可试 |
| 8 | grid-split FlashDecoding | 最高 1.25× | 仅大页 decode 且 grid≤4 |

### 陷阱 / 证伪方向

**证伪方向全表要点**（§5.2，不要重跑）：prefill GQA KV 复用不可行；kernel 内多页合并 / 行基址向量拼页 /
in-kernel split-KV → MLIR `PlanMemory Failed`；两阶段 skip-mask 实测 62.2µs > 单循环 always-load 54.5µs；
单循环内运行时 `if` → BLOCK_N=128 编译失败；num_stages 被忽略；grid≥8 的 grid-split 有害；
p hi+lo 二段拆分（golden 带 cast 链时）前提反转且实测 -49%；fp32 PV dot"更精确"= 不可复刻；
回退单轮 online 超 MERE；放大 BLOCK_BS 超页大小是精度红线；三参数 dot / num_warps / compile_hint 均无收益；
prefill BLOCK_M=128 最优。

**陷阱表**：

| 现象 | 根因 | 处理 |
|---|---|---|
| 只看总 Duration 就下"Cube 墙 / HBM 峰值 / 无法优化"结论 | 未看 mac/vec/mte2 分核占比，向量核/标量开销瓶颈被误诊为硬件极限（曾有形态因此错过 19×，0.0436×→0.8281×） | 分核诊断（profiler-core-breakdown 铁律，见类别卡天花板估算） |
| BLOCK_N > PAGE_SIZE | 跨页补零被 keep mask 当真实 key | L1.2 |
| `grid = (num_q_heads, batch)` | 同组 q-head 重复加载 KV（G× 流量） | L1.4 |
| verify 失败集合**严格按 dtype 划分** | aclnnMuls 在 fp16 下把标量先舍入到 fp16（RNE），Triton 标量走 fp32 路径 | L1.11 `_scale_for_dtype` |
| online softmax"看起来更精确"却超 MERE 阈值 | golden 的 p 在归一化后才 `.to(v.dtype)`，online 的 p 在运行 max/未归一化尺度舍入 | L1.9 两轮 KV 循环 + 判定顺序 |
| G=1 case 全面偏慢 | dot M=1 pad 到 16 行，15/16 Cube MAC 浪费且无 KV 复用收益 | G==1 向量通路分派 |
| 两轮结构轮 2 每页整 tile 除法拖慢 | 循环不变量 `l_safe` 未外提 | inv_l 预计算 |
| NS=1 的 case speedup 上不去还在反复调 tile | GQA tasks = KH < 核数，无合法并行扩展（拼页 PlanMemory Failed） | 弱 case 结构约束 |
| npu.Event 计时严重偏大 | 含 ~3× host dispatch 开销（实测 prefill BLK=256 下 Event 186µs vs kernel-only 54µs，3.4×） | §7 kernel-only 口径 |
| 算术 mask 占向量核与 softmax 竞争 | 比较与按位与走向量核 | L1.5 aux_mask 表 load（走 mte2） |
| 从零重写首版大幅劣化 | explicit-ptr masked load + 算术 mask，流水线未触发 | L1.1 block_ptr-in-helper + L1.7 op 顺序 |
| 拆两段循环"省掩码"反而变慢 | 破坏单循环 auto-multi-buffer | L1.8 保持单循环 always-load |
| 以为调 num_stages 能加速 | 本工具链 bishengir 忽略 num_stages | 改 op 顺序触发（L1.7） |
| 误判目标加速比可达而反复尝试 | 没看分核，未识别计算下限/带宽峰值/结构约束 | 天花板估算 |

**测量口径**：必须 kernel-only（`device_perf_npu` / `kernel_details.csv` 本 kernel `Duration(us)` 均值，
`Name` 过滤 `aclnn` 开头行）；禁止 op-level `npu.Event`。framework 参照系：paged decode 的 golden 通常
是逐 seq Python 循环 + `.item()` 同步 + `repeat_interleave` 物化 ⇒ 小 case 恒 60~130µs 起步，speedup
数字天然偏大——评估优化收益看 impl 绝对时间 / geomean 变化，不要被单 case 大 speedup 带偏。simulator
占比 ≠ 端到端收益，终判以 benchmark 为准。
