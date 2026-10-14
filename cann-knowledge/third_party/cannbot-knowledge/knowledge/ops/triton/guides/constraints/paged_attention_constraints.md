---
type: Guide
title: paged_attention 类算子设计约束（分页 KV Cache Attention）
description: KV 经 block_table 页表间接寻址、跨页不跳块的分页注意力（decode / prefill）算子的判别特征、通用经验与 Layer
  1 设计硬约束（含 golden 舍入链精度契约、性能数字与证伪方向）。
tags: [task.design, topic.attention, topic.implementation.masking, topic.compute.normalization, topic.precision, topic.data_movement.memory_access]
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

# paged_attention 类算子设计约束（分页 KV Cache Attention）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；
> attention 家族共享底座与写法类路由见 [flash_attention_constraints.md](flash_attention_constraints.md) /
> [attention_writing_classification.md](attention_writing_classification.md)；
> GQA 任务划分、页表标量 load 红线、scores 掩码（G6）、tile·UB 预算见 [gqa_constraints.md](gqa_constraints.md)；
> 页表寻址 **＋ 块级选择**（topk/CSR/块掩码）叠加形态见 [block_sparse_attention_constraints.md](block_sparse_attention_constraints.md)。

本卡负责 **二·3 页表寻址形态（不跳块）**：KV 经 `block_table` 间接寻址、页与页在显存中不连续，
但**整段 KV 都参与 softmax（无块选择）** 的 decode / prefill 全部形态。

> ⚠️ **核心优化哲学：先分形态再动手。** decode 大页是**内存受限**（优化方向是减 HBM 流量、提并行），
> 批量小页 decode 是**每页固定开销受限**（优化方向是**减少页迭代数**，加大 tile 内计算无效），
> prefill 是**计算受限**（核心是"让流水线重叠生效"而非"减流量"）。**三者目标函数不同，
> 禁止跨形态套用**（典型反例：把 decode 的 GQA-grouped 减 KV load 套到 prefill，实测不可行，见证伪表）。

- **§0 判别特征（四问 + 子形态分类）** · **§1 通用经验 S1-S4** · **Layer 1 设计约束 L1.1-L1.12**
- **证伪方向（§5.2 全表要点 + 天花板估算）**

> **证据基础**：三个分页形态算子的完整探索轨迹——`paged_prefill_kernel`（因果 GQA，**计算受限**，
> 重写持平 baseline 即可达上限）、`_swa_paged_decode_kernel`（SWA/GQA 单序列大页，**内存受限**，
> geomean 硬上限 ~0.89×）、`paged_attention_v2_decode_kernel`（批量 GQA decode，vLLM 风格小页 +
> golden 舍入链，**每页固定开销受限**）。数字取自 Ascend950PR（prefill / 大页 decode）与
> Ascend910B2（批量小页 decode）实测。

## 本类别的算子级实现手册

- [`operators/paged_attention__paged_kv_cache_attention.md`](../operator_playbooks/paged_attention__paged_kv_cache_attention.md) — Paged KV Cache Attention（decode / prefill）三子形态的 Layer 2 骨架、Layer 3 关键技巧、性能基准与陷阱/证伪（命中本类算子时必须一并加载）

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. KV 经 `block_table[seq][logical]` 间接寻址（`phys = block_table[...][...]` → `K = cache[phys]`），
   页与页在显存中不连续，但**不跳块**——整段 KV 都要算；
2. 参考实现里有 `cache[block_indices].view(-1, KH, D)` / 按页表 gather 页再 reshape 的取数写法；
3. SWA/窗口死区的**整页跳过属算术区间收缩**（只收缩循环范围，不做块选择），不算跳块，仍用本文件。

有块级选择结构（块掩码 / CSR / topK 选块）→ `block_sparse_attention.md`；
主链为矩阵吸收（`o = softmax(s) @ ckv`）→ `mla.md`；
KV 是连续区间、仅 head 共享 → `gqa.md` §3.2 ＋ `flash_attention.md`。

### §0.2 ★ 形态识别四问（Phase 2 第一步必须回答）

| # | 问题 | 影响 |
|---|---|---|
| **Q1** | decode 还是 prefill？query_len 是否为 1？ | 定子形态（批量小页 decode / 大页 decode / prefill），三套目标函数 |
| **Q2** | golden 的 cast 链长什么样——有没有 `einsum(...).float()`、`softmax(...).to(dtype)`、`x * scale` 这类显式 dtype 边界？ | 决定精度契约：**有显式 cast ⇒ 复刻舍入链、两轮 KV 循环**（L1.9）；无 ⇒ 单轮 online |
| **Q3** | G = H/KH 是 1 还是 ≥2？ | G==1 走向量通路 kernel（算子手册 §4.5），G≥2 走 Cube dot 通路（host 一个 if 分派） |
| **Q4** | 页大小 BS 是多少、是否 2 幂？ | `BLOCK_BS ≤ BS 且整除`是精度红线（L1.2）；BS 全 2 幂时直接取 `BLOCK_BS = BS`，页内行掩码整体消除 |

### §0.3 子形态分类表

| 子形态 | 场景特征 | 计算特征 | 优化哲学 |
|---|---|---|---|
| 大页 decode | SWA/GQA，页大、序列少 | M=1（GQA 组 pad BLOCK_M≥16），**内存受限** | GQA-grouped grid + block_ptr DMA + SWA 死区跳页 |
| 批量小页 decode | 批量 NS 条序列，GQA+滑窗，页小（BS=16/32/64），golden 带 cast 链 | M=G≤8，**每页固定开销（SCALAR pipe）受限**，MMAD 个位数占比 | persistent 任务循环 + **两轮 KV 循环（精度契约）** + G==1 向量通路分派 |
| prefill | 因果 GQA，M=BLOCK_M 多 query 行 | **cube 计算受限** | split-q persistent + helper 内 block_ptr 触发 auto-multi-buffer 流水线 + aux_mask 表 load |

## §1 通用经验（跨形态，首次生成必须遵守）

### S1 ★ 先用 Q1 定形态，禁止跨形态套用优化方向

三个子形态的瓶颈画像不同（内存 / 每页固定开销 / 计算），一个形态的头号杠杆在另一个形态上
可能是证伪项（见证伪表）。Phase 2 草图必须先写明子形态与目标函数，再选骨架。

### S2 性能度量必须 CANN profiler kernel-only

op-level `npu.Event` 紧凑 Python 循环含 ~3× host dispatch 开销，对 ~10-54µs 级 kernel 严重失真
（实测 prefill BLK=256 下 Event 报 186µs vs kernel-only 54µs）。

### S3 num_stages 在本工具链被忽略，勿浪费时间

decode/prefill 侧 2/3/4 均无变化。流水线的正确触发方式是 helper 内 op 顺序（L1.7），不是 num_stages。

### S4 杂项工程习惯（实测踩过）

- masked load 的 `other` 不可靠：掩码 lane 用 `tl.where(mask, x, 0.0)` 显式清零后再参与运算。
- `kv_len_cap = max_blocks_per_seq * block_size` 在 host 算好传标量，kernel 里 `min(kv_len, cap)` 防页表越界。
- q/out 直接按目标布局落盘，host 侧禁止 `permute(...).contiguous()`（`gqa.md` L1.8）。
- 核数获取：cube = `torch.npu.get_device_properties(0).cube_core_num`；
  vector = `triton.runtime.driver.active.utils.get_device_properties(dev)["num_vectorcore"]`。

## Layer 1 设计约束（首次生成就要全部满足）

### L1.1 ★★★ K/V 加载：block_ptr（或标量页基址的 affine tile load），禁止向量基址跨页 gather

- **必须**: 分页 K/V cache 的加载用 `tl.make_block_ptr` + `boundary_check=(0,1), padding_option="zero"`
  （页内连续布局 → DMA），再喂给 `tl.dot`。prefill 侧还要求 K/V/Q/O 全部 block_ptr 且 KV load 写在
  helper 内部（L1.7）。
- **禁止**: `tl.load(ptr + 向量页基址[:,None]*stride + ...)` —— 页基址是向量的跨页 gather，
  bishengir 把 dot 逐出 Cube、访存不走 DMA。
- **Why**: 实测（decode）offset-load `mac 1.7% / vec 91% / mte2 0.1%` → block_ptr
  `mac 17-27% / mte2 35-69% / vec 51-73%`；geomean **251.8µs → 17.6µs**。实测（prefill）同工作量下
  explicit-ptr 版 90.7µs（重叠效率 1.12×）→ block_ptr+helper 版 **54.5µs（重叠效率 1.72×）**；
  mac/mte2 工作量相同，差异全在软件流水线。
- **边界**: 红线的本质是「**页基址是向量**」。当 tile 严格不跨页（BLOCK_BS ≤ PAGE_SIZE）且页基址是
  **标量**时，`tl.load(k_ptr + 标量页基址 + 行距 + 列距)` 的 affine 2D offset load 也能保住 dot 上
  Cube（MMAD 正常发射）。block_ptr 版 MTE2 占比更优，两种写法都合规，按页大小与形态选。

### L1.2 ★★ BLOCK_N ≤ PAGE_SIZE 且整除（精度红线）

- **规则**: `BLOCK_N = min(128, next_pow2(PAGE_SIZE))`，保证 `PAGE_SIZE % BLOCK_N == 0` → tile 完全
  落在单页内，每个 KV tile 只查一次 block_table。
- **禁止**: BLOCK_N > PAGE_SIZE —— block_ptr 按 `shape=(PAGE_SIZE,…)` 对跨页行补零，但 keep mask
  按 `j < kv_seq_len` 判定，会把补零产生的 `qk=0` 当作真实 key（应为 -inf）→ 精度炸。
- **UB**: block_ptr 连续 BN=128 ≈92KB（OK）；gather 跨页 BN=128 触发 `ub overflow`（需 3.0M bits >
  1.77M）。
- **How**: paged 寻址 `logical_page = kv_start // PAGE_SIZE`；`in_page = kv_start % PAGE_SIZE`；
  `phys = tl.load(block_tables + b*stride_bt_batch + logical_page*stride_bt_block)`。批量小页 decode
  直接取 `BLOCK_BS = BS`（BS 全 2 幂时恒整除）：granule 恰好一页，页内行掩码整体消除，唯一掩码点
  是 scores 位置掩码。

### L1.3 GQA head 映射（强制，注意两方向互逆）

- decode（program 按 kv_head **反推**组内 q-head）: ABAB `head_idx = kv + r*nkv`；AABB
  `head_idx = kv*ratio + r`（r=arange(BLOCK_M)，mask r<ratio）。
- prefill（task 按 q_head **正推** kv_head）: ABAB `kv_head = q_head % NUM_KV_HEADS`；AABB
  `kv_head = q_head // (NUM_Q_HEADS//NUM_KV_HEADS)`。
- 批量 decode（`k[page, row, kv_head, d]` 布局，q 头按 kv 头分组连续排布）: `head = kh * G + g`
  （AABB 语义，g∈[0,G)）。

### L1.4 ★★★ 任务划分：decode 按 (kv_head, batch) 分组；prefill 按 split-q persistent

- **decode（G≥2）**: `grid = (num_kv_heads, batch_size)` 或 persistent 任务循环
  （task=(seq, kv_head, gb)，`grid = min(核数, tasks)` + 核内步长循环）。每个 program 加载分页 KV
  **一次**，用 `tl.dot` 一次性算出同组 G 个 q-head（pad BLOCK_M≥16）。
- **禁止**: `grid = (num_q_heads, batch)` —— 同组 q-head 各自重复加载同一 kv-head 的 KV（G× 冗余流量）。
- **边界（G==1）**: MHA 时分组无 KV 复用收益（每 kv_head 只服务 1 个 q head），dot 的 M=1 pad 到
  16 行浪费 15/16 Cube MAC——必须走**向量通路分派**（算子手册 §4.5），不走本条。
- **prefill**: `grid = (cube_num,)`（按芯片取，如 950PR cube_num=28），kernel 内轮询领取
  `(batch, q_block, q_head)` 任务。禁止把任务全展开成 grid（粒度太细、launch/sync 开销高）。
- **prefill 禁 GQA KV 复用**: GROUP 头共享 K/V 减 load 不可行——顺序处理多组头寄存器压力过大，
  融合进 M 维需慢 gather load；且 prefill 是计算受限，减 load 无效（证伪表）。

### L1.5 ★★ 因果掩码用 operator 下发的 aux_mask 表 load（prefill）

- **必须**: 因果掩码用下发的下三角 `aux_mask` bool 表，按 `offset = clip(kv_start - q_abs_start, …)`
  索引 `tl.load`。该 load 走 **mte2（不占向量核）**。
- **禁止**: 算术掩码 `mask = (col_idx[None,:] <= q_abs[:,None]) & col_valid & row_valid` —— 比较与
  按位与占**向量核**，与 softmax 的 exp/max/sum/where 争夺向量单元。
- **Why**: 实测算术 mask 0.84× → aux_mask load 1.00×。掩码成本从向量核挪到 mte2，被流水线重叠掩盖。

### L1.6 ★ SWA：大页 decode 死区整页跳过；query_len=1 折叠成区间收缩

- **死区跳页（大页）**: 页粒度遍历 KV；整页落在死区 `[global_window, kv_seq_len-1-local_window)`
  则跳过（不加载 K/V）。判据 `page_in_global = page_start < global_window`；
  `page_in_local = page_end > kv_seq_len-1-local_window`；二者皆假则 skip。
- **Why**: 长序列（kv=8192, local=1023）只加载 ~1024 keys 而非全量。
- **区间收缩（批量小页，query_len=1）**: `kv_start = max(kv_len - sw, 0)`，KV 循环从
  `kb0 = kv_start // BLOCK_BS` 起——golden 的 triu 掩码在 query_len=1 时等价于这个区间，直接
  **减少页迭代数**，无逐页分支。
- **How**: 位置比较走 fp32（`(pos + offs).to(f32)` 与 `kv_len.to(f32)` 比）：int 比较会触发标量降级。

### L1.7 ★★ helper 内 op 顺序决定流水线（prefill 核心）

- **必须**: 把单 KV block 的 K/V load + softmax + dot 封进 helper，且 op 顺序为：**早 V-load →
  `tl.math.exp` → `propagate_nan` → 分离的 scale / sub → 3-arg 累加 dot `tl.dot(p, v, acc)`**。
  只有这个顺序触发 bishengir auto-multi-buffer。
- **禁止**: inline 展开、或把 V-load 放在 softmax 之后（晚 V-load）—— 流水线不触发。
- **Why**: bishengir 的 auto-multi-buffer pass 对 op 调度顺序敏感；早 V-load 让 V 的 DMA 与
  QK→softmax 的标量/向量计算重叠，形成 K/V 双缓冲。实测 82µs → **64.7µs**。

### L1.8 ★★ 禁止破坏单循环流水线的"优化"（prefill）

- **禁止①**: 单循环内运行时 `if` 跳过 full-block 的 mask —— BLOCK_N=128 触发 bishengir 编译失败。
- **禁止②**: 两阶段 skip-mask / skip-load（把循环拆成"整页跳掩码"和"边界页算掩码"两段）——
  破坏单循环流水线，实测省下的 mask load 抵消不了流水线损失（62.2µs > 单循环 always-load 54.5µs）。

### L1.9 ★★★ golden 舍入链契约：p 必须在归一化后舍入 ⇒ 两轮 KV 循环（批量 decode 精度硬门槛）

- **契约**: golden 的 p 是 `softmax(attn).to(v.dtype)`——softmax（全局 max + 归一化）**之后**才舍入
  到 v 的 dtype。kernel 必须复刻这个顺序：先拿到终态 `m_g/l_g`，再算 `p = exp(s - m_g)/l_g`（fp32）
  → `.to(native)` → PV dot。
- **为什么单轮 online 不行**: online softmax 的 p_eff 在**运行 max、未归一化尺度**上舍入，与 golden
  的舍入尺度不同 ⇒ diff ≈ golden 自身 p 舍入噪声，bf16 下足以超 MERE 阈值（7.81e-3）。
- **结构（两轮，单 kernel 内）**:
  - 轮 1（stats）: online 求 m_g/l_g —— QK dot + round-trip + scores 掩码 + max/exp-sum；
    无 V load、无 PV dot（max 精确 ⇒ 终态 m 逐位 ≡ golden 的 m_g）。
  - 轮 2（apply）: 重算 QK scores（与轮 1 同一舍入链，逐位一致），
    `p = exp(s - m_g) / l_g` → `.to(native)` → 单次 PV dot 累加；m 固定 ⇒ 无 rescale；
    出口不再除 l（p 已归一化）。
- **代价账**: 每 granule 3 个 dot（2×QK + 1×PV）vs online 2 个 dot；UB 反而更省（无 v.to(f32) 大缓冲、
  无 rescale 的 acc 重写）。**精度是硬门槛，性能账在 Phase 4 再算**——不要为省一个 dot 回退单轮。
- **禁止项**: ① 回退 online 单轮；② fp32 PV dot（"更精确"= 不可复刻）；③ p_hi/p_lo 二段拆分
  （前提反转，见证伪表）。

### L1.10 ★ einsum round-trip + 标量乘法同序舍入

- golden 的 `torch.einsum('qhd,khd->hqk', q, k).float()` 输出**先落 q/k 的原生 dtype 再升 fp32**。
  kernel 复刻为 `scores = tl.dot(q, tl.trans(k)).to(native).to(tl.float32)`——少一个 round-trip 都会
  在 MERE 上显形。**两轮循环里的 QK 必须走完全相同的舍入链**（重算而非缓存，保证逐位一致）。
- `scale` 乘在 q_tile 上（每 task 一次、KV 循环外）：`(q_tile * scale).to(native)`，复刻 golden 的
  `query * scale` 同序舍入（标量本身按 L1.11 舍入）。

### L1.11 ★★★ golden 的标量乘法在 NPU 上有 dtype 舍入——host 侧必须 dtype-aware 复刻

- **现象指纹**: verify 失败集合**严格按 dtype 划分**（全部 fp16 case 失败、bf16 全过）⇒ 不是精度不够，
  是**算术路径不匹配**，别去调 tile / 累加器精度。
- **根因**: golden 的 `tensor * python_float` 在 NPU 上走 `aclnnMuls`，**fp16 下把标量先舍入到 fp16
  （RNE）再算**；Triton kernel 的标量参数走 fp32 路径。两条路径的差在 fp16 判据阈值（9.766e-04）
  量级——其余舍入链没对齐时被掩盖，**对齐之后才显形**。bf16 的 aclnnMuls **不**预舍入标量
  （fp32 路径本就 bitexact）。
- **修复**（host 侧按 dtype 舍入标量后再传 kernel）:

```python
def _scale_for_dtype(scale, dtype):
    """复刻 aclnnMuls 的标量处理：fp16 先舍入到 fp16（RNE），bf16/fp32 保持 fp32。"""
    if dtype == torch.float16:
        return struct.unpack("e", struct.pack("e", scale))[0]
    return scale
```

- **泛化**: 适用于任何 golden 含 `tensor * float` 的算子（不限于 attention）。排查顺序排在舍入链
  复刻（L1.9/L1.10）之后。

### L1.12 ★ scores 上的位置掩码用有限极小值，禁 -inf

- KV 维 padding/越界必须掩在 **scores** 上（`tl.where(valid, s, -3.0e38)`），m 初值用有限极小值
  （-1.0e30）而非 -inf，出口除法 guard（`l>0`）。完整论证见 `gqa.md` G6——清零操作数会把 padding
  列的 score 变成 0，`exp(0 - m)` 是非零的，l 被灌进虚假质量。

## 证伪方向（§5.2 全表要点，不要重跑这些死路）

| 方向 | 结果 |
|---|---|
| **prefill 的 GQA KV 复用**（GROUP 头共享 K/V 减 load） | ⛔ 不可行——顺序处理多组头寄存器压力过大，融合进 M 维需慢 gather load；且 prefill 计算受限，减 load 无效 |
| **kernel 内多页合并 / 行基址向量拼页 / in-kernel split-KV** | ⛔ MLIR `PlanMemory Failed`（`gqa.md` 三种写法全灭）。扩展并行只允许双 kernel split+reduce 形态 |
| **两阶段 skip-mask / skip-load**（拆"整页跳掩码"+"边界页算掩码"） | ⛔ 破坏单循环流水线，实测 62.2µs > 单循环 always-load 54.5µs（L1.8） |
| **单循环内运行时 `if` 跳过 full-block mask** | ⛔ BLOCK_N=128 触发 bishengir 编译失败（L1.8） |
| **num_stages 2/3/4** | ⛔ 被忽略，无变化（S3） |
| **对 grid≥8 的 case 用 grid-split** | ⛔ tile-work 守恒，split 开销 > 收益（grid=8 → 0.97×，grid=16 → 0.78×） |
| **p 的 hi+lo 二段拆分（golden 自带 cast 链时）** | ⛔ 前提反转：golden 自己舍入 p 时，二段拆分制造一条 golden 里不存在的"更干净"路径，语义不可复刻；实测 -49% |
| **fp32 PV dot（golden 自带 cast 链时）** | ⛔ 同上，"更精确"= 不可复刻（L1.9） |
| **回退单轮 online（golden 自带 cast 链时）** | ⛔ p 舍入尺度不同，diff ≈ golden 自身舍入噪声量级，超 MERE 阈值（L1.9） |
| **放大 BLOCK_BS 超过页大小** | ⛔ 页界上限（L1.2），且跨页补零是精度红线 |
| **三参数 `tl.dot(p,v,acc)` / `num_warps` 调参 / `tl.compile_hint("tile_cube_loop")`** | ⛔ 无收益 / 无变化 / 本版本 `tl` 命名空间无此 API |
| **BLOCK_M 扫描（prefill）** | 128 最优（64 变慢、256 寄存器压力）；PROGS∈{28,56,84} 无影响 |

> **天花板估算（先分形态算账，再动手）**:
> **prefill（计算受限）**：分核画像 mac 0.464 / mte2 0.466 / 重叠 1.72×——cube_mac（~25µs）是硬下限、
> 总 54.5µs；要 1.2×（45µs）需 mac_ratio ≥ 0.56（不减 MAC 下重叠再提 ~20%），本工具链不可行，
> **从零重写持平 baseline（geomean 1.0× 量级）即可达上限**，应诚实报告 infeasible 而非编造。
> **大页 decode（内存受限）**：baseline 已近 HBM 峰值（~18µs/case ≈ 16MB @ ~1.3TB/s）时 geomean
> 硬上限 ~0.89×；真正卡目标是 vec 核 softmax（exp×2/max/sum/where/除法；分核 mac ~7% / vec 50-62% /
> mte2 ~35%），BN=128 已是 UB 上限无法靠增大 tile 减轮数；softmax 是数学必需，无 codegen 辅助时无法与
> Cube/DMA overlap。**批量小页（每页固定开销受限）**：排除法确诊 **SCALAR pipe 每页固定开销主导**
> （页表标量访存 + 循环控制 + CV flag）；压缩手段全被结构约束封死（页表标量 load 禁向量 gather /
> 两轮结构是精度契约 / BLOCK_BS=BS 页界），此时继续减访存或调 tile 无用，可动的只有减少页迭代数
> （滑窗收缩）与循环不变量外提。
> **弱 case 结构约束**：NS=1 时 GQA tasks = KH < 核数，核没用满且无合法并行扩展——speedup 上限受
> 任务数约束，属形态结构约束，不要在 tile/流水线上反复消耗轮次。
