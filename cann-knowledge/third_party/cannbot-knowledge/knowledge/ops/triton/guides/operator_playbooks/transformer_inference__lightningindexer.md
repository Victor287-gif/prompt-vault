---
type: Guide
title: LightningIndexer 算子（topk-index） 实现手册
description: '算子类别: topkindex。'
tags: [task.implementation, topic.attention, topic.compute.ordering, topic.compute.indexing, topic.compute.composite, topic.data_movement.layout_transform]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---
# LightningIndexer 算子（topk-index） 实现手册

> 类别约束见 [transformer_inference_constraints.md](../constraints/transformer_inference_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §6 LightningIndexer 算子（topk-index）

**算子类别**: `topk-index`
**典型特征**: per-(B,S1) 对 S2 个 score 取 top-K；score = Σ_n w_n · ReLU(q_n @ k)；causal mask；BSND/PA_BSND/TND layout
**性能基准**: 54/54 pass，几何平均加速比 **0.6152x** vs torch_npu golden（当前纯 Triton 上限）

> ⚠️ **关键区分**：LightningIndexer 的核心瓶颈是 **decode/small-batch 场景下的 top-k 选择**（radix / sort / iterative argmax 的固定开销与并行度），而非 score 矩阵乘的 Cube 算力。

### §6.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 精度基准用 torch_npu golden，但 golden 内部已做 bf16 舍入

- 使用 `torch_npu.npu_lightning_indexer(..., return_value=True)` 作为 golden，但输出前会按 `(value, index)` 升序排序以解耦排列。
- native golden 内部 score 计算使用 bf16 累加并对 val 做 bf16 RNE 舍入；纯 PyTorch fp32 直接 `torch.topk` 会在 val 输出上差 0.05%–数千，导致 `|diff|<=1` 失败。
- 参考：[[lightning-indexer-triton-gen]]

#### L1.2 top-k 索引输出必须按 index 升序

- `torch_npu.npu_lightning_indexer` 返回的 `sparse_indices` 本身不一定有序；verify 框架按"排序后比集合"判定。
- 实现最后必须对输出做一次 stable argsort：`perm = torch.argsort(sparse_indices, dim=-1, stable=True); sparse_indices = torch.gather(...)`。

#### L1.3 top-k tie-break 必须对齐 native `(value, index)` 升序

- 在 iterative argmax / radix select 中，等值淘汰时必须淘汰 **index 最大** 的元素，才能保住小 index。
- 已验证代码模式：
  ```python
  is_min = active & (best_val == bmin)
  cur_idx = tl.max(tl.where(is_min, best_idx, -2147483648), axis=0)
  bslot   = tl.min(tl.where(is_min & (best_idx == cur_idx), kmax, 2147483647), axis=0)
  better  = (tmax > bmin) | ((tmax == bmin) & (tpos < cur_idx))
  ```

#### L1.4 输出值必须做 bf16 RNE 舍入

- `x.to(tl.bfloat16).to(tl.float32)` 在 Ascend Triton 上可能是空操作，不舍入。
- 必须用手写位运算实现 RNE：
  ```python
  bits = val.to(tl.int32, bitcast=True)
  sign = bits & -2147483648; mag = bits & 2147483647
  l = (mag >> 16) & 1
  mag_r = (mag + 32767 + l) & -65536
  val = tl.where(mag >= 2139095040, val, (sign | mag_r).to(tl.float32, bitcast=True))
  ```

#### L1.5 禁止用 torch.topk / torch.sort / torch.argsort 作为实现路径

- 这些 op 在 NPU tensor 上会调用原生 aclnn kernel，违反"纯 Triton"约束。
- 曾被判定为作弊并回退；参考：[[lightningindexer-hybrid-native-topk-reverted]]

#### L1.6 score workspace 布局固定为 `[B*S1, S2]`

- 行索引必须等于 `b * S1 + s1`；错位会导致 S1>1 或 B>1 时全错甚至设备流中毒。
- 禁止在 top-k 阶段重新压缩/重排 score 数据布局；tie-break 与原始 S2 扫描序耦合。

#### L1.7 当前 Ascend Triton 后端下 fused score+topk kernel 不可靠

- `_li_fused_argmax_kernel` 会触发编译器 SIGSEGV（后端 bug）。
- `_li_fused_score_topk_kernel` 报 UB overflow（需求 17–19 Mbits，可用 1.5 Mbits）。
- 实际运行路径会回退到 score + topk 两阶段；设计 dispatch 时**不要依赖 fused kernel 能跑通**。

#### L1.8 PA_BSND host gather 是当前推荐做法

- device-side gather 已验证无整体收益，部分 case 反而因新增 kernel 开销变慢。
- 在获得新的确定性证据前，保持 `key[block_table.long()].contiguous().view(B, S2, 1, D)`。

#### L1.9 0.8x / 1.0x 目标在当前纯 Triton 约束下不可达

- 已穷尽常规方向（host overhead、tile dispatch、device gather、lightweight topk、FND、block topk、radix 调参等）。
- 当前纯 Triton 最佳为 **0.6152x**（exp_b_bf16_dot，54/54）。 report 中必须标注此天花板。

### §6.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分派决策树

```python
# 1. layout conversion + contiguous
# 2. PA_BSND: host gather
# 3. dispatch:
if K == 1:
    score_kernel(...) + lightweight_argmax_kernel(...)
elif K <= 64 and R < num_cores:
    try_fused_score_topk(...)  # 大概率编译失败并回退
if not done:
    score_kernel(...)
    if S2 <= 2048 and K > max(256, S2//2):
        radix_topk_parallel(...)
    elif S2 <= 2048:
        tl.sort_topk(...)
    elif K <= 16:
        iterative_topk_small(...)
    elif K <= 64 and R >= num_cores:
        iterative_topk(...)
    else:
        radix_topk_parallel(...)
# 4. sort indices ascending + zero-fill return_value=False values
```

#### L2.2 Score kernel 骨架

```python
for task in grid:
    load k_tile[BLOCK_S2, D] bf16
    for s1_local in BLOCK_S1:
        score_row = 0
        for n_chunk in range(0, N1, 16):
            qk = tl.dot(q_chunk[16,D], tl.trans(k_tile))  # fp32
            qk = tl.maximum(qk, 0.0)
            score_row += tl.sum(w_chunk[:,None] * qk, axis=0)
        apply causal & col_ok mask -> -inf
        store score_row
```

#### L2.3 Radix select 骨架

```python
# Phase 1: minmax key (f2key)
# Phase 2: 3-pass histogram bisect threshold tau
# Phase 3: parallel select (>tau + ==tau by index scan order)
# tie-break: cumsum on eq elements within each chunk + per-row quota
```

### §6.3 Layer 3: 关键代码模式（参考，禁止复制变量名）

#### L3.1 decode score kernel 调大 BLOCK_S2

- 当 `S1 == 1` 时，可尝试 `BLOCK_S2 = 256`（S2≥4096）或 `512`（S2≥16384）。
- 已验证对 S2=32768 的 case 有显著收益；但需配合 topk 路径一起评估。
- 注意：仅调 score tile 不足以让整体突破，topk 仍是 decode 瓶颈。

#### L3.2 bf16 dot + fp32 accumulate

- load q/k as bf16，`tl.dot` 输出 fp32，再 relu / weighted sum。
- 这是 exp_b_bf16_dot 的 baseline 配置，是当前最稳路径。

#### L3.3 iterative top-k 模式（K 较小）

- 维护 `best_val[MAX_K]` / `best_idx[MAX_K]`。
- 每 tile 用 iterative max-extraction 更新 running top-K。
- tie-break 严格按 L1.3。

---
