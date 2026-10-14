---
type: Guide
title: Top-P Filter 算子（核采样过滤） 实现手册
description: '典型特征: 逐行双 pass softmax over 排序后 top-k 概率，按 batch 行分 program'
tags: [task.implementation, topic.compute.ordering, topic.compute.reduction, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-26T08:28:46Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---

# Top-P Filter 算子（核采样过滤） 实现手册

> 类别约束见 [../sampling_constraints.md](../constraints/sampling_constraints.md)（L1.1~L1.5 硬约束在此），
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §1 Top-P Filter 算子（sampling）

**算子类别**: `sampling`　**典型特征**: 逐行双 pass softmax over 排序后 top-k 概率，按 batch 行分 program。
**性能基准**: 全 case 精度通过（`sorted_topk_indices` 逐元素与 torch 比对）；几何平均 **1.297×**（目标 1.2× **已达**），Ascend950PR_9579。

### §1.1 Layer 1 设计约束（算子专属）

- **L1.1 ★** 索引与 `torch.topk` 精确一致：wrapper 保留 `torch.topk`，kernel 内不自实现 top-k；`sorted=False` 用 `sort`+`gather` 恢复顺序。
- **L1.2 ★** 数值归约全程 fp32（进 top-k 前 cast；kernel 内 max/exp/sum/cumsum fp32）。
- **L1.3 ★** 移除规则用闭式 `remove_i = ((c_i - p_i) > top_p) & (i >= min_tokens_to_keep)`，勿自行推导。
- **L1.4 ★** forward 热路径禁 `F.softmax` / `masked_fill`。
- **L1.5** `BLOCK_K` 补零可选（非强制）：提交版直接 `TOP_K=top_k` + `tl.arange(0, TOP_K)`。

### §1.2 Layer 2 算法骨架（逐行双 pass softmax）

```
grid = (batch_size,)

每个 program（一行）:
    offsets = tl.arange(0, TOP_K)
    load sorted_logits[TOP_K]
    p = softmax(sorted_logits)            # pass 1: logits → 概率
    c = inclusive_cumsum(p)
    remove = ((c - p) > top_p) & (offsets >= min_keep)
    filtered = where(remove, filter_value, sorted_logits)
    out = softmax(filtered)               # pass 2: 过滤后重新归一化
    store out[TOP_K]
```

- **按 batch 行分 program 是自然粒度**：cumsum 沿 K 轴向串行（K ≤ 1024），拆分 K 跨 program 引入同步开销、收益为零；**不要**把 K 维拆到多个 program。

### §1.3 Layer 3 关键技巧（可参考，不可照抄）

**L3.1 Kernel 签名（HEAD 版本，无 padding）**

```python
@libentry()
@triton.jit
def _top_p_filter_kernel(sorted_logits_ptr, output_ptr, top_p, filter_value,
                         min_tokens_to_keep, stride_logits_b, stride_logits_k,
                         stride_out0_b, stride_out0_k, TOP_K: tl.constexpr):
    pid = tl.program_id(0)
    row_logits_ptr = sorted_logits_ptr + pid * stride_logits_b
    row_out_ptr = output_ptr + pid * stride_out0_b
    offsets = tl.arange(0, TOP_K)
    logits = tl.load(row_logits_ptr + offsets * stride_logits_k)
    x_max = tl.max(logits, 0); e = tl.exp(logits - x_max); p = e / tl.sum(e, 0)
    c = tl.cumsum(p, 0)
    remove = ((c - p) > top_p) & (offsets >= min_tokens_to_keep)
    filtered = tl.where(remove, filter_value, logits)
    f_max = tl.max(filtered, 0); fe = tl.exp(filtered - f_max); f_p = fe / tl.sum(fe, 0)
    tl.store(row_out_ptr + offsets * stride_out0_k, f_p)
```

**L3.2 `@libentry()` 减少 per-call launch 开销**：小 B case（B=15/18）微弱正向收益，大 B case 影响可忽略。

**L3.3 自适应 topk 分派（1.2× 达标的关键杠杆）**：**最大性能杠杆不在 kernel 内部，而在 wrapper 如何获取排序后的 top-k**。Ascend 上 `torch.topk(..., sorted=True)` 内部全排序开销大，`sorted=False` 跳过排序，按 `top_k` 大小分支：

```python
if top_k >= 512:
    # 大 k：额外排序 k 个元素的开销 > 省下的排序 → 直接用 sorted topk
    sorted_logits, sorted_topk_indices = torch.topk(logits_f, top_k, dim=-1)
else:
    # 小 k：unsorted topk + 对小 k 个元素排序
    vals, idx = torch.topk(logits_f, top_k, dim=-1, sorted=False)
    sorted_logits, order = torch.sort(vals, dim=-1, descending=True)
    sorted_topk_indices = torch.gather(idx, dim=-1, index=order)
```

- 两条路径产生**完全一致**的降序 (values, indices)——topk 无论 `sorted` 取值都返回相同 k 个值；`sort`+`gather` 仅恢复顺序，**bit-for-bit 等价**。
- 分界点 512 是当前 shape（k ∈ {100, 1000}）的经验值；workload 的 k 分布变化时需重新测定。
- 配合 `sorted_logits` 原地写输出，几何平均从 ~1.14×（仅 sorted=True）提升到 **1.297×**。

### §1.4 性能基准 / 优化记录表（Ascend950PR_9579）

| 方案 | 几何平均 | 目标 1.2× |
|------|---------|-----------|
| 仅 `torch.topk(sorted=True)` | ~1.14× | 未达 |
| **自适应 topk 分派 + `sorted_logits` 原地写输出（L3.3）** | **1.297×** | **已达** |

- 精度：全 case 通过（`sorted_topk_indices` 逐元素比对）。

### §1.5 陷阱 / 证伪方向（不要重跑）

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 索引精度失败 | `sorted_topk_indices` 与 torch 参考逐元素比对 | L1.1 wrapper 保留 `torch.topk`；sorted=False 时 `sort`+`gather` 恢复顺序 |
| 概率精度失败 | remove mask 偏移公式错误 | L1.3 闭式公式，勿自行推导 |
| 1.2× 目标未达 | `torch.topk` 占总延迟大头 | 先用自适应 topk 分派（L3.3），再考虑 fused kernel |
| `sorted=False` 反而更慢 | 大 k 下额外排序 k 个元素代价高 | 分支：`top_k >= 512` 时用 `sorted=True` |
