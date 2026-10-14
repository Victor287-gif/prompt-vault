---
type: Guide
title: sampling 类算子设计约束（核采样过滤）
description: top-p/核采样过滤算子的判别特征与设计硬约束：索引与 torch.topk 精确一致、数值归约 fp32、移除规则闭式公式、forward
  内禁 PyTorch 回退。
tags: [task.design, topic.compute.ordering, topic.compute.reduction, topic.precision, topic.compute.elementwise.arithmetic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-26T08:28:46Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---

# sampling 类算子设计约束（核采样过滤）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）。

适用范围：**Top-P / 核采样过滤**算子（锚定 `_top_p_filter_kernel`，op25）。
核心优化哲学：排序后 top-k 概率的**逐行双 pass softmax**（load → softmax → inclusive_cumsum → remove mask → 过滤 → 重归一），按 batch 行分 program；**最大性能杠杆在 wrapper 如何获取排序后的 top-k**，不在 kernel 内部。

> 性能参考：Ascend950PR_9579。几何平均加速比 **1.297×**（目标 1.2× **已达**）；精度全 case 通过（`sorted_topk_indices` 逐元素与 torch 参考比对）。

## 本类别的算子级实现手册

- [`operators/sampling__top_p_filter.md`](../operator_playbooks/sampling__top_p_filter.md) — Top-P Filter（核采样过滤）实现手册：Layer 2 逐行双 pass 骨架、Layer 3 自适应 topk 分派与性能基准（1.297×）。

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

1. 参考实现对 `logits[B, V]` 做 top-p / 核采样过滤：先取 top-k，softmax 后用 `top_p` 累积阈值决定保留/移除，再对保留概率重新归一化。
2. 输出含 `sorted_topk_indices`（int64/int32）——与 `torch.topk` 的结果**逐元素比对**。
3. 按 batch 行分 program、K 轴（≤1024）串行 cumsum 是自然粒度 → 命中本文件。

## §1 通用经验（跨算子，首次生成必须遵守）

- **top-k 保留给 torch**：wrapper 中保留 `torch.topk(logits, top_k, dim=-1)`，kernel 内不做自定义 top-k（L1.1）。
- **全程 fp32 归约**：logits 进 top-k 前 cast fp32，kernel 内 max/exp/sum/cumsum 保持 fp32（L1.2）。
- **移除规则用闭式公式**，勿自行推导等价式（L1.3）。
- **热路径零 PyTorch 回退**（L1.4）；**不要**把 K 维拆到多个 program。

## Layer 1 设计约束（硬性规则，sampling 通用）

- **L1.1 ★ 必须 索引与 `torch.topk` 精确一致**
  - **Why:** mojo_opset 精度测试将 `sorted_topk_indices`（int64/int32）与 torch 参考**逐元素比对**。
  - **How to apply:** wrapper 中保留 `torch.topk(logits, top_k, dim=-1)`，**不要在 kernel 内实现自定义 top-k**，除非能证明对测试分布输出顺序完全一致；`sorted=False` 时用 `sort`+`gather` 恢复顺序（见 playbook L3.3，bit-for-bit 等价）。

- **L1.2 ★ 必须 数值归约 fp32**
  - **Why:** fp16/bf16 中间精度导致 cumsum 累积误差使 remove mask 错位。
  - **How to apply:** logits 进 top-k 前由 wrapper cast 到 fp32；kernel 内所有 softmax 归约（max, exp, sum）保持 fp32。

- **L1.3 ★ 必须 移除规则与 torch 参考一致（闭式公式）**
  - **Why:** 对已排序 top-k 概率 `p` 及 inclusive cumsum `c`，`c - p` 即 exclusive cumsum；这一行闭式写法替代 torch 参考的三步 mask 操作，任何偏离都改变参考语义。**禁止**自行推导等价的 mask 公式（看似等价实则浮点舍入路径不同）。
  - **How to apply:**
    ```
    remove_i = ((c_i - p_i) > top_p) & (i >= min_tokens_to_keep)
    ```

- **L1.4 ★ 必须 forward 内禁止 PyTorch 回退**
  - **Why:** `F.softmax` / `masked_fill` 出现在热路径时，kernel 调用往返开销会吃掉所有性能收益。
  - **How to apply:** Triton kernel 必须完成 filter 和两次 softmax。

- **L1.5 BLOCK_K 补零为可选（非强制）**
  - **Why/How:** 提交版 wrapper 直接传 `TOP_K=top_k`，分配 `output_probs[B, top_k]`，kernel 内 `tl.arange(0, TOP_K)` 无需额外 mask；若选择 pad 到 next_pow2，加 `BLOCK_K: tl.constexpr` 和 `mask_k`，保持 wrapper / output tensor 一致。

## §2 Layer 2 算法骨架要点（逐行双 pass softmax）

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

- **按 batch 行分 program 是自然粒度**：cumsum 沿 K 轴向串行（已知 workload 中 K ≤ 1024），拆分 K 跨 program 会引入同步开销，收益为零；**不要**把 K 维拆到多个 program。

## §3 陷阱表 / 证伪方向

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 索引精度失败 | `sorted_topk_indices` 与 torch 参考逐元素比对 | L1.1 wrapper 保留 `torch.topk`；sorted=False 时 `sort`+`gather` 恢复顺序 |
| 概率精度失败 | remove mask 偏移公式错误 | L1.3 闭式公式，勿自行推导 |
| 1.2× 目标未达 | `torch.topk` 占总延迟大头 | 先用自适应 topk 分派（playbook L3.3），再考虑 fused kernel |
| `sorted=False` 反而更慢 | 大 k 下额外排序 k 个元素代价高 | 分支：`top_k >= 512` 时用 `sorted=True` |
