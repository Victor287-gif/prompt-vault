---
type: Guide
title: transformer_training 类算子设计约束
description: transformer_training 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.attention, topic.precision, topic.data_movement.layout_transform, topic.compute.normalization, topic.implementation.masking]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# transformer_training 类算子设计约束

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## Transformer 训练类算子优化经验

本文档合并 Transformer 训练类算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 FusionAttention**（attention，标准稠密 MHA / FlashAttentionScore）
- **§3 常见陷阱**（按算子分小节）
- **§4 相关记忆**

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/transformer_training__fusionattention.md`](../operator_playbooks/transformer_training__fusionattention.md) — FusionAttention 算子（attention）

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| FusionAttention | `attention` | 标准稠密 MHA，Q/K/V 同构，支持 BSH/BSND/BNSD/SBH，可选 pse/sink/mask/keep_prob | 多 layout / 多参数下的精度优先 + 延迟归一化 + 单 pass softmax |

> ⚠️ **关键区分**：Transformer 训练类算子的核心难点通常是**多 layout、多参数下的精度对齐**，其次才是性能。必须先保证全量精度通过，再谈优化；训练场景对中间态 dtype 舍入和 dropout RNG 行为极为敏感。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束是 Transformer 训练类算子**共有**且**未在其他 template 文件覆盖**的工程约束。其他文件已提取的通用约束（如 GC-1）此处不再重复，各算子章节引用时标注。

### G1 精度基准优先使用 `torch_npu` 原生 golden

- **必须**优先使用 `torch_npu` 对应原生算子作为 golden，其内部已对齐 CANN 实现。
- **必须**在 `Model.forward` 中对特殊参数（如 `keep_prob`、`dropout` 相关概率）先做 `clamp(min=0.0, max=1.0)`，避免 NPU dropout RNG offset 自增导致失败。
- **Why:** 训练类算子常含 dropout / mask / scale 等语义，PyTorch 原生实现与 NPU 原生实现在 RNG 路径、填充值、归约顺序上可能存在系统性差异。

### G2 `tl.dot` 必须按输入精度执行

- fp16 / bf16 输入时，`tl.dot` 必须直接以 `tl.float16` / `tl.bfloat16` 输入（`allow_tf32=False`），累加器用 `tl.float32`。
- **禁止**先将输入 cast 到 fp32 再 `tl.dot`。
- **Why:** 提前升精度会改变累加舍入，在极端 score 量级下产生 ~1e-2 偏差，经 exp 放大后 softmax_sum 超 fp32 阈值。

### G3 延迟归一化与单 pass softmax

- **必须**按 `E = exp(score - row_max)` 落盘，`output = (E @ V) / softmax_sum` 的顺序计算 PV。
- **禁止**先归一化 `probs = E / softmax_sum` 再 `probs @ V`。
- softmax 的 `max / sum / exp` 必须合并为对同一行 score 向量的单次加载。
- **Why:** 延迟归一化天然兼容 sink、全 masked 行等边界；三 pass 实现会引入额外 GM 往返和精度差异。

### G4 mask 填充值使用 `-FLT_MAX`

- mask 填充值必须是 `-FLT_MAX`（约 `-3.4028234663852886e+38`），**禁止**使用 `float('-inf')`。
- **Why:** `-inf` 在部分后端会被特殊处理，导致 exp 行为不一致；`-FLT_MAX` 是稳定可复现的 masked 值。

### G5 多 layout 输入在 host 侧统一，禁止强制 `.contiguous()`

- BSH/BSND/SBH 等 layout 应统一在 host 侧转为 BNSD 后再传入 kernel；kernel 内只处理一种 layout，输出前再转回原始 layout。
- layout 转换使用 `transpose` / `view` / `permute` 即可，**不要**调用 `.contiguous()`。
- **Why:** 无意义的 host 侧拷贝会显著增加 framework 延迟（实测去 `.contiguous()` 可提升 ~8% geomean）。

### G6 grid 并行度使用实际物理核数

- cube-heavy kernel 用 `cube_core_num`，vector-heavy kernel 用 `vector_core_num`，禁止硬编码核数。
- **Why:** 训练类算子 shape 变化大，硬编码核数会导致小 shape 时空跑、大 shape 时负载不均。

---

## §3 常见陷阱与避免方法

### §3.1 FusionAttention 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 忽略 keep_prob clamp | NPU dropout RNG offset 自增 | L1.1 |
| Q/K 先 cast fp32 再 dot | 累加偏差经 exp 放大 | L1.2 |
| 先归一化 probs 再 @V | MERE 必然超标 | L1.3 |
| 三 pass softmax | 额外 GM 往返 + 精度差异 | L1.4 |
| 使用 `float('-inf')` 填充 mask | 后端 exp 行为不一致 | L1.5 |
| pse 未处理 broadcast | 触发 `MemRefLayoutAttrInterface` assert | L1.6 |
| host 侧强制 `.contiguous()` | 增加 framework 延迟 | L1.7 |
| 期望轻松达到 1.0x | 当前天花板 0.9627x（实验）/ 0.8553x（归档） | L1.12 |

---

## §4 相关记忆

- [[fusion-attention-golden-precision-paths]] — 精度对齐三条硬约束
- [[fusion-attention-optimization-record]] — 当前最佳版本与迭代记录
