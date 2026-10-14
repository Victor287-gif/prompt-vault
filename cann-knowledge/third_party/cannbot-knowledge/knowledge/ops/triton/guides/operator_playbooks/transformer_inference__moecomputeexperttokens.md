---
type: Guide
title: MoeComputeExpertTokens 算子（indexing-gather / counting） 实现手册
description: '算子类别: indexinggather / counting。'
tags: [task.implementation, topic.attention, topic.mixture_of_experts, topic.tiling, topic.implementation.atomic, topic.implementation.masking]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Indexing_Ops/gather.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# MoeComputeExpertTokens 算子（indexing-gather / counting） 实现手册

> 类别约束见 [transformer_inference_constraints.md](../constraints/transformer_inference_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 MoeComputeExpertTokens 算子（indexing-gather / counting）

**算子类别**: `indexing-gather / counting`
**典型特征**: expert 维度 token 计数（histogram）+ 小数组前缀和（prefix sum / cumsum）
**性能基准**: 几何平均加速比 **1.7764x** vs torch（50/50 cases 全通过）

### §3.0 算子描述

输入：
- `sorted_expert_for_source_row`: int32 1D tensor，每个 token 对应的 expert ID（已排序）
- `num_expert`: int32 scalar，专家总数

输出：
- int32 1D tensor，长度 `num_expert`，`output[i]` = 专家 `0..i` 的累计 token 数

本质：先对每个 expert 做 token 计数（histogram），再做前缀和（prefix sum / cumsum）。

### §3.1 Layer 1: 设计约束（硬性边界）

#### L1.1 禁止在计数阶段使用 `tl.atomic_add`
- **原因**：所有 block 竞争写入同一 `counts` 地址，Ascend NPU 上 atomic 操作开销极大，导致性能严重退化（实测劣化至 0.3x 以下，归档 1.93x → 当前 0.30x）。
- **正确做法**：每个 expert 分配一个独立的 block，该 block 独享一个输出地址，通过串行循环累加完成计数，无需原子操作。
- **⚠️ 反模式警示**：`grid = (cdiv(N, BLOCK_SIZE),)` + block 内 `for expert in range(num_expert)` + `tl.atomic_add(output+expert, count)` 是错误的并行方向，必然触发本条禁忌，性能劣化 6 倍以上。

#### L1.2 禁止在 prefix sum 阶段使用 `tl.cumsum`
- **原因**：`tl.cumsum` 在 Ascend 后端可能退化为低效实现或引发 PyTorch fallback；对于 `num_expert <= 64` 的小数组，单 block 串行 scan 更可靠且足够快。
- **正确做法**：单 block 串行 `for` 循环读取-累加-存储。

#### L1.3 禁止在计数 kernel 中做跨 expert 的循环
- **禁止**：`for e in range(num_expert)` 在每个 block 内遍历所有 expert。
- **原因**：这会导致每个 block 对所有 expert 都做判断，计算冗余；且当 `num_expert` 较大时（如 64），循环展开后指令膨胀。
- **正确做法**：grid 维度 1 = `num_expert`，每个 block 只负责一个 expert 的计数。

#### L1.4 禁止在 `forward()` 中引入 `torch_npu` 依赖或动态 grid 计算
- **原因**：引入 `torch_npu` 增加环境依赖，且动态 grid 对编译器优化不友好；计数阶段 grid 固定为 `num_expert` 即可。
- **正确做法**：grid 直接传 `(num_expert,)` 和 `(1,)`，无需运行时计算。
- 与 G1 的差异：G1 要求动态读取 num_cores 用于一般算子；MoeComputeExpertTokens 的 grid 由语义维度（num_expert）决定，与核数无关，固定即可。

#### L1.5 grid 维度必须 = num_expert（expert-parallel），禁止 token-parallel
- **必须** `grid = (num_expert,)`，每 block 处理 1 个 expert，block 内向量化遍历所有 token。
- **禁止** `grid = (cdiv(N, BLOCK_SIZE),)`（token-parallel），这会强制 block 内跨 expert 循环 + atomic_add，违反 L1.1/L1.3，性能劣化至 0.3x。
- **正确骨架**：
  ```python
  # grid = (num_expert,), block 内向量化遍历 token
  expert_id = tl.program_id(0)
  count = 0
  for chunk_start in range(0, N, BLOCK_SIZE):
      tokens = tl.load(token_ptrs + chunk_start + tl.arange(0, BLOCK_SIZE), mask=...)
      is_match = (tokens == expert_id).to(tl.int32)
      count += tl.sum(is_match)  # 向量化规约，无 atomic
  tl.store(counts_ptr + expert_id, count)
  ```
- **Why:** expert-parallel 让每 block 独享输出地址（无竞争），向量化 tl.load + tl.sum 远快于逐元素循环。

### §3.2 Layer 2: 算法骨架（两阶段分离架构）

```
Phase 1: 计数（Count）
  Grid: (num_expert,)
  每个 block 对应一个 expert_id
  block 内：for chunk over tokens (BLOCK_SIZE=1024)
    load tokens -> compare == expert_id -> sum -> accumulate to local count
  最后 store count to counts[expert_id]

Phase 2: 前缀和（Prefix Sum）
  Grid: (1,)
  单 block 串行 scan
  for e in range(num_expert):
    count = load counts[e]
    prefix += count
    store prefix to output[e]
```

**核心设计决策**：

| 决策 | 选择 | 理由 |
|------|------|------|
| 计数并行维度 | expert 维度（grid = num_expert） | 无竞争、无 atomic、自然负载均衡 |
| 前缀和并行度 | 单 block 串行 | num_expert 很小，串行足够快且避免复杂同步 |
| BLOCK_SIZE | 1024 | 经验值，在 Ascend 上向量 load/store 效率较高 |
| 中间 buffer | `torch.empty` 分配 counts | 避免在 kernel 内做复杂内存管理 |

### §3.3 Layer 3: 关键技巧

#### L3.1 无竞争计数（Expert-Parallel Counting）

```python
@triton.jit
def moe_count_kernel(
    sorted_expert_ptr, counts_ptr, num_tokens,
    num_expert: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    expert_id = tl.program_id(0)
    if expert_id >= num_expert:
        return

    count = 0
    num_chunks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * BLOCK_SIZE
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens

        tokens = tl.load(sorted_expert_ptr + offsets, mask=mask, other=0)
        is_match = (tokens == expert_id) & mask
        chunk_count = tl.sum(is_match.to(tl.int32))
        count += chunk_count

    tl.store(counts_ptr + expert_id, count)
```

**要点**：
- `grid = (num_expert,)`，每个 block 只写一个输出地址。
- `tl.sum(is_match.to(tl.int32))` 将 mask 向量规约为标量，无需 atomic。
- `other=0` 避免越界读取影响比较结果。

#### L3.2 小数组串行前缀和

```python
@triton.jit
def prefix_sum_kernel(counts_ptr, output_ptr, num_expert: tl.constexpr):
    if tl.program_id(0) > 0:
        return

    prefix = 0
    for eid in range(num_expert):
        count = tl.load(counts_ptr + eid)
        prefix += count
        tl.store(output_ptr + eid, prefix)
```

**要点**：
- 仅允许 `program_id(0) == 0` 执行，其余 block 直接返回。
- `num_expert` 为 `tl.constexpr`（T4），编译器可展开循环。
- 对于 `num_expert <= 64`，latency 极低（实测约 2.8ms）。

#### L3.3 避免 `torch.zeros` 引入额外算子

```python
# 不良：counts = torch.zeros(...)
# 会引入 aclnnInplaceZero / aten::zero_ 等 PyTorch 算子

# 良好：counts = torch.empty(...)
# 计数 kernel 会覆盖全部 num_expert 个元素，无需预清零
```

**要点**：`torch.empty` 不初始化，但计数 kernel 保证每个位置都被写入，安全且避免额外算子。

### §3.4 MoeComputeExpertTokens 性能基准

- **框架延迟**：0.0126 ms（平均）
- **实现延迟**：0.0108 ms（平均）
- **几何平均加速比**：**1.7764x**
- **Shape 范围**：tokens = 50~10000，experts = 4/8/16/32/64
- **最差 case**：experts=64, tokens=1000，仍达 0.66x（未劣化）
- **最佳 case**：experts=4, tokens=500，达 6.68x

> **基准差异说明**：上述 1.7764x 基准对应 PyTorch 参考为纯 Python 实现（`bincount + cumsum`）或较低优化度的 torch 路径。若任务文件的 `Model.forward` 直接调用 `torch_npu.npu_moe_compute_expert_tokens`（CANN 原生高度优化算子），Triton 实现受限于两阶段 kernel 启动开销与 O(num_expert × num_tokens) 计数复杂度，几何平均加速比可能显著下降（实测约 0.25~0.30x）。此时小 shape（tokens ≤ 500、experts ≤ 8）仍可接近或超过 0.8x，大 tokens / 大 experts 是主要瓶颈。

---
