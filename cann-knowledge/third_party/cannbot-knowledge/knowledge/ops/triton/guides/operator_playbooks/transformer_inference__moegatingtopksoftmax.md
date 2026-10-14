---
type: Guide
title: MoeGatingTopKSoftmax 算子（sort-topk） 实现手册
description: '算子类别: sorttopk。'
tags: [task.implementation, topic.attention, topic.compute.ordering, topic.mixture_of_experts, topic.compute.normalization, topic.tiling]
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
# MoeGatingTopKSoftmax 算子（sort-topk） 实现手册

> 类别约束见 [transformer_inference_constraints.md](../constraints/transformer_inference_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §4 MoeGatingTopKSoftmax 算子（sort-topk）

**算子类别**: `sort-topk`
**典型特征**: softmax over last dim + iterative top-k selection, per-row independent processing
**性能基准**: 几何平均加速比 **0.8194x** vs torch（50/50 cases pass，最新验证版本 v10_20260624）
**历史最佳版本**: v8_20260624，**0.8967x**（50/50 cases pass）

### §4.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 禁止在 kernel 内使用 PyTorch 计算
（见 §1 T3）
- **必须** 将所有计算（softmax + top-k）放在 `@triton.jit` kernel 内完成。
- **禁止** 在 `forward()` 中使用 `torch.softmax`、`torch.topk` 等 PyTorch 算子。

#### L1.2 禁止 bool/int1 GM 缓冲区
- **禁止** 使用 `torch.bool` 类型作为 GM 临时缓冲区并通过 `tl.load`/`tl.store` 访问。
- **Why:** Triton 将 `tl.int1` GM 存取视为 `int8`，导致 `tl.where` 条件类型不匹配，在 f32 大 shape 上产生精度错误（max_rel_err ~1.0）。
- **How to apply:** `finished` 标志应直接以 `torch.bool` 传入 kernel，`tl.load` 后直接作为 `tl.where` 条件使用，无需中间 GM 缓冲区。

#### L1.3 禁止 GM temp buffer 存储中间 softmax 结果
- **禁止** 将 `softmax_val` 写入 GM temp buffer 再读回用于 top-k。
- **Why:** GM store/load 引入额外内存带宽，且 `tl.where` 修改后的向量经 GM 往返后 `tl.argmax` 不可靠。
- **How to apply:** 保持 `softmax_val` 在寄存器中，top-k 循环内通过 `curr = tl.where(offsets == best_idx, -inf, curr)` 直接更新寄存器向量。

#### L1.4 Grid 大小必须匹配 num_cores（见 §1 T1）

#### L1.5 禁止在 kernel 参数中使用运行时变量作为 tl.arange 参数（见 §1 T4）
- **必须** 将 `num_experts`、`k` 等作为 `tl.constexpr` 传入 kernel。

#### L1.6 禁止在 kernel 内使用 Python if/else 判断指针是否为 None
- **禁止** 在 `@triton.jit` kernel 内使用 `if finished_ptr is not None:`
- **Why:** Triton 不支持 Python 的 `is not None` 指针判断，会编译失败或行为不可预期。
- **How to apply:** 在 host 侧 `forward()` 中处理：若 `finished is None`，创建一个全 `False` 的 dummy tensor 传入 kernel，kernel 内直接 `tl.load(finished_ptr + row)` 无需判断。

#### L1.7 禁止在 kernel 内使用 Python break/continue
- **禁止** 在 `@triton.jit` kernel 内使用 `break` 或 `continue`。
- **Why:** Triton 不支持 Python 循环控制流，会报 `unsupported AST node type: Break`。
- **How to apply:** 使用 `tl.where` 条件赋值替代。例如：`if row >= end_row: break` → 改为 `row_valid = row < end_row` + `tl.where(row_valid, ...)` 或确保循环范围正确。

#### L1.8 NUM_EXPERTS 较小时必须单次整行 load，禁止 chunked multi-pass
- **必须** 当 `NUM_EXPERTS` 是编译期常量且 `NUM_EXPERTS <= BLOCK_E`（通常 ≤1024，pow2 时编译期 `tl.arange` 安全）时，用**单次整行 load**：`offsets = tl.arange(0, NUM_EXPERTS)` 一次 load 整行 gating 权重到寄存器，单 pass 完成 max/sum/topk。
- **禁止** 对小 `NUM_EXPERTS` 使用 chunked multi-pass（PASS1 求 max + PASS2 求 sum + PASS3 topk 分三次循环 load 同一 chunk），这会把内存带宽放大 3 倍，大 shape 性能跌到 0.1-0.4x。
- **Why:** `tl.arange(0, NUM_EXPERTS)` 当 NUM_EXPERTS 是 `tl.constexpr` 时是编译期固定长度向量，单次 load 到寄存器后所有 max/sum/argmax 运算都在寄存器内完成，零重复访存。chunked 路径仅当 `NUM_EXPERTS > UB 容量`（实测 >4096）才需要。
- **L1.3 适用范围澄清**：L1.3 警告"GM temp buffer 会使 argmax 不可靠"指的是**写回 GM 再读**的场景；寄存器内的 `tl.argmax` 在单 chunk 内是可靠的，可直接使用。
- **How to apply:** 见 L3 中"单次整行 load + 寄存器 topk"骨架；仅当 num_experts 编译期未知或 >4096 时才走 chunked 路径（且应合并 max+sum 为 2-pass）。

### §4.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树

```python
forward(x, finished, k):
  1. x_flat = x.view(-1, num_experts).contiguous()        # G7
  2. finished_flat = finished.view(-1) if finished else zeros(batch_size, bool)  # L1.6
  3. grid_size = min(batch_size, VEC_CORE_NUM)            # T1
  4. kernel[grid_size](..., num_cores=grid_size, k=k)      # num_cores == grid_size（L1.4）
```

#### L2.2 Kernel 内按行分配骨架

```python
pid = tl.program_id(0)
rows_per_core = cdiv(batch_size, num_cores)  # num_cores == grid_size（T1）
start_row = pid * rows_per_core
end_row = min(start_row + rows_per_core, batch_size)

for row in range(start_row, end_row):
    # 1. Load row + softmax（fp32，见 T2）
    # 2. Top-k in registers（k iterations of max+argmax+where mask）
    # 3. Store top-k values/indices
    # 4. Store row_idx
```

#### L2.3 纯寄存器 top-k 骨架

```python
vals_vec = tl.full((k,), 0.0, tl.float32)
idxs_vec = tl.full((k,), 0, tl.int32)
curr = softmax_val  # register vector

for i in range(k):
    best_val = tl.max(curr, axis=0)
    best_idx = tl.argmax(curr, axis=0)
    final_idx = tl.where(finished_flag, num_experts, best_idx)

    mask = tl.arange(0, k) == i
    vals_vec = tl.where(mask, best_val, vals_vec)
    idxs_vec = tl.where(mask, final_idx.to(tl.int32), idxs_vec)

    curr = tl.where(offsets == best_idx, -inf, curr)
```

### §4.3 Layer 3: 关键技巧

#### L3.1 纯寄存器路径（已验证有效）

```python
# 完全避免 GM temp buffer
# softmax_val 保持在寄存器中
curr = softmax_val
for i in range(k):
    best_val = tl.max(curr, axis=0)
    best_idx = tl.argmax(curr, axis=0)
    # ... accumulate ...
    curr = tl.where(offsets == best_idx, -float('inf'), curr)
```

**可替代方向**: 若 k 极大且寄存器压力过高，可考虑分块处理，但需验证 `tl.argmax` 在 GM 往返后的可靠性。

#### L3.2 k=1 快速路径（已验证有效）

```python
if k == 1:
    best_val = tl.max(softmax_val, axis=0)
    best_idx = tl.argmax(softmax_val, axis=0)
    final_idx = tl.where(finished_flag, num_experts, best_idx)
    tl.store(topk_values_ptr + row * k, best_val)
    tl.store(topk_indices_ptr + row * k, final_idx.to(tl.int32))
    tl.store(row_idx_ptr + row * k, row.to(tl.int32))
```

**可替代方向**: 也可统一走循环路径，但 k=1 快速路径减少标量循环开销。

#### L3.3 finished 标志直接 bool 传入（已验证有效）

```python
# Host side
finished_flat = finished.view(-1) if finished is not None else torch.zeros(
    batch_size, dtype=torch.bool, device=x.device)

# Kernel side
finished_flag = tl.load(finished_ptr + row)  # directly bool
final_idx = tl.where(finished_flag, num_experts, best_idx)
```

**可替代方向**: 若编译器对 bool GM 支持不稳定，可尝试 int8，但 int32 会引入额外 cast 开销。

#### L3.4 向量存储 top-k 结果（已验证有效）

```python
# Accumulate into register vectors during loop
vals_vec = tl.full((k,), 0.0, tl.float32)
idxs_vec = tl.full((k,), 0, tl.int32)
# ... loop ...
# Single vector store per row
tl.store(topk_values_ptr + row * k + store_offsets, vals_vec)
tl.store(topk_indices_ptr + row * k + store_offsets, idxs_vec)
```

**可替代方向**: 循环内逐元素 store 也可工作，但向量 store 减少 GM 写指令数。

#### L3.5 小 shape profiler 测量异常处理（已验证有效）

**问题**: framework 延迟 < 0.1ms 的 case，profiler 测量误差导致 speedup 异常高（2.5x~1251x）

**解决方案**:
1. 在 benchmark 后过滤异常 case：排除 framework 延迟 < 0.1ms 或 speedup > 2.5x 的 case
2. 重新计算几何平均加速比
3. 报告时注明过滤的 case 数量和原因

```python
MIN_FRAMEWORK_MS = 0.1   # 最小 framework 延迟阈值
MAX_SPEEDUP = 2.5        # 最大合理加速比

valid_cases = [
    r for r in per_shape_results
    if r['framework']['avg_latency_ms'] >= MIN_FRAMEWORK_MS
    and r['speedup_vs_torch'] is not None
    and r['speedup_vs_torch'] <= MAX_SPEEDUP
]
```

**可替代方向**: 增加 repeats 次数（50→100）可能减少测量误差，但无法完全消除系统噪声。

#### L3.6 大 NUM_EXPERTS 自适应 chunk 策略（2026-07-02 验证）

**问题**: 当 NUM_EXPERTS > 1536 时，单次整行 load 在 Ascend 上产生错误结果（输出大部分为 0）；而固定小 chunk（如 1024）会把大 E shape 拆成过多 chunk，内存遍历次数高，性能跌至 0.04x 以下。

**解决方案**: 采用自适应 CHUNK_E + online softmax + all-candidates 寄存器合并：
1. **单 chunk 安全阈值**: NUM_EXPERTS <= 1536 走整行寄存器路径（与 L1.8 一致，但实测安全上限为 1536 而非 1024）。
2. **大 E chunk 选择**:
   - 1536 < NUM_EXPERTS <= 2048：CHUNK_E = 2048（pow2，单 chunk）。
   - NUM_EXPERTS > 2048：CHUNK_E = 4096（在 UB 容量内，最大化每 chunk 利用率，减少 chunk 数）。
3. **online softmax**: 第一遍遍历同时更新 global_max 与 global_sum，避免 3-pass。
4. **all-candidates 寄存器缓冲区**: 每 chunk 选出 top-K 后存入大小为 `NUM_CHUNKS * K` 的寄存器向量，最后统一选全局 top-K，避免每 chunk 与全局 top-K 反复合并。
5. **mask 加载**: 对 chunk 使用 `tl.load(..., mask=offsets < NUM_EXPERTS, other=-inf)`，避免越界并保证大 E 向量稳定。

**性能基准**: 该策略将 50 case 几何平均从 Phase 3 基线 0.46x 提升至 0.66x（仍低于 0.8x 目标，主要瓶颈为超大 batch + 大 E 的标量 row 循环）。

**可替代方向**: 对超大 batch 场景可进一步尝试 ROW_TILE=2/4 的行向量化，但需处理 2D top-K 候选缓冲区，复杂度较高。

### §4.4 MoeGatingTopKSoftmax 性能基准（几何平均）

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 小 shape (batch<=32) | 1.0x - 2.5x | 调度开销占比高，profiler测量不可靠，需过滤后评估 |
| 中 shape (32<batch<=512) | 0.4x - 0.8x | 主要优化区间 |
| 大 shape (batch>512) | 0.15x - 0.5x | 受限于内存带宽和 top-k 迭代开销 |
| 3D shape | 0.12x - 0.17x | 大内存 footprint，性能最差 |

**关键结论**（2026-06-24 更新）：
1. 纯寄存器路径（无 GM temp buffer）是正确性和性能的关键
2. `tl.argmax` 在寄存器向量上可靠，但在 GM 往返后不可靠
3. `grid_size = num_cores` 且 `num_cores` 参数匹配 grid 是正确性的硬性要求（T1）
4. bool finished 直接传入可避免 host-side cast 开销和 int8 警告
5. `multibuffer=True, unit_flag=True` 编译选项在本算子上未带来性能提升（v3: 0.8527x vs 基线 0.8836x，v9: 0.7353x vs 基线 0.7700x，v10: 0.5720x vs 基线 0.8194x），不建议默认开启（T6）
6. 该算子在 Ascend 上整体慢于 PyTorch（0.67x~0.89x），主要瓶颈是大 shape 的 top-k 迭代和内存带宽
7. 小 shape case（framework 延迟 <0.1ms）的 profiler 测量不可靠，应设置最小延迟阈值或排除这些 case
8. 环境变量 `LLVM_ROOT` 必须正确设置，否则 Triton 编译失败（symbol lookup error）
9. CANN 9.1.0 与 Triton 存在兼容性常量名差异（`RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` → `RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE`）

---
