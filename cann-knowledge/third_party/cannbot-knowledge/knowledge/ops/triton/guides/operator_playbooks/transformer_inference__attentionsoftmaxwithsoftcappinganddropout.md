---
type: Guide
title: AttentionSoftmaxWithSoftcappingAndDropout 算子（reduce） 实现手册
description: '算子类别: reduce（行级归约 + 前置 elementwise 变换 softcapping）。'
tags: [task.implementation, topic.attention, topic.tiling, topic.implementation.masking, topic.compute.normalization, topic.compilation]
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
# AttentionSoftmaxWithSoftcappingAndDropout 算子（reduce） 实现手册

> 类别约束见 [transformer_inference_constraints.md](../constraints/transformer_inference_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §5 AttentionSoftmaxWithSoftcappingAndDropout 算子（reduce）

**算子类别**: `reduce`（行级归约 + 前置 elementwise 变换 softcapping）
**典型特征**: Gemma3 风格 softcapping `tanh(x/30)*30` + 行级 softmax(dim=-1)，多 dtype (fp32/fp16/bf16) 混合输入
**性能基准**: **1.1388x**（geomean, 50/50），4-kernel 分离 + 分核优化（grid 钳制到 num_cores + 循环处理多块）

### §5.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 bf16/fp16 输入下 softcapping 必须拆分为独立 kernel（见 §1 T5）
- **必须** 将 `tanh(x/scale)*scale` 拆分为 div / tanh / mul 三个独立 kernel，每个 kernel 通过 GM store+load 往返强制中间 dtype 舍入。
- **禁止** 在 bf16/fp16 输入下用单 kernel 完成 `tanh(x/30)*30` 这类 softcapping 表达式。
- **Why:** Triton Ascend 编译器会将 `tanh(x/scale)*scale` 融合为单一 fp32 计算，跳过中间 bf16/fp16 舍入。PyTorch 参考实现在每个算子处物化中间结果为输入 dtype，融合后逐 bit 不匹配。
- **How to apply:** 当算子含 softcapping 且输入 dtype 为 bf16/fp16 时，必须拆分为独立 kernel。fp32 输入下不触发（融合后仍 fp32，无舍入损失）。

#### L1.2 softcapping 各子算子须 fp32 计算 + cast 回输入 dtype
- **必须** 在 div/tanh/mul kernel 内部将输入 load 后 cast 到 fp32，在 fp32 下做 div/tanh/mul，再 cast 回输入 dtype 后 store GM。
- **Why:** PyTorch 参考实现 `attn_weights / 30.0`、`torch.tanh(...)`、`clamped * 30.0` 中 Python float 30.0 会触发 PyTorch 内部提升到 fp32 计算，再 cast 回输入 dtype。若 kernel 内用输入 dtype 直接计算，bf16/fp16 下除法/乘法精度不足。
- **How to apply:** `sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32); scaled = (x_fp32 / sc_fp32).to(in_dtype)`

#### L1.3 softmax kernel 必须 fp32 内部计算（见 §1 T2）
- **必须** 在 softmax kernel 内部将输入 load 后 cast 到 fp32，再做 max/exp/sum/div。
- **Why:** 参考实现 `F.softmax(x, dim=-1, dtype=torch.float32)` 显式指定 fp32 计算。
- **How to apply:** load → cast fp32 → where(mask, x, -inf) → max → exp → where(mask, exp, 0) → sum → div → cast output dtype。

#### L1.4 mask 元素必须排除出归约
- **必须** 在 max 前用 `tl.where(mask, x, -inf)`，在 sum 前用 `tl.where(mask, exp_val, 0.0)`。
- **Why:** padding 元素若参与 max 会污染结果；若参与 sum 会让 sum 偏大。
- **How to apply:** load 时 `other=-inf`，max 前 where，exp 后 where 为 0 再 sum。

#### L1.5 引入循环后必须收紧 UB budget
- **必须** 当 kernel 内引入 `for block_id in range(pid, n_blocks, num_pids)` 循环时，UB budget 从 128KB 收紧到 48KB。
- **Why:** 循环展开后编译器需多份副本空间，大 K (BLOCK_N=1024) + ROW_TILE=16 时 2D tensor 占 64KB，循环内若 multibuffer 会溢出 192KB UB，触发 BiShengIR `ub over` 编译错误。
- **How to apply:** `_pick_row_tile` 中 `ub_budget = 48 * 1024`，确保 `ROW_TILE * BLOCK_N * 4 <= 48KB`。

### §5.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树

```
输入 x [B, H, Q, K], dtype ∈ {fp32, fp16, bf16}
    ↓
view → x2d [num_rows=B*H*Q, K]
    ↓
判断算子是否含 softcapping:
  ├─ 是 → 4-kernel 分离架构（div / tanh / mul / softmax）   # L1.1
  └─ 否 → 1-kernel softmax
    ↓
BLOCK_N = next_pow2(K), 上限 4096                            # G2
    ↓
ROW_TILE 自适应选择:
  num_rows >= 512 → ROW_TILE = 16
  num_rows >= 128 → ROW_TILE = 8
  num_rows >= 32  → ROW_TILE = 4
  num_rows >= 8   → ROW_TILE = 2
  otherwise       → ROW_TILE = 1
  + UB guard: ROW_TILE * BLOCK_N * 4 <= 48KB（循环场景，L1.5）
    ↓
num_cores = torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)  # G1
natural_blocks = ceil(num_rows / ROW_TILE)
grid_size = min(natural_blocks, num_cores)                   # T1，分核优化关键
grid = (grid_size,)
    ↓
逐 kernel 启动（每个 kernel 内部循环处理多块）
```

#### L2.2 多核并行骨架模式（分核优化版）

**模式 - grid 钳制 + 循环处理多块**:
```python
@triton.jit
def kernel(x_ptr, y_ptr, num_rows, K, stride_row, num_pids,
           BLOCK_N: tl.constexpr, ROW_TILE: tl.constexpr):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(num_rows, ROW_TILE)
    # grid 钳制到 num_pids 后，每个 program 处理多个 block（连续划分，stride=num_pids）
    for block_id in range(pid, n_blocks, num_pids):
        row_start = block_id * ROW_TILE
        row_offs = row_start + tl.arange(0, ROW_TILE)[:, None]
        col_offs = tl.arange(0, BLOCK_N)[None, :]
        mask = (row_offs < num_rows) & (col_offs < K)
        # ... load → compute → store
```

**关键点**:
- `num_pids` 作为 kernel 入参传入（= grid_size），用于循环 stride（T1）
- `for block_id in range(pid, n_blocks, num_pids)` 是连续划分（符合 checklist「禁止交织」规范）
- grid 不再远超核数，避免 NPU 串行调度开销

### §5.3 Layer 3: 关键技巧

#### L3.1 分核优化的 grid 钳制 + 循环模式

```python
# Host 侧
num_cores = torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)
natural_blocks = (num_rows + ROW_TILE - 1) // ROW_TILE
grid_size = natural_blocks if natural_blocks < num_cores else num_cores
grid = (grid_size,)

# Kernel 内
@triton.jit
def kernel(..., num_pids, ROW_TILE: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(num_rows, ROW_TILE)
    for block_id in range(pid, n_blocks, num_pids):
        row_start = block_id * ROW_TILE
        # ... 2D block 处理
```

**收益**: 相比"每个 program 处理 1 块、grid=natural_blocks"的朴素模式，grid 钳制到大 K (1024) + 大 num_rows 场景加速比从 1.06x 提升到 1.14x（+7.6%）。

**可替代方向**: 可用 `@triton.autotune` 让编译器自动搜索最优 ROW_TILE，但多 shape 场景下 autotune 会触发多次重新调优，对小 shape 可能引入额外开销。

#### L3.2 4-kernel 分离的 div/tanh/mul 模板

```python
@triton.jit
def _div_kernel(x_ptr, s_ptr, num_rows, K, stride_row, num_pids,
                SOFTCAP: tl.constexpr, BLOCK_N: tl.constexpr, ROW_TILE: tl.constexpr):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(num_rows, ROW_TILE)
    for block_id in range(pid, n_blocks, num_pids):
        row_offs = block_id * ROW_TILE + tl.arange(0, ROW_TILE)[:, None]
        col_offs = tl.arange(0, BLOCK_N)[None, :]
        mask = (row_offs < num_rows) & (col_offs < K)
        in_dtype = x_ptr.dtype.element_ty
        x = tl.load(x_ptr + row_offs * stride_row + col_offs, mask=mask, other=0.0)
        # fp32 计算 + cast 回输入 dtype（对齐 PyTorch 参考实现，T2/L1.2）
        x_fp32 = x.to(tl.float32)
        sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32)
        scaled = (x_fp32 / sc_fp32).to(in_dtype)
        tl.store(s_ptr + row_offs * stride_row + col_offs, scaled, mask=mask)
```

**关键点**:
- `sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32)` — 常量用 fp32，保证 div/mul 在 fp32 下计算
- `.to(in_dtype)` — 显式 cast 回输入 dtype，但真正强制舍入的是 store 到 GM 再由下个 kernel load（T5）
- tanh/mul kernel 结构相同，只是中间算子不同

#### L3.3 softmax kernel 的 fp32 + mask 排除归约

```python
@triton.jit
def _softmax_kernel(x_ptr, y_ptr, num_rows, K, stride_row, num_pids,
                    BLOCK_N: tl.constexpr, ROW_TILE: tl.constexpr):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(num_rows, ROW_TILE)
    for block_id in range(pid, n_blocks, num_pids):
        row_offs = block_id * ROW_TILE + tl.arange(0, ROW_TILE)[:, None]
        col_offs = tl.arange(0, BLOCK_N)[None, :]
        mask = (row_offs < num_rows) & (col_offs < K)
        x = tl.load(x_ptr + row_offs * stride_row + col_offs, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)
        x_fp32 = tl.where(mask, x_fp32, -float("inf"))
        row_max = tl.max(x_fp32, axis=1)[:, None]   # ← [:, None] 关键
        shifted = x_fp32 - row_max
        exp_val = tl.exp(shifted)
        exp_val = tl.where(mask, exp_val, 0.0)
        row_sum = tl.sum(exp_val, axis=1)[:, None]
        out_fp32 = exp_val * (1.0 / row_sum)
        tl.store(y_ptr + row_offs * stride_row + col_offs,
                 out_fp32.to(y_ptr.dtype.element_ty), mask=mask)
```

**注意**: `tl.max(x, axis=1)` 返回 1D tensor，必须加 `[:, None]` 才能 broadcast 回 2D。

### §5.4 AttentionSoftmaxWithSoftcappingAndDropout 性能基准（几何平均）

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 大 K (1024) + fp16 | 1.58~1.60 | 最佳区间，分核优化充分发挥多核并行 |
| 中等 shape + bf16/fp16 | 1.1~1.6 | 良好区间，2D block 充分利用 vector core |
| 小 shape + fp32 | 0.73~0.93 | 较弱区间，4-kernel GM round-trip 开销在小 shape 下显著 |
| 极小 shape (K=1) | 1.33 | case 47 [1,1,1,1]，单元素 softmax 退化 |

**关键结论**: 4-kernel 分离是精度必需（bf16/fp16 编译器融合问题），分核优化是性能关键（grid 钳制 + 循环处理多块，大 K 场景 +7.6%）。小 shape + fp32 场景因 4-kernel GM round-trip 开销仍有优化空间。

---
