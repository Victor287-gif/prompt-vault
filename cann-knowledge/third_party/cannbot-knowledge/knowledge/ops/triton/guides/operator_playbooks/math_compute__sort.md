---
type: Guide
title: Sort 算子（sort-topk） 实现手册
description: '算子类别: sorttopk。'
tags: [task.implementation, topic.compute.reduction, topic.compute.ordering, topic.tiling, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Scan_Sort_Ops/sort.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Sort 算子（sort-topk） 实现手册

> 类别约束见 [math_compute_constraints.md](../constraints/math_compute_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §4 Sort 算子（sort-topk）

**算子类别**: `sort-topk`
**典型特征**: 沿指定维度（通常最后一维）对张量排序，支持 `descending`、任意 `dim`
**性能基准**: 31 cases 精度全过；几何平均 **2.0393x** vs torch（另一次实测 1.793x），implementation avg latency 约 **0.2243 ms**，framework avg latency 约 **0.4355 ms**

> ## ⚠️ 排序原语：用 `tl.sort`，且它不是硬件指令
>
> **API 名**：本环境（Triton 3.2.0 + Ascend）`triton.language.extra.cann` 模块**不可用**，
> 写 `ext.sort(...)` 会 `MLIRCompilationError`。可用调用是
> `tl.sort(x, dim=..., descending=...)`。下文所有 `tl.sort` 均指它。
>
> **实现真相（2026-08-14 实测更正）**：`tl.sort` 在 Ascend 后端**不是** CANN Vector Sort
> 硬件指令，而是 Triton `language/standard.py` 里的**软件 bitonic**（hypercube reshape）。
> 本卡旧版称其为「CANN Vector Sort 硬件指令」，与实测不符，已更正。它仍显著快于
> 自研软件 bitonic（编译器可整体优化 reshape 链），但不要按「硬件排序」估算收益。
>
> ### ⛔ 工具链警告：下表配置在 **2026-08-13 14:27 之前**的 bishengir 上验证
>
> 该时刻 `bishengir-compile` 与 triton `compiler.py` 被整体替换。**新工具链上实测**
> （8_Sort，2026-08-14，逐 dtype 诊断）：
>
> ```
> fp32 L=2048 desc:  FAIL 104.6s   [ConvertLinalgRToBinary] IR Dump After PlanMemory Failed
> fp16 L=4096 desc:  FAIL 242.8s
> fp32 L=4096 asc:   FAIL 150.2s
> ```
>
> 即 **`tl.sort` 在 L >= 2048 已不可用**（fp32/fp16、asc/desc 全失败），且每次失败
> 耗时 100~240s，31 case 累计会撞 verify 的 900s 超时。
>
> **因此**：下表仅作历史记录，说明「旧工具链上此配置可用、geomean 1.793」。
> 在新工具链上**不要据此认定 `tl.sort` 是 L<=4096 的可用路径**，须先用小脚本实测
> 当前 bishengir 的可用 L 上界，再决定分段边界。
>
> **已验证可用的调用配置**（来源：迁移前基线实现，31/31 精度、geomean 1.793 实测）：
>
> | 要点 | 配置 |
> |---|---|
> | `BLOCK_ROWS` **必须取 2 的幂** | `br = 1 << (br.bit_length() - 1)`；`BR*L` 非 2 的幂即编译失败 |
> | 2-D 路径 | `tl.sort(X, dim=1)`，`X` 形状 `[BR, L]`，`L = next_pow2(n) <= 4096` |
> | 1-D 半段路径 | `tl.sort(X, dim=0)`，`X` 形状 `[HALF]` |
> | 升序 | 取负 trick `-tl.sort(-X, dim=0, descending=True)`，见 L1.7 |
>
> **已知失败模式**：形状/BR 不满足上表时报
> `error: cannot align N axis for hivm.hir.bitcast memref<1x2x2x...x1xf32>`
> （`standard.py:361` 的 bitcast），N 随 `log2(L)` 增长。此时**不要直接改写自研软件
> bitonic**——那会让 `[512]` 这类小 case 慢 14 倍（45 轮 gather+compare vs 一次调用）。
> 应先按上表调整 BR 与 dim。
>
> **反面教材（8_Sort，2026-08-14）**：本卡旧版 `ext.sort` 措辞 27 处、`tl.sort` 仅 2 处，
> 生成时照主导措辞写 `ext.sort` → 编译失败 → **全量回落自研软件 bitonic**，
> geomean 从 1.79 掉到 0.47，17/31 case 跌破 0.6x 门槛。

> **评判口径**: framework 参考为 `torch.sort`；目标聚焦在小~中 L（末维长度）上利用 `tl.sort` 取得正加速，同时保证 31/31 精度全过。注意 case 24（shape `[8192, 16384]`）speedup 0.8749 < 1，是软件 bitonic 在超大 L 上的固有弱项，整体几何平均仍 > 1.9x。

### §4.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 必须按 L（末维长度）分段选择排序策略
- **必须**按 `L = next_pow2(n)` 分段，不同段用不同 kernel：
  - `L ≤ 4096` → `tl.sort`（单 kernel，最优路径）
  - `4096 < L ≤ 8192` → 软件 bitonic `sort_2d`（单 kernel）
  - `L == 16384 且 fp16` → 三段式 sort-merge（多 launch，见 L2.3）
  - `L > 8192`（一般） → 递归 bitonic（chunk sort + 多级 merge，见 L2.4）
- **Why:**
  - `tl.sort` 是 Triton `standard.py` 的软件 bitonic（非硬件指令，见顶部说明），但编译器可
    整体优化其 reshape 链，L ≤ 4096 时仍远快于自研逐 stage `tl.gather` 实现
  - `tl.sort` 对单次排序长度有上限（UB 约束），L 过大必须拆分
  - 软件 bitonic 在大 L 上访存与比较次数指数增长，必须递归分块
- **How to apply:** host 侧 `_sort_lastdim` 中按 L 分支决策。

#### L1.2 必须将排序维度归一化到最后一维
（见 §1 M1）
- **必须**在 `forward()` 中处理 `dim != x.ndim - 1`：permute 使目标维到末维 → contiguous → 排序 → permute 回。
- **Why:** kernel 内部只处理 `(n_rows, n)` 二维逻辑，降维复杂度；多维 grid 在该算子无收益。

```python
if dim < 0: dim += x.ndim
if dim != x.ndim - 1:
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_t = x.permute(perm).contiguous()
    y_t = self._sort_lastdim(x_t, descending)
    return y_t.permute(perm).contiguous()
```

#### L1.3 必须 pad 到 2 的幂并按方向填极值
（见 §1 M2）
- **必须**将末维 `n` pad 到 `L = next_pow2(n)`（上限 32768），pad 值按方向取 `+inf`（asc）/ `-inf`（desc）。
- 排序完成后截断回 `[:, :n]`。
- **How to apply:** `pad_val = -float("inf") if DESCENDING else float("inf")`。

#### L1.4 禁止在 kernel 内或 forward 中调用 torch 排序算子
（见 GC-4）
- **禁止**使用 `torch.sort` / `torch_npu.npu_sort` 等预计算。
- **Why:** 违反纯 Triton 算子实现约束，失去可复用性；`tl.sort` 是 Triton 内建原语，不算 PyTorch 退化。
- **How to apply:** 用 `tl.sort(x, dim=..., descending=...)`；**禁止** `triton.language.extra.cann.extension.sort`（该模块本环境不可用，见顶部说明）。

#### L1.5 Grid 大小必须动态读取实际 vector core 数量
（见 GC-1）
- **必须**通过 `triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)` 获取核数。
- **禁止**硬编码 `num_cores`。
- **How to apply:** `_get_num_vector_cores()` 兜底返回 48。

#### L1.6 BLOCK_ROWS 必须按排序方法分别自适应
（见 §1 M4）
- **必须**为 `tl.sort` 和软件 bitonic 分别提供 BR 选择函数：
  - `tl.sort` UB：`L * BR ≤ 8192 (fp32) / 16384 (fp16/bf16)` → `_block_rows_for_extsort`
  - 软件 bitonic UB：约 50KB / (L × itemsize × 8 live buffers) → `_block_rows_for_L`
  - 两者目标都是 `grid = cdiv(n_rows, BR) ≤ num_cores`，且 `n_rows ≤ num_cores` 时 `BR=1`。
- **Why:** 两种排序的 UB 占用差异巨大，共用一个 BR 函数会溢出或浪费并行度。

#### L1.7 升序大 L 用 tl.sort 时必须用取负 trick
- **必须**对 `tl.sort(8192, descending=False)` 路径使用 `Y = -tl.sort(-X, dim=1, descending=True)`。
- **Why:** `tl.sort` 在 8192 长度上 `descending=True` 仅对 fp16 稳定可用；asc 路径需通过取负转换为 desc。
- **How to apply:** `sort_half_kernel` 中按 `DESCENDING` 分支。

#### L1.8 禁止在 kernel 内使用 `continue` / `break`
（见 transformer-inference.md 相关）
- **Why:** 该后端对非规整控制流支持有限。
- **How to apply:** 所有场景，用 `tl.where` 条件赋值替代。

### §4.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树

```python
def forward(self, x, dim=-1, descending=False):
    # 归一化 dim 到末维（L1.2）
    ...
    return self._sort_lastdim(x.contiguous(), descending)

def _sort_lastdim(self, x, descending):
    n = x.shape[-1]
    x_2d = x.reshape(-1, n)
    n_rows = x_2d.shape[0]
    L = _next_pow2(n)          # cap 32768
    LOG_L = _log2_exact(L)
    y_2d = torch.empty((n_rows, n), ...)

    if L <= 4096:                                  # tl.sort 硬件
        BR = _block_rows_for_extsort(L, ...)
        sort_extsort_kernel[grid](x_2d, y_2d, n_rows, n, n, L=L, DESCENDING=descending, BLOCK_ROWS=BR)
        return y_2d.reshape(orig_shape)

    if L <= 8192:                                 # 软件 bitonic 单 kernel
        BR = _block_rows_for_L(L, ...)
        sort_lastdim_kernel[grid](x_2d, y_2d, n_rows, n, n, L=L, LOG_L=LOG_L, DESCENDING=descending, BLOCK_ROWS=BR)
        return y_2d.reshape(orig_shape)

    if L == 16384 and dtype == fp16:              # 三段式 sort-merge（L2.3）
        ...
        return y_2d[:, :n].reshape(orig_shape)

    # L > 8192 一般情况：递归 bitonic（L2.4）
    ...
    return src[:, :n].reshape(orig_shape)
```

#### L2.2 tl.sort kernel 骨架

```python
@triton.jit
def sort_extsort_kernel(X_ptr, Y_ptr, n_rows, n_cols, L_orig,
                        L: tl.constexpr, DESCENDING: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offs = tl.arange(0, L)
    mask = (row_offs < n_rows)[:, None] & (col_offs < L_orig)[None, :]
    pad_val = -float("inf") if DESCENDING else float("inf")
    X = tl.load(X_ptr + row_offs[:, None] * n_cols + col_offs[None, :], mask=mask, other=pad_val)
    Y = tl.sort(X, dim=1, descending=DESCENDING)   # BR 必须为 2 的幂，见顶部说明
    tl.store(Y_ptr + row_offs[:, None] * n_cols + col_offs[None, :], Y, mask=mask)
```

#### L2.3 L=16384 fp16 三段式 sort-merge 骨架

```python
HALF = 8192
mid = x_2d.contiguous()
# Stage 1a/1b: 两半分别 tl.sort（方向相反）→ 构成 bitonic-16384
sort_half_kernel[(n_rows,)](mid, y_2d, ..., HALF=HALF, OFFSET=0,      DESCENDING=descending)
sort_half_kernel[(n_rows,)](mid, y_2d, ..., HALF=HALF, OFFSET=HALF,   DESCENDING=not descending)
# Stage 2: 逐元素 min/max 合并（无 gather）→ [min-8192, max-8192]，各半 bitonic
elem_merge_halves_kernel[grid](y_2d, y_2d, ..., HALF=HALF, DESCENDING=descending, BR=1)
# Stage 3a/3b: 各半再 tl.sort（同方向）→ 完成排序
sort_half_kernel[(n_rows,)](y_2d, y_2d, ..., HALF=HALF, OFFSET=0,    DESCENDING=descending)
sort_half_kernel[(n_rows,)](y_2d, y_2d, ..., HALF=HALF, OFFSET=HALF, DESCENDING=descending)
return y_2d[:, :n]
```

#### L2.4 递归 bitonic 骨架（L > 8192 一般情况）

```python
L_TILE = 4096
N_CHUNKS = L // L_TILE
# pad 到 L（L1.3）
mid = torch.cat([x_2d, pad_col], dim=1).contiguous()

# Pass 1: 每个 chunk 用 tl.sort 排序，方向按 bitonic 递归交替
chunk_dirs = _bitonic_chunk_directions(N_CHUNKS, descending)   # [T,F,T,F,...] 交替
for c in range(N_CHUNKS):
    sort_extsort_kernel[grid](chunk_slice, chunk_slice, ..., L=L_TILE, DESCENDING=chunk_dirs[c], ...)

# Pass 2: level-1 merge（相邻 chunk 对 → size 8192），用 bitonic_merge_2d
for p in range(N_CHUNKS // 2):
    bitonic_merge_subrange_kernel[grid](mid, mid, ..., HALF=L_TILE, LOG_HALF=..., OFFSET=offset, DESCENDING=merge_desc, ...)

# Pass 3: cross-tile + within-tile merge（每 stage 1 row × 1 tile 一个 program）
for s in range(LOG_L - LOG_L_TILE):
    cross_tile_merge_stage_kernel[(n_rows * N_CHUNKS,)](src, dst, ..., stride_shift=LOG_L-1-s, ...)
    src, dst = dst, src
within_tile_merge_kernel[(n_rows * N_CHUNKS,)](src, dst, ..., L_TILE=..., LOG_L_TILE=..., ...)
src, dst = dst, src
return src[:, :n]
```

#### L2.5 chunk 方向递归计算
（见 §1 M6 方向交替）

```python
def _bitonic_chunk_directions(n_chunks, final_descending):
    if n_chunks == 1:
        return [final_descending]
    half = n_chunks // 2
    left = _bitonic_chunk_directions(half, final_descending)
    right = _bitonic_chunk_directions(half, not final_descending)   # 右半反向 → 构成 bitonic
    return left + right
```

#### L2.6 bitonic stage 核心比较模式

```python
# XOR 索引 partner：partner_idx = cols ^ stride_mask
partner_val = tl.gather(x, partner_idx_2d, axis=1)
# 单次比较 + select：swap = (x > partner) == want_min
x = tl.where((x > partner_val) == want_min_2d, partner_val, x)
```

### §4.3 Layer 3: 关键技巧

#### L3.1 tl.sort 替代自研软件 bitonic

**关键**: `tl.sort` 编译为整体优化的 bitonic 网络，L ≤ 4096 时远快于逐 stage `tl.gather` 的自研实现；`triton.language.extra.cann.extension.sort` 本环境不可用（见顶部说明）。

```python
from triton.language.extra.cann import extension as ext
Y = tl.sort(X, dim=1, descending=DESCENDING)
```

**可替代方向**: 仅在 `tl.sort` 不可用时回落到软件 bitonic；硬件路径优先级最高。

#### L3.2 升序取负 trick

```python
# asc 路径（tl.sort 8192 desc 仅 fp16 稳定）
if DESCENDING:
    Y = tl.sort(X_2d, dim=1, descending=True)
else:
    Y = -tl.sort(-X_2d, dim=1, descending=True)
```

**可替代方向**: 若未来 `tl.sort` asc 路径稳定，可直接 `descending=False`，省去取负开销。

#### L3.3 逐元素 min/max 合并替代 gather merge（核心加速）

**关键洞察**: 两个已排序半（方向相反）构成 bitonic 序列。逐元素 min/max 即可拆分为 `[min-半, max-半]`，**无需 gather**。

```python
# 输入: [sorted-half1 (dir), sorted-half2 (opposite dir)] = bitonic
# 输出: [min-half, max-half] — 各半仍 bitonic
if DESCENDING:
    first  = tl.maximum(X1, X2)   # 前 8192
    second = tl.minimum(X1, X2)   # 后 8192
else:
    first  = tl.minimum(X1, X2)
    second = tl.maximum(X1, X2)
```

**收益**: 历史 V30 测得比 cross-tile gather merge 快约 **1230x**（消除 gather 随机访存）。

**可替代方向**: 该 min/max 拆分仅适用于「两半构成 bitonic」的前置条件；普通两路已排序序列合并仍需归并 merge。

#### L3.4 小 L 软件 bitonic 静态展开

```python
@triton.jit
def bitonic_sort_2d(x, L: tl.constexpr, LOG_L: tl.constexpr, DESCENDING: tl.constexpr):
    BLOCK_ROWS: tl.constexpr = x.shape[0]
    if LOG_L <= 8 and BLOCK_ROWS == 1:   # 1D 小 L 静态展开
        return bitonic_sort_2d_static(x, L=L, LOG_L=LOG_L, DESCENDING=DESCENDING)
    # 否则动态循环
    for k in range(1, LOG_L + 1):
        for j in range(k):
            ...

# static 版本用 tl.static_range 完全展开
@triton.jit
def bitonic_sort_2d_static(x, L, LOG_L, DESCENDING):
    for k in tl.static_range(1, LOG_L + 1):
        for j in tl.static_range(0, k):
            x = _bitonic_stage_static(x, K=k, J=j, ...)
    return x
```

**可替代方向**: 静态展开仅在 `LOG_L ≤ 8`（L ≤ 256）且 `BLOCK_ROWS==1` 时收益明确；2D 或大 L 用动态循环避免编译爆炸。

#### L3.5 bitonic merge 的方向内联

```python
# 把方向判断内联进 swap 条件，消除 want_min 中间张量
# asc:  swap when (x > partner) != is_right → 取 partner 放左侧 min
# desc: swap when (x > partner) == is_right → 取 partner 放左侧 max
if DESCENDING:
    x = tl.where((x > partner_val) == is_right_2d, partner_val, x)
else:
    x = tl.where((x > partner_val) != is_right_2d, partner_val, x)
```

**收益**: 减少 1 个 int32 中间 buffer，缓解 UB 压力。

#### L3.6 cross-tile / within-tile merge 的 grid 分配

```python
# 每 stage：1 row × 1 tile 一个 program，grid = (n_rows * N_CHUNKS,)
n_tiles_per_row: tl.constexpr = L // L_TILE
row = pid // n_tiles_per_row
tile = pid - row * n_tiles_per_row
```

**Why**: 跨 tile merge 每个 tile 独立比较 partner tile，按 tile 并行最自然；within-tile merge 则在每个 tile 内做完整 bitonic merge。

**可替代方向**: 可尝试按 row 聚合多 tile 到单 program（核内循环），但 UB 占用会上升。

#### L3.7 BLOCK_ROWS 自适应模板
（见 §1 M4）

```python
def _block_rows_for_extsort(L, dtype_itemsize, n_rows=1, num_cores=None):
    if num_cores is None: num_cores = _get_num_vector_cores()
    ub_total = 8192 if dtype_itemsize == 4 else 16384   # fp32 / fp16
    ub_limit = max(1, ub_total // L)
    if n_rows <= num_cores: return 1
    core_target = max(1, (n_rows + num_cores - 1) // num_cores)
    return min(ub_limit, core_target, n_rows)

def _block_rows_for_L(L, dtype_itemsize, n_rows=1, num_cores=None):
    if num_cores is None: num_cores = _get_num_vector_cores()
    max_buf_elems = 50 * 1024 // dtype_itemsize
    ub_limit = max(1, max_buf_elems // (L * 8))         # ~8 live buffers
    if n_rows <= num_cores: return 1
    core_target = max(1, (n_rows + num_cores - 1) // num_cores)
    return min(ub_limit, core_target, n_rows)
```

**可替代方向**: UB 系数（50KB / 8 buffers）为经验值，可按实测 UB 峰值微调。

### §4.4 Sort 性能基准

| 指标 | 数值 |
|------|------|
| Implementation Avg Latency | 0.3056 ms |
| Framework Avg Latency | 0.3262 ms |
| 几何平均加速比 | **1.9578x** |
| 精度通过率 | 31/31 |
| 异常 shape 索引 | 无（nan/inf/zero/negative/none 均为空） |

**关键结论**:
1. **tl.sort 是小 L 的首选**：L ≤ 4096 时远快于自研逐 stage bitonic（它本身也是软件实现，见顶部说明）
2. **逐元素 min/max 合并是 bitonic merge 的关键加速**：利用「两半方向相反构成 bitonic」性质，消除 gather 随机访存
3. **L 分段策略是架构基础**：tl.sort / 软件 bitonic / 三段式 sort-merge / 递归 bitonic 各有适用区间
4. **UB 约束决定 BLOCK_ROWS**：tl.sort 与软件 bitonic 的 UB 占用差异巨大，必须分别自适应
5. **升序大 L 用取负 trick**：`tl.sort` 8192 desc 仅 fp16 稳定，asc 通过取负转换
6. **超大 L（16384+）是固有弱项**：case 24 speedup < 1，软件 bitonic 比较次数指数增长，几何平均仍 > 1.9x

---
