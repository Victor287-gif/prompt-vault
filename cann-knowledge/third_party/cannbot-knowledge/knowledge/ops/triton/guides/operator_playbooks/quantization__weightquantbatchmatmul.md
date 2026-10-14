---
type: Guide
title: WeightQuantBatchmatmul 算子（weight-quantized batch matrix multiplication） 实现手册
description: '算子类别: matmul + quantization。'
tags: [task.implementation, topic.compute.quantization, topic.compute.composite, topic.implementation.memory_management, topic.programming_model.execution_unit.cube, topic.compute.indexing]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# WeightQuantBatchmatmul 算子（weight-quantized batch matrix multiplication） 实现手册

> 类别约束见 [quantization_constraints.md](../constraints/quantization_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §4 WeightQuantBatchmatmul 算子（weight-quantized batch matrix multiplication）

**算子类别**: `matmul` + `quantization`
**典型特征**: `x` fp16/bf16 `[M, K]`, `weight` int8 `[K, N]`, `antiquant_scale/offset` 支持 per-tensor `[1]` / per-channel `[1, N]` / per-group `[G, N]`；可选 `bias`；输出 dtype 与 `x` 相同
**性能基准**: 48/48 verify pass，几何平均加速比 **0.9133x** vs torch（本轮 repeats=50 实测，run-to-run 方差带 0.91–0.95；超 target 0.8x 达 14%；refactor_iter_20 = opt_iter_19 + N-strip chunking for L2-exceeding workspaces）

> **评判口径**: framework 参考实现为 `torch_npu.npu_weight_quant_batchmatmul`，是 NPU 高度优化的原生算子；Triton DSL 在部分 shape 上存在结构性劣势。优化目标应聚焦在 **implementation latency** 降低与 **48/48 精度全过**。
>
> ⚠️ **方差带提示（benchmark 噪声）**: 本算子含多个 ~17-20us 的极小 M case（c1/5/8/11/14/21/24/27/30/33/38/40/42），对 profiler 抖动敏感，单次 geomean 浮动 ±0.03 属正常。代码不变（sha256 一致）时多次实测会落在 **0.91–0.95** 区间。**判 target_reached 必须看是否稳定 ≥ 0.8，不要纠结单次 0.94 vs 0.91 的差异**。

### §4.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 必须按 group / non-group 分裂为两个专用 kernel
- **必须**为 `antiquant_group_size > 0`（group 模式）和 `antiquant_group_size == 0`（non-group 模式）分别编写 `@triton.jit` kernel，host 侧通过简单 `if is_group` 分支启动对应 kernel。
- **Why:** group 模式需要按 group id 加载 scale/offset，non-group 模式 scale/offset 在 K 维度恒定，合并会导致大量 `tl.where(IS_GROUP, ...)` 分支，降低向量利用率并增加 UB 压力。
- **注意**: 当前项目的 `validate_triton_impl.py` 要求 `kernel[grid](...)` 必须直接出现在 `ModelNew.forward()` 内部，不能封装到 `self._route()` 等辅助方法中。

#### L1.2 group 模式必须保留原始 `[G, N]` scale/offset，禁止 host 侧物化为 `[K, N]`
- **必须**将 group scale/offset 以原始 `[G, N]` 布局传入 kernel，由 kernel 按 `group_id = k_start // group_size` 加载单条 `[1, N]` 行。
- **禁止**在 host 侧用 `view(G,1,N).expand(G,group_size,N).contiguous()` 物化为 `[K, N]`。
- **Why:** host 物化会引入大量 CPU 时间与额外 HBM 写入；原始 `[G, N]` 每 K-block 只读一条 `[1, N]`，总读取量仅为 `K/group_size * N`，远低于物化后的 `K * N`。

#### L1.3 group kernel 中每个 K-block 必须落在单一 group 内
- **必须**选择 `BLOCK_K` 满足 `BLOCK_K <= group_size` 且 `group_size % BLOCK_K == 0`。
- **Why:** 保证当前 K-block 内所有 K 位置共享同一个 group 的 scale/offset，从而只需加载一条 `[1, N]` scale/offset 行并广播到 `[BLOCK_K, N]`。
- **推荐配置**:
  - `group_size % 128 == 0` 且 dtype 为 fp16: `BLOCK_K = 128`
  - 其他 group 场景（含 bf16 group）: `BLOCK_K = 64`

#### L1.4 dequant 域必须按 `x.dtype` 分流
- **fp16 输入**: 在 fp16 低精度域完成 `(weight + offset) * scale`，最后 cast 到 fp32 累加。
- **bf16 输入**: 在 fp32 域完成 `(weight + offset) * scale`，再 cast 到 bf16 后回 fp32（模拟 NPU cube L0C 行为）。
- **Why:** bf16 加法尾数损失会导致 verify fail；fp16 路径若在 fp32 域 dequant 也会与 NPU 参考实现产生微小差异。

#### L1.5 必须使用 1D Grid，grid 大小钳制到实际 Vector Core 数量
- **必须** `grid = (min(total_tiles, num_cores),)`，其中 `num_cores = triton.runtime.driver.active.utils.get_device_properties(device).get("num_vectorcore", 48)`。
- **禁止**多维 grid。
- **Why:** 1D grid 配合核内 `for tile_idx in range(pid, total_tiles, num_programs)` 是 Ascend vector core 上最稳定的负载均衡模式。

#### L1.6 `ModelNew.forward()` 中禁止任何 torch 计算，kernel 启动必须直接可见
- **必须**所有核心计算在 `@triton.jit` kernel 内完成；`forward()` 仅做 shape 处理、contiguous、buffer 分配、条件分派。
- **Why:** `validate_triton_impl.py` 会 flag `forward()` 中的 torch 计算或间接 kernel 调用。

#### L1.7 必须使用 int32 索引
- **必须**将 `tl.program_id`、`tl.arange` 结果 `.to(tl.int32)`，避免 int64 地址计算降级。

#### L1.8 bf16 group 路径必须保守选择 block size
- **必须**对 bf16 group 使用 `BLOCK_N = 128, BLOCK_K = 64`；更大的 `BLOCK_N` 或 `BLOCK_K` 会触发 `MLIRCompilationError` / `hivm.hir.vcast` 不支持。

#### L1.9 non-group 大 N fp16 路径可放大 block size
- 当 `x.dtype == fp16` 且 `N > 1024` 时，可安全使用 `BLOCK_N = 256, BLOCK_K = 128`。
- 当 `x.dtype == bf16` 或 group 模式时，`BLOCK_N = 256` 容易触发编译错误，应保持 `BLOCK_N = 128`。

#### L1.10 non-group 大 M 路径必须解耦为「dequant workspace + 纯 GEMM」（msprof Roofline 验证）
- **必须**对 `is_group == False` 且 `M >= 256` 的 case 走**两 kernel 解耦**路径：
  1. `dequant_weight_kernel`：int8 weight 一次性 dequant 到 low_dtype workspace `[K, N]`（消除 fused kernel 中 dequant 被 M/BLOCK_M 个 tile 重复执行的冗余）；
  2. `matmul_kernel`：纯 GEMM `x[M,K] @ w_fp[K,N]`（+bias），K 循环无 dequant 依赖，Cube 可软件流水转为 compute-bound。
- **禁止**对小 M（`M < 256`）的 non-group 路径使用解耦：无 dequant 冗余可消，workspace 分配/读写开销主导，实测劣化。
- **Why（msprof 证据）**: fused kernel 经 `msprof op --aic-metrics=Roofline` 判定为 `latency bound: pipeline caused`，Cube ratio 仅 **7.6%（bf16）/ 15.8%（fp16 大 N）**，MTE2 21–24% 未饱和、compute <20%。根因是 int8→low_dtype dequant 位于每个 `tl.dot` 关键路径且跨 M-tile 重复执行，Cube↔Vector 同步等待饿死 Cube。解耦后 non-group 大 M case 实测 geomean 0.5807→0.6240。
- **UB 约束**: dequant kernel 用 `BLOCK_K=64, BLOCK_N=128`（fp32 中间缓冲 + group 全 tile scale/offset 控制 <192KB）。纯 GEMM BLOCK 选择见 **L1.12**（修正早期错误：BM128×BN256×BK128 恰好容纳 192KB 双 buffer，是最优配置；**仅 BM256×BN256 溢出**）。
- **精度保障**: workspace dequant 与 fused 逐元素数学完全一致（bf16 走 fp32 中间域），48/48 verify 全过。

#### L1.11 group 路径可在「结构信号」命中时解耦，但 dequant 必须用标量 g（msprof 验证，修正 L1.10 旧禁令）
- ⚠️ **本条修正 L1.10 早期的「禁止对 group 解耦」禁令**——该禁令是基于首版 group 解耦的**逐元素 gather dequant** 实测劣化得出的，但根因是 gather 而非 group 本身。修正后 group 解耦在结构信号命中时显著收益。
- **结构信号（命中任一即对 group 解耦）**:
  1. `group_size <= 64`：fused kernel 被迫 `BLOCK_K <= group_size`（16/32/64），Cube 在微 dot 上严重饿死；
  2. `x.dtype == bfloat16`：fused 把 fp32 dequant 留在 dot 关键路径（Vector/MTE2 主导，msprof fixpipe 仅 0.5%——bf16 真实瓶颈是关键路径 dequant 而非 fixpipe）；
  3. `M >= 512`：M-tile dequant 冗余高 + `[K,N]` workspace 摊销好。
- **保留 fused group** 的场景：`fp16 + group_size >= 128 + M < 512`（fused 本就用 `BLOCK_K=128`，解耦 workspace 开销略亏）。
- **关键实现约束——dequant 必须用标量 g + 单行连续 scale 加载**:
  - host 选 `DQ_BLOCK_K` 使 `DQ_BLOCK_K | group_size`，整个 tile 落在单一 group；
  - kernel 内 `g_scalar = k_start // group_size`（标量），加载单条连续 `[1,N]` scale/offset 行并广播；
  - **禁止** `g = offs_k // group_size`（`[BLOCK_K]` 向量）做 2D gather——离散访存使 dequant 慢 ~4x，首版因此对全部 group case 劣化（0 胜 15 负）。
- **msprof 证据**: fused group case Cube 常 <8%，标量 g 解耦后 geomean 0.6240→0.6770（+8.5%），11 个 group 解耦 case 全部净收益、0 回退。

#### L1.12 解耦 GEMM 必须用 BLOCK_N=256（大 N bandwidth-bound，msprof Roofline 验证）
- **必须**对解耦路径的纯 GEMM（`matmul_kernel`）使用 `BLOCK_N = 256, BLOCK_K = 128`（BLOCK_M 自适应 32/64/128），适用于 fp16 与 bf16（matmul 只加载 low_dtype，UB 占用与 dtype 无关）。
- **Why（msprof 证据）**: 大 N 解耦 `matmul_kernel` 为 bandwidth bound：`Cube 20% / MTE2 54% / aic_main_mem_read_bw ~1.6 TB/s 聚合`（HBM 打满）。根因是 dequant workspace fp16 `[K,N]`（64–128MB）远大于 ~96MB L2，被每个 M-tile 从 HBM 重读。放大 BLOCK_N 128→256 减半 N-tile 数，纯 GEMM geomean 0.6770→0.7946（+17.4%）。
- **UB 边界**: `BM=128 × BN=256 × BK=128` 双 buffer = 192KB（恰好容纳）。**仅 `BM=256 × BN=256` 溢出**。
- **⚠️ 否决方向（避免重复尝试）**:
  1. 改 tile 遍历序为 n-slow/m-fast 企图 L2 复用 workspace——实测更慢；L2 跨核共享不足以摊销大 footprint。
  2. 大 N 改回 fused 靠 int8 权重命中 L2——多数更差；大 M 时 dequant 在 dot 关键路径主导，fused 同样不命中 L2。
- **适用判据**: `MTE2 > 40% 且 Cube < 30% 且 read_bw≈HBM 峰值` → bandwidth bound → 放大 BLOCK_N。

#### L1.13 大 N 解耦路径必须做 N-strip chunking（L2-exceeding workspace 的算法级解法，refactor_iter_20）
- **必须**对解耦路径（L1.10/L1.11 命中）且 `K * N * sizeof(low_dtype) > L2_THRESHOLD`（取 **60MB**，~96MB L2 留 1/3 余量）的 case 走 **N-strip 切分**：把 N 切成多个 strip，每个 strip 的 dequant workspace `[K, chunk_n]` 能完整驻留 L2，再对该 strip 跑全部 M-tile 的 GEMM——此时 workspace 从 L2 命中而非 HBM 重读。
- **Why（修正 L1.12 末尾"workspace 重读是固有代价"的判断——该判断在 N-strip 下不再成立）**: L1.12 的 BLOCK_N=256 把单次 GEMM 访存减半，但整个 `[K,N]` workspace（大 N 时 64–128MB）仍远大于 ~96MB L2，被每个 M-tile 从 HBM 重读 8×。**正确破法是 N-strip**：每次只物化一个 strip 的 workspace（32MB ≤ L2），让它跨 M-tile 全程命中 L2，把 HBM 重读从 "M-tile 数 × 全 workspace" 降到 "num_strips × strip workspace"。
- **实现要点（deliverable 实测 1.48–1.75× on c4/37/45/46/47/48）**：
  - 触发条件：`K * N * sizeof(low_dtype) > 60MB`（约 ~96MB L2 的 2/3）。
  - 计算单 strip 大小：`chunk_n = floor(48MB / (K * dtype_size))`，再向下对齐到 `BLOCK_N_GEMM=256` 的整数倍，保证至少一个完整 N-tile。
  - Host 侧按 `chunk_idx` 循环 N 维：对每段 `weight[:, n_start:n_end]` 启动 `dequant_weight_kernel` 写 `[K, chunk_n]` workspace；再启动 `matmul_kernel` 让**所有 M-tile 顺序消费该 strip**。workspace 全程驻留 L2，跨 M-tile 命中复用，避免整 `[K,N]` buffer 被反复从 HBM 重读。
  - GEMM tile 仍用 `BM128/BN256/BK128`（L1.12）。
  - 输出按 N 维 slice 写入 `out[:, n_start:n_end]`，不同 strip 输出 disjoint，**禁止用 atomic_add**（相对 K-split 的核心优势）。
  - strip 内 dequant 仍遵守 L1.11：group 模式用标量 `g = k_start // group_size` + `DQ_BLOCK_K | group_size` + 单行连续 scale/offset 加载，`DQ_BLOCK_N=128`。
- **触发 case 实测（repeats=50）**: c4 M=1024 K=4096 N=8192 → 1.481×；c37/47 M=1024 K=4096 N=16384 → 1.58–1.59×；c45 → 1.752×；c46/48 M=2048 K=8192 N=4096 → 1.56–1.60×。相对 opt_iter_19（这些 case 仅 ~0.54×）实现质变，geomean **0.7946 → 0.9133**（+15%）。
- **适用判据**: 解耦路径 + `K*N*2 > 60MB` → 必走 N-strip；`K*N*2 ≤ 60MB` 走常规单 workspace 解耦（L1.12）即可（strip 切分对能装进 L2 的 workspace 反而增加 host 循环开销）。

### §4.2 Layer 2: 算法骨架（纯描述，无完整源码）

#### L2.1 Host 侧分派

`forward()` 仅负责：
1. 输入 `contiguous()` + reshape 为 2D `[M, K]` / `[K, N]`；
2. 按 `antiquant_group_size>0` 判定 `is_group`；
3. 按 dtype / shape 选择 `(BLOCK_M, BLOCK_N, BLOCK_K)`；
4. 启动对应 kernel。

分派策略表（遵守 L1.8–L1.13）：

| 场景 | kernel | BLOCK_N | BLOCK_K | BLOCK_M |
|------|--------|---------|---------|---------|
| non-group, N≤1024 或 bf16 | fused | 128 | 128 | 32/64/128 按 M |
| non-group, N>1024, fp16 | fused | 256 | 128 | 32/64/128 按 M |
| group, bf16 | group | 128 | 64 | 32/64/128 按 M |
| group, gs%128==0, fp16 | group | 128/256 | 128 | 32/64/128 按 M |
| 其他 group | group | 128 | 64 | 32/64/128 按 M |

解耦判定见 L1.10–L1.13：命中解耦时先启动 `dequant_weight_kernel` 写 workspace，再启动 `matmul_kernel` 做纯 GEMM；命中 N-strip 时 host 侧循环 N 维 strip。

#### L2.2 Group kernel 骨架

1. **1D grid + 核内 stride 循环**：`pid = program_id(0)`，`for tile_idx in range(pid, total_tiles, num_programs)`。
2. 每个 tile 取 `m_tile = tile_idx // num_n_tiles`，`n_tile = tile_idx % num_n_tiles`。
3. K-loop 内用**标量 group id**：`g = k_start // group_size`；按 `g` 加载单条 `[1, N]` scale/offset，广播到 `[BLOCK_K, BLOCK_N]`。
4. 每个 K-block 必须满足 `BLOCK_K | group_size`（L1.3），保证不跨 group。
5. dequant 域按 `IS_BFLOAT16` 分流（L1.4）：fp16 在 fp16 域，bf16 在 fp32 域。
6. 累加用 fp32 accumulator，`tl.dot(x_tile, w_tile, acc)`；最后加 bias 并 cast 回 `x.dtype` store。

#### L2.3 Non-group kernel 骨架

1. 同样 1D grid + stride 循环。
2. scale/offset 在 K 维恒定，**提到 K 循环外预加载 + cast**（L3.4）：
   - 预计算 `scale_f32` / `scale_low`（fp16 路径）和 `offset_f32` / `offset_low`（若存在 offset）。
3. K-loop 内只做：load int8 weight → cast → `*` scale `(+ offset)` → cast → `tl.dot`。
4. 最后加 bias、cast、store。

#### L2.4 N-strip chunking 骨架（解耦路径）

当 `K*N*sizeof(low_dtype) > 60MB` 时：
1. 计算 `max_chunk_n = 48MB // (K * dtype_size)`，向下对齐到 `BLOCK_N_GEMM=256`。
2. Host 侧 `for chunk_idx in range(cdiv(N, n_chunk))`：
   - 取 `weight[:, n_start:n_end]`；
   - 启动 `dequant_weight_kernel` 写 `w_fp_slice[K, chunk_n]`；
   - 启动 `matmul_kernel` 计算 `x @ w_fp_slice → out[:, n_start:n_end]`。
3. 所有 M-tile 顺序处理同一个 strip，workspace 在 L2 中命中复用；输出 N 维 disjoint，**无需 atomic**。

### §4.3 Layer 3: 关键技巧（Agent 可参考但不可复制代码结构）

#### L3.1 group scale 单 row 广播替代 host 物化

**问题**: group 模式若将 `[G, N]` expand 为 `[K, N]`，会占用大量 host 时间与 HBM 带宽。

**解决**: kernel 内按 `g = k_start // group_size` 只加载一条 `[1, N]`，通过 `scale_row[None, :]` 广播到 `[BLOCK_K, N]`。

**可替代方向**: 若未来后端支持更高效的 gather，可尝试 vector gather；但在当前 Ascend 后端，单 row 广播更稳定。

#### L3.2 按 `group_size` 选择 `BLOCK_K`

**问题**: group_size 为 64 与 128 时，最优 `BLOCK_K` 不同。

**解决**:
- `group_size % 128 == 0` 且 dtype 为 fp16: `BLOCK_K = 128`，减少 K 循环迭代数。
- 其他 group 场景: `BLOCK_K = 64`，避免跨 group 并保证编译通过。

**可替代方向**: 对固定 group_size 可离线搜索最优 `BLOCK_K`，但需保证不跨 group。

#### L3.3 按 dtype / shape 分 block 策略

| 场景 | 推荐 BLOCK_N | 推荐 BLOCK_K | 原因 |
|------|-------------|-------------|------|
| fp16 non-group, N > 1024 | 256 | 128 | 大 tile 提升 Cube 利用率 |
| fp16 non-group, N <= 1024 | 128 | 128 | 平衡 |
| bf16 non-group | 128 | 128 | BLOCK_N=256 会触发 MLIRCompilationError |
| fp16 group, gs%128==0 | 128/256 | 128 | 单 group 单 block |
| bf16 group | 128 | 64 | 保守，避免编译错误 |

**可替代方向**: 对小 M (<64) 使用 `BLOCK_M=32`，中等 M (64-127) 使用 `BLOCK_M=64`，大 M 使用 `BLOCK_M=128`；`BLOCK_M=256` 易触发 UB overflow，不建议。

#### L3.4 循环不变量 cast 外提

**问题**: non-group 模式下 `scale_tile` / `offset_tile` 在 K 循环内重复 cast。

**解决**: 在 K 循环外预计算 `scale_f32` / `scale_low` / `offset_f32` / `offset_low`，K 循环内直接使用。

**可替代方向**: 现代编译器可能自动外提，但显式外提更可控，实测有小幅收益。

### §4.4 性能基准与优化尝试记录

| 版本 | cases | geomean | 相对 Phase 3 基线 | 说明 |
|------|-------|---------|------------------|------|
| Phase 3 基线 | 48/48 | 0.1026x | 1.0x | 通用 kernel，host 物化 group scale |
| opt_iter_9 (kernel 分裂 + group 单 row 广播) | 48/48 | 0.4558x | 4.44x | 首次算法级优化，group 路径去掉 host 物化 |
| opt_iter_10 (fp16 大 N BLOCK_K=128) | 48/48 | 0.4979x | 4.85x | non-group 大 N 同时放大 N/K tile |
| opt_iter_11 (group_size=128 BLOCK_K=128) | 48/48 | 0.5290x | 5.16x | group_size=128 的 fp16 group 减少循环 |
| opt_iter_12 (bf16 non-group BLOCK_K=128) | 48/48 | 0.5582x | 5.44x | bf16 非 group 放大 K tile |
| opt_iter_14 (fp16 group N>=1024 BLOCK_N=256) | 48/48 | 0.5705x | 5.56x | fp16 group N=1024 时放大 N tile |
| **opt_iter_15b (cast 外提)** | **48/48** | **0.5807x** | **5.66x** | fused 最终版本 |
| opt_iter_16 (纯解耦 dequant+GEMM) | 48/48 | 0.3640x | 3.55x | 全 case 解耦：non-group 大 M 赢，group/小 M 大输 → 总体劣化（group 输因逐元素 gather dequant，见 opt_iter_18 修正） |
| opt_iter_17 (mixed: 解耦 non-group M>=256 + fused 其余) | 48/48 | 0.6240x | 6.08x | msprof Roofline 驱动 non-group 大 M 解耦 |
| opt_iter_18 v1 (group 解耦-逐元素 gather) | 48/48 | 0.4275x | 4.17x | group 解耦首版：dequant 用 `g[:,None]` 2D gather，15 个 group case 全劣化（0 胜 15 负） |
| **opt_iter_18 (group 解耦-标量 g + 结构路由)** | **48/48** | **0.6770x** | **6.60x** | **msprof 驱动 group 路径解耦：标量 g 连续 scale 加载消除 gather；gs<=64/bf16/M>=512 解耦，其余 fused** |
| **opt_iter_19 (解耦 GEMM BLOCK_N=256)** | **48/48** | **0.7946x** | **7.74x** | **msprof 驱动：大 N 解耦 GEMM bandwidth-bound（MTE2 54%, Cube 20%, HBM 打满），workspace(64-128MB)>L2 被 M-tile 重读；放大 BLOCK_N 128→256，纯 GEMM +36-38%** |
| **refactor_iter_20 (N-strip chunking, 最终 deliverable)** | **48/48** | **0.9133x**（方差带 0.91-0.95） | **8.90x** | **算法级消除 L2-exceeding workspace 重读：`K*N*2>60MB` 时按 N 切 L2-resident strip（48MB 上限，对齐 BN256），每 strip dequant 一次 → 全部 M-tile L2 命中复用；输出 N 维 disjoint 无需 atomic。大 N case 0.54→1.48-1.75×（c4/37/45/46/47/48）** |

**有效优化点**:
1. group / non-group kernel 分裂 + group 单 row 广播（最大收益，4.4x）
2. fp16 大 N 使用 BLOCK_N=256 / BLOCK_K=128
3. 按 group_size 选择 BLOCK_K=128（gs=128 时）
4. bf16 non-group 使用 BLOCK_K=128
5. 循环不变量 cast 外提（小幅收益）
6. **non-group 大 M 解耦为 dequant workspace + 纯 GEMM**（msprof 驱动，0.5807→0.6240）：消除 dequant 跨 M-tile 重复执行，Cube 从 latency-bound(7.6-15.8%) 转 compute-bound
7. **group 路径解耦（标量 g + 结构路由，0.6240→0.6770）**：gs<=64/bf16/M>=512 命中时解耦；dequant 用标量 g（`DQ_BLOCK_K | group_size`）+ 单行连续 scale 加载，禁用逐元素 2D gather；fp16/gs>=128/M<512 保留 fused
8. **解耦 GEMM 放大 BLOCK_N=256（大 N bandwidth-bound，0.6770→0.7946）**：msprof 显示大 N 解耦 GEMM 是 bandwidth bound（MTE2 54% / Cube 20% / HBM 打满），workspace(64-128MB)>L2 被 M-tile 重读；放大 BLOCK_N 128→256，纯 GEMM +36-38%（case37/46/4）
9. **N-strip chunking（L2-exceeding workspace 算法级解法，0.7946→0.9133）**：L1.12 的 BLOCK_N=256 只减半单次访存，整 workspace 仍被 M-tile 重读。N-strip 把 N 切成 L2-resident 的 strip（48MB 上限），每 strip dequant 一次供全部 M-tile L2 复用，输出 N 维 disjoint 无需 atomic。大 N case 0.54→1.48-1.75×（详见 L1.13）

**已验证无效/失败的优化方向**:

| 优化方向 | 结果 | 原因 |
|----------|------|------|
| group scale host 物化为 `[K, N]` | 劣化 | host CPU 时间与额外 HBM 写入 |
| group 模式 in-kernel per-K gather（未广播） | 劣化 | gather 开销大，0.0735x |
| bf16 non-group / bf16 group **fused** BLOCK_N=256 | 编译失败 | `hivm.hir.vcast` unsupported / UB overflow（仅限 **fused** kernel 的 fp32 dequant 中间体；解耦 GEMM 的 BN=256 不受此限，见 L1.12） |
| bf16 group BLOCK_K=128 | 编译失败 | `hivm.hir.vcast` unsupported |
| BLOCK_M=256 × BLOCK_N=256 | UB overflow | `cc overflow`（仅 BM256×BN256 组合溢出；BM128×BN256×BK128 恰好容纳 192KB） |
| 解耦 GEMM tile 遍历序改 n-slow/m-fast（L2 复用 workspace N-strip） | 更慢 | case37 GEMM 0.94 vs 0.85ms；L2 跨核共享不足以摊销 128MB footprint |
| 大 N 改回 fused（靠 int8 权重 64MB 命中 L2） | 多数更差 | 仅 case45 略胜；大 M 时 dequant 在 dot 关键路径主导，fused 重读 int8 8× 同样不命中 L2 |
| **scale-hoisted fused / fold-out（post-deliverable 重构）** | 35/48 挂 | 把 per-channel scale 移出 K-loop 直读 int8 权重省 workspace 往返。no-offset 精度超阈（fold-out 比 torch_npu 更精）；offset 触发 `hivm.hir.vcast root alloc` UB overflow。精度与编译双重阻塞 |
| **BLOCK_M=64 并行填充（post-deliverable 重构）** | 0.9437→0.7903(-0.15) | 推翻并行假设：decoupled matmul 是 per-tile Cube 效率受限而非 aggregate-MLP 受限，[128,128]@[128,256] 是高效 Cube dot shape，BM=64 崩 Cube 利用率 |

**关键结论**:
1. **Kernel 分裂是架构基础**：group 与 non-group 的 scale 访问模式差异巨大，必须分 kernel。
2. **group scale 单 row 广播是核心优化**：避免 host 物化和 gather，是收益最大的单点改进。
3. **block size 必须按 dtype/shape 分派**：bf16 和 group 模式对大块更敏感，fp16 non-group 可激进放大。
4. **BLOCK_M=256 仅在与 BLOCK_N=256 组合时溢出**：`BM=256×BN=128` 单独可行；但解耦 GEMM 实测 `BN=256×BM=128` 更优（N-tile 减半胜于 M-tile 减半），故 BLOCK_M 仍保持 128。
5. **dequant 域必须按 dtype 分流**：fp16 在 fp16 域，bf16 在 fp32 域，否则精度 fail。
6. **msprof Roofline 是判定 latency-bound 的权威手段**：fused kernel 经 profiling 判定 `latency bound: pipeline caused`（Cube 7.6-15.8%），直接驱动「non-group 大 M 解耦 + 纯 GEMM」重构，geomean 0.5807→0.6240。后续 quant 类 matmul 若 Cube<20% 且带宽未饱和，应优先考虑解耦 dequant。
7. **解耦须按 case 分流（mixed strategy），且 group 解耦成败取决于 dequant 访存模式**：早期「group 解耦必劣化」的结论是**错误**的——根因是首版用了逐元素 2D gather dequant（离散访存慢 ~4x）。改用标量 g + 连续行加载后，group 解耦在结构信号（gs<=64 / bf16 / M>=512）命中时显著收益（opt_iter_18：11 个 group case 全净收益，0 回退）。仍须保留 fused 的场景：fp16+gs>=128+M<512（fused 本就 BLOCK_K=128，workspace 开销略亏）。
8. **bf16 真实瓶颈是关键路径 dequant 而非 fixpipe**：msprof 显示 fused bf16 的 fixpipe 仅 0.5%，bf16 慢是因为 fp32 dequant 留在 dot 关键路径（Vector/MTE2 主导）。解耦即消除——最差 bf16 group case 由 group 解耦一并修复（c28 0.283→0.519, c29 0.279→0.607, c36 0.518→0.910）。
9. **目标 0.8x 已达成（refactor_iter_20 实测 0.9133x，方差带 0.91-0.95，超目标 14%）**：N-strip chunking（L1.13）是破局点——把大 N 解耦 case 从 opt_iter_19 的 ~0.54 拉到 1.48-1.75×，彻底解决 L1.12 遗留的 "workspace > L2 被 M-tile 重读" 问题。剩余拖低 geomean 的 laggard 转移到 **medium-N 解耦 case（c2/12/25/28/41/44，0.62-0.79）**，这些 case 见结论 11。
10. **bandwidth-bound 与 compute-bound 的 msprof 区分（L1.12）**：解耦后 GEMM 的瓶颈从 fused 的「Cube 饿死（latency-bound, Cube<16%）」转为「HBM 打满（bandwidth-bound, MTE2 54% / Cube 20% / read_bw 接近峰值）」。前者靠解耦/放大 tile 提升 Cube，后者放大 BLOCK_N 减半访存次数。**判据**：msprof `MTE2>40% 且 Cube<30% 且 read_bw≈HBM 峰值` → bandwidth bound → 放大 BLOCK_N；若 workspace 体积>L2，重读无法靠遍历序（n-slow/m-fast）消除（已实测更慢）。
11. **medium-N laggard 已到 Triton toolchain 天花板**：refactor_iter_20 达 0.9133x 后，剩余 laggard 集中在 M∈[200,512] 的 medium-N 解耦路径（0.62–0.79）。msprof 显示该路径 `latency bound: memory caused` 且 MTE2 带宽余量 <80% peak，说明有 headroom 但 Triton 抓不到。两条算法重构均证伪：scale-hoisted fused 被精度（no-offset）和编译（offset UB overflow）双重阻塞；BLOCK_M=64 并行填充 -0.15 崩 Cube 效率。medium-N 剩余 gap 到 1.0 是 **torch_npu 硬件 dequant-on-load 单 kernel 的结构性优势**——Triton 无法软件流水 dequant-across-dot（vmul/vcast root-alloc rejection），也无法访问硬件 dequant 路径。**后续 medium-N 量化 matmul 若解耦路径 <0.8x，不要调参/fold-out/BM缩放，直接标注 toolchain ceiling**。

---
