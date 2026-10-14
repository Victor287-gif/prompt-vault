---
type: Guide
title: matrix_compute 类算子设计约束
description: matrix_compute 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.compute.composite, topic.compute.quantization, topic.tiling, topic.compute.elementwise.activation, topic.precision]
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

# matrix_compute 类算子设计约束

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## 矩阵计算类算子优化经验

本文档合并矩阵计算类算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 QuantMatmul**（quantization-matmul，int8 矩阵乘 + 反量化）
- **§3 GroupedMatmulSwigluQuant**（grouped-matmul-swiglu-quant，分组 int8 矩阵乘 + SwiGLU + 动态量化）
- **§4 常见陷阱**（按算子分小节）
- **§5 相关记忆**

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/matrix_compute__groupedmatmulswigluquant.md`](../operator_playbooks/matrix_compute__groupedmatmulswigluquant.md) — GroupedMatmulSwigluQuant 算子（grouped-matmul-swiglu-quant）
- [`operators/matrix_compute__groupedmatmulswigluquantv2.md`](../operator_playbooks/matrix_compute__groupedmatmulswigluquantv2.md) — GroupedMatmulSwigluQuantV2 算子（grouped matmul + SwiGLU + quant，MoE 专家分组矩阵乘）
- [`operators/matrix_compute__quantmatmul.md`](../operator_playbooks/matrix_compute__quantmatmul.md) — QuantMatmul 算子（quantization-matmul）
- [`operators/matrix_compute__ffn.md`](../operator_playbooks/matrix_compute__ffn.md) — FFN 算子（feed-forward-network，两级 GEMM + 门控激活 + int8 量化 + MoE）

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| QuantMatmul | `quantization-matmul` | int8×int8 矩阵乘，后接反量化（scale/pertoken_scale）、bias、offset、cast 到 fp16/bf16/int8/int32 | int32 累加保精度 + 动态 tile + 小 shape 限 grid + 后处理顺序严格对齐参考 |
| GroupedMatmulSwigluQuant | `grouped-matmul-swiglu-quant` | 分组 int8 矩阵乘 + SwiGLU 激活 + per-row 动态量化到 int8 | 3-kernel 结构保精度 + Grouped MatMul cube 利用率 + dquant 标量链消除 |
| GroupedMatmulSwigluQuantV2 | `grouped matmul` + `activation` + `quantization` | `x [M,K] int8` 按 group_list（cumsum/count）切 E 个 expert 区间 @ `weight [E,K,N] int8` + per-channel scale → SwiGLU → per-token int8 quant | expert 边界对齐 BLOCK_M 合并（dot M 维 ≥16）+ 扁平 (m_tile,n_tile) 并行 + partial row_max 免 atomic + BLOCK_K 按 dequant 路径分档；fp16 预缩放路径 BK 锁死 256 |

> ⚠️ **关键区分**：矩阵计算类算子的核心瓶颈通常是 **Cube 利用率、UB/L0 容量约束和 int8 量化语义下的精度路径**，而非单纯访存。优化时必须优先保证 int32 累加、后处理顺序、scale 截断方式与参考实现完全一致；在此基础上再调整 tile、grid 和 kernel 拆分策略。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束是矩阵计算类算子**共有**且**未在其他 template 文件覆盖**的工程约束。其他文件已提取的通用约束（如 GC-1）此处不再重复，各算子章节引用时标注。

### G1 动态读取核数，禁止硬编码

```python
VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)
CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get('cube_core_num', 24)
```

- MatMul / grouped matmul 使用 `CUBE_CORE_NUM`；后处理 / activation / quant 使用 `VEC_CORE_NUM`。
- **Why:** 硬编码核数会导致不同芯片/驱动版本上 grid 不匹配，轻则负载不均，重则结果错误。

### G2 1D Grid + 核内循环

- `grid = (min(total_blocks, num_cores),)`
- kernel 内 `for block_idx in range(pid, num_blocks, num_programs)` 交织循环
- **禁止**多维 grid 或 `grid=(M,)` 导致 program 过饱和
- **Why:** Ascend NPU 上固定核心数启动、每个核心循环处理多个块，是最稳定的并行模式；`grid=(M,)` 会造成大量空 program 和调度开销。

### G3 Host 侧缓存 driver / stream

```python
self._driver = triton.runtime.driver.active
self._stream_cache = {}
```

- 在 GroupedMatmulSwigluQuant 上实测 geomean +3%。
- **Why:** 多 expert、小 shape 场景下 enqueue 开销占比高，缓存 driver/stream 可减少重复查询。

### G4 禁止在 forward() 中做 torch/torch_npu 计算

- 所有数值计算必须在 `@triton.jit` kernel 内完成，否则触发 Type-3 退化检查。
- shape 计算、分派、host 预计算、buffer 分配可以放在 forward() 中。
- **Why:** Triton-Ascend 算子要求核心逻辑用 Triton 表达；forward 中调用 torch.matmul / torch.softmax 等会被判退化。

### G5 BLOCK 向上取 2 的幂

- `BLOCK_N` 等向量化长度应声明为 `triton.next_power_of_2(N)` 并作为 `tl.constexpr` 传入，避免 dynamic-shape load 退化为标量循环。
- **例外**：当实际维度本身不是 2 的幂且接近 UB 边界时（如 GroupedMatmulSwigluQuant 的 `HALF_N=1280`），应直接用原值而非 next_pow2，否则触发 UB overflow。

### G6 int8 矩阵乘必须使用 int32 accumulator

- **必须**使用 `tl.dot(a, b, acc)`，accumulator dtype 为 `tl.int32`。
- **禁止**使用 fp32/fp16 accumulator 或直接转浮点后做 matmul。
- **Why:** torch_npu 量化 matmul 在 matmul 阶段使用 int32 累加，后处理阶段再转 fp32 反量化；浮点累加器会改变舍入行为，导致 fp16 输出 case 精度不过。

### G7 Batch 维度必须 flatten 到 3D

- 必须将 x1/x2 的 leading dims 展平为 B，reshape 为 `[B,M,K]` / `[B,K,N]`。
- **禁止**在 kernel 内处理 4D/5D/6D 索引。
- **Why:** kernel 内降为 3D 逻辑能最大化向量化与并行度；多维坐标解码会引入标量除法/取模和非合并访存。

### G8 禁止在 kernel 内使用 `%` 模运算

- 解码 1D grid 三维索引时，必须改用 `a - (a // b) * b`。
- **Why:** `%` 在部分后端会被 lower 为标量循环或产生额外开销；`a-(a//b)*b` 更稳定。

---

## §4 常见陷阱与避免方法

### §4.1 QuantMatmul 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| fp32/fp16 accumulator 做 int8 matmul | 改变舍入行为 | L1.1 / G6 |
| 所有 shape 使用单一固定大 tile | 小 shape 调度 overhead 大 | L1.2 |
| 小 shape 固定满核 grid | 大量空 program | L1.3 |
| fp16 输出 scale 四舍五入截断 | 与参考实现不一致 | L1.4 |
| 后处理顺序错误 | scale/pertoken/bias/offset 语义错位 | L1.5 / L1.6 |
| forward() 中保留 nonzero/index_put/nextafter | host 辅助算子链退化 | L1.7 |
| kernel 内处理 4D/5D/6D 索引 | 标量坐标解码 | L1.8 / G7 |
| 单次 benchmark 采信异常 case | profiler 干扰 | §2 复现注意 |

### §4.2 GroupedMatmulSwigluQuant 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 融合 SwiGLU + dquant | 数值链系统性差异，0/60 失败 | L1.1 |
| gmm_dot 中乘 x_scale | 与 npu_grouped_matmul 语义不一致 | L1.2 |
| dquant 使用 `tl.isnan` | 被 lower 为标量循环 | L1.4 |
| BLOCK_N=512 配 BLOCK_K=512 | K≥7168 时 cbuf overflow | L1.5 |
| HALF_N=1280 时用 next_power_of_2 | 2048 触发 UB overflow | L3.1 |
| 激进动态 GMM tiling | 整体劣化或编译失败 | §3.4 |

---

## §5 相关记忆

- [[lightning-indexer-triton-gen]] — LightningIndexer 精度根因（tie-break + bf16 RNE）
- [[sfa-golden-native-vs-purepytorch]] — SFA 精度基准选择
- [[fusion-attention-optimization-record]] — 稠密 attention 优化记录（与量化 matmul 共用部分 tiling 思想）
---

## §6 BatchMatmul（bmm）算子（微小 shape batched GEMM）

> 2026-09-02 从 `1_BatchMatmul`（51 shape，fp32/fp16/bf16，batch≤8、M/N≤128、K≤256）实测提炼。终态 geomean 0.9105x（基线 0.8899x）。

### Layer 1（设计约束，硬性）

- **L1.1 索引 mask 禁止转 fp32**：`off_m < M` 这类 tl.arange 索引 mask 必须保持 int32。按「int32 CMP→scalar fallback」建议转 fp32 比较会**aicore 挂死**（非慢，是执行超时，且污染设备致连带失败）。SCALAR 降级是性能问题，不是正确性修复手段。
- **L1.2 对齐 case 必须走无 mask 快路径**：`M%BLOCK_M==0 && N%BLOCK_N==0 && K%BLOCK_K==0` 时禁止带 mask load/store——微小 shape 下 mask 计算是主要标量开销（simulator 实测 SCALAR pipe 41-55%）。用**单一 `ALL_ALIGNED` constexpr 标志**分派（Triton-Ascend 不支持链式布尔 `A and B and C`，须 host 合并为单标志）。
- **L1.3 微小 shape 禁止 tile 拆分增 program**：launch-bound 场景拆分大 shape 成多 tile 整体劣化（实测 -25%）；仅当单 case 计算量足够（如 fp32 128×256×128）时按 dtype 混合策略拆分才有效。
- **L1.4 查询核数必须 `import torch_npu`**：`torch_npu.npu.current_device()` 缺 import 会全 case NameError。

### Layer 2（算法骨架）

- 1D persistent grid + kernel 内 batch 解码：`tile → (b, m_blk, n_blk)`，`b=tile//tiles_per_batch`，`rem=tile-b*tiles_per_batch`（禁止 `%`）。每 program 一个 `[BM,BN]` 输出 tile，K 维核内循环。禁止 3D batched tl.dot（GC-15）。
- Host 按 `max(M,N)` 分档 `BM/BN ∈ {16,32,64,128}`；`BLOCK_K` 按 tile 大小 + dtype 分档：中小 tile（BM,BN≤64）fp32 封顶 128、fp16/bf16 封顶 256；128-tile fp32 封顶 64（UB 预算）、fp16/bf16 封顶 128。
- fp16/bf16/fp32 均保持原生 dtype 进 `tl.dot`，累加器恒 fp32，store cast 回原 dtype（tl.dot 契约）。

### Layer 3（关键技巧）

- **BLOCK_K 分档减 dot 调用数**：微小 shape 每 program 的 dot 调用数是关键开销。对中小 tile 放大 BLOCK_K（fp32 64→128、fp16/bf16 128→256）减少 K 迭代，实测 geomean +2.3%，mid 区破 1.0。
- **对齐快路径**（可替代方向：block_ptr boundary_check，但需验证编译稳定性）：
  ```python
  # host: ALL_ALIGNED = (M%BM==0) and (N%BN==0) and (K%BK==0)
  # kernel: if ALL_ALIGNED: 无 mask load/store; else: 带 mask
  ```
  simulator 确定性确认对齐路径 Cube cyc -3.8%、Vector -16~32%。

### §6.1 陷阱表

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 索引 mask 转 fp32 | aicore 挂死 + 设备污染 | L1.1 保持 int32 |
| 链式布尔 constexpr if | UnsupportedLanguageConstruct | 单一 constexpr 标志 |
| 大 shape 拆 tile | launch-bound 下 program 调度开销 | L1.3 按 dtype 混合策略 |
| 跨运行 benchmark 对比 | 并发任务噪声污染 geomean | 以确定性 simulator + 无尖峰运行为准 |

### §6.2 复跑并列经验（2026-09-04，1_BatchMatmul 同 51 shape，Phase3 基线 1.3355 → 1.5891x）

> 与上方 §6（0.9105x 归档）为**同一算子的两次独立探索**，按多样性保护并列记录。
> 本轮先做 #1 全参数 constexpr 静态化，再叠加 #17 mask 裁剪；个别结论与旧 Layer3 相左，已标注。

**Layer 1（追加硬性）**
- **L1.5 全参数 constexpr 静态化是最优先杠杆（本轮 +16%，1.3355→1.5488）**：`B/M/N/K/stride` 全部声明 `tl.constexpr`，触发启动级特化——K 循环编译期 unroll、tile decode 除法与 stride 乘法常量折叠。微 GEMM 全体 ~2µs launch/dot-issue bound，此项直接削减每 program 标量算术，收益大于任何 tile 调整。51 shape 各生成特化二进制可接受。

**Layer 3（追加技巧）**
- **BLOCK_K 放大在本轮为负（与旧 §6 Layer3 "+2.3%" 相左）**：本轮 fp16/bf16 BK cap 128→256、fp32 64→128（K=256 单次 dot），geomean 1.5891→1.5563 **回退**。旧档 +2.3% 发生在 128-tile 大 shape 簇；本轮 51-shape 混合集（中小 tile 居多）分段 dot 非瓶颈，维持 BK cap 128(16bit)/64(fp32) 更稳。命中旧 Layer3 需先看 shape 构成。
- **分核（BM/BN cap 64→32 增 program）无效（1.5488→1.5309）**：与旧 §6 L1.3 一致，launch-bound 下拆分劣化。中小 tile cap 保持 64。

### §6.2 陷阱表

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 盲目套用旧 Layer3「放大 BLOCK_K」 | 旧档基于 128-tile 大 shape 簇，混合 shape 集回退 | 先看 shape 构成再套用 |
| 优化首轮不静态化先调 tile | 微 GEMM 标量算术主导，tile 收益被掩盖 | 先 #1 全 constexpr，再谈 tile/BK |

---

## §7 GroupedMatmul / BatchedGEMM（fp32，grouped_mm 语义）

> 2026-09-04 从 `2_GroupedMatmul`（50 shape，fp32，A 2D+offsets 等分 / A 3D 直接 batch 混合）实测提炼。
> 终态 geomean 1.0478x（Phase 3 基线 1.0068x）。

### Layer 1（设计约束，硬性）

- **L1.1 统一扁平 batched 视图**：2D+offsets(end 等分) 与 3D 直批本质是 G 组 × R 行的 batched GEMM；A 按 (G*R,K) 连续扁平寻址即可一 kernel 通吃，offsets 仅判模式不参与计算，R=A.rows//G（2D）或 A.shape[1]（3D），输出 grouped→(G*R,N)、batched→(G,R,N)。
- **L1.2 分组/小 R(R≤16) 必须 BM=16 单行块 + A 跨 N 复用**：BM=64 会让 10 行组 pad 6.4x、且 A 按 N 块重复加载，实测 20x 劣化（grouped G16 R10 K1024 N1280 0.049x）。行块 persistent + N-inner（A 每 program 只 load 一次）是必须形态。
- **L1.3 大 batched(R,N≥256) 必须保留 BM=BN=128 大 tile**：中小 tile 或 row-persistent 在全量切分会令大 shape 失 cube 效率（opt_iter_0 全量改版 geomean 0.9796 回归，hybrid 才回 1.03）。
- **L1.4 对齐 shape 必须走 ALL_ALIGNED constexpr 无 mask 分支**（同 §6 L1.2）：fp32 matmul 对齐大 shape 每 tile mask 的标量/谓词开销可观（0.65→改善）。

### Layer 2（算法骨架）

- **hybrid 双 kernel 分派**：`(R,N)>=256` → decode persistent kernel，1D grid=min(G*MT*NT, cores)，核内 range(pid,TOT,cores) 交织，tile (g,mt,nt) 解码用 `a-(a//b)*b` 免 `%`；其余 → row-persistent kernel，grid=min(G*MT,cores)，核内 `for rp in range(pid,ROW_BLOCKS,cores)` + 外层 `for nt`（A 复用）+ 内层 K 循环。
- fp32 dot 原生输入 + fp32 累加器；BM/BN∈{16,64,128}，BN 按 N 分桶，BK fp32 档（面积≤4096 时 128，否则 64，128-tile ≤64）；acc=BM*BN≤16384。

### Layer 3（关键技巧）

- 分组 case 单组行数极小时（R≈10，K/N 大），无论 tile 怎么调均受 BM 小 → dot M 维小的 cube 利用率地板限制（实测 ~0.08x 为硬地板，与 torch 预编译 kernel 差距不可消除），不要在 case11 类 shape 上反复烧轮。
- 微小 shape（K/N≤32）簇 ~0.65-0.85x 属 kernel-launch 地板（1.5-2us），非算法缺陷。

### §7.1 陷阱表

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 全量 row-persistent 覆盖大 shape | 大 tile(128) cube 效率丢失 | L1.3 hybrid 分派 |
| 分组小 R 用 BM64/BN 固定 tile | 6.4x pad + A 多读 | L1.2 BM16+复用 |
| 对齐 shape 仍带全 mask | 每 tile 标量/谓词开销 | L1.4 ALL_ALIGNED 分支 |
