---
type: Guide
title: RotaryMul 算子（rotarymul） 实现手册
description: '算子类别: rotarymul（RoPE 旋转位置编码乘法）。'
tags: [task.implementation, topic.attention, topic.tiling, topic.compute.quantization, topic.compilation, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# RotaryMul 算子（rotarymul） 实现手册

> 类别约束见 [transformer_inference_constraints.md](../constraints/transformer_inference_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 RotaryMul 算子（rotarymul）

**算子类别**: `rotarymul`（RoPE 旋转位置编码乘法）
**典型特征**: 4D 张量 `[B, H, S, D]`，支持 `half` / `interleave` 两种模式，支持 fp16/bf16/fp32，r1/r2 支持 broadcast
**性能基准**: 几何平均加速比 **1.02x** vs torch（50 cases 全通过）

### §2.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 禁止 flat-1D 索引分解
- **禁止**在 kernel 内将一维 flat index 通过 `%` 和 `//` 运算反向分解为 `(b, h, s, d)` 多维坐标。
- **Why:** Ascend 编译器会将这些 per-element 的地址计算标量化，生成大量 `scf.for` step-1 循环和 `memref<1xf16>` 标量 load/store，导致 HIVM intrinsics 利用率极低（实测仅 19.8%）。
- **How to apply:** 每个 program 直接处理一个完整的 `(b, h, s)` 位置，使用 `tl.arange` 做 contiguous vector load/store。

#### L1.2 必须显式处理 broadcast stride
- **必须**在 Host 侧计算 r1/r2 的 broadcast stride：若某维 shape 为 1 且 input 对应维 >1，则该维 stride 设为 0。
- **Why:** RotaryMul 的 r1/r2 常见 broadcast 模式（如 `[B,1,S,D]` 对 `[B,H,S,D]`），若直接传 `t.stride()` 会导致 kernel 内地址计算错误。
- **How to apply:**
  ```python
  def _bc_stride(t, input_shape):
      return tuple(0 if t.shape[i] == 1 and input_shape[i] > 1 else t.stride(i) for i in range(4))
  ```

#### L1.3 禁止在 kernel 内做 dtype 分支判断
- **禁止**在 `@triton.jit` kernel 内通过 `if dtype == torch.float16` 这类运行时分支判断数据类型。
- **Why:** Triton kernel 内无法直接访问 PyTorch dtype 对象；应通过 `tl.constexpr` 布尔标志（如 `IS_FP16`, `IS_BF16`）在编译期确定分支。
- **How to apply:** Host 侧计算 `IS_FP16 = (dtype == torch.float16)` 等标志，作为 `tl.constexpr` 传入 kernel。

#### L1.4 fp16/bf16 必须在 kernel 内升精度到 fp32 计算
（见 §1 T2）

#### L1.5 禁止 Adaptive TILE_S（编译期动态 tile 大小）
- **禁止**在 Host 侧根据 S/D 大小选择不同的 `TILE_S`（如 `TILE_S = 32 if S >= 512 and D <= 64 else 16`）。
- **Why:** `TILE_S` 作为 `tl.constexpr`，若在不同 shape 间变化会导致 kernel 被重新编译；更大的问题是过大的 2D tile（如 32x32）在 Ascend 上会被编译器标量化，造成灾难性性能退化（实测 65x slowdown）。
- **How to apply:** 固定 `TILE_S = 16`，通过 uniform grid splitting 解决负载均衡问题。

#### L1.6 必须添加 `tl.assume` 编译器提示
- **必须**对 stride 和 shape 添加 `tl.assume` 提示，尤其是 `stride_d == 1`、`half_d >= 16`、`TILE_S > 0` 等。
- **Why:** 帮助 Ascend 编译器生成 vector load/store 而非标量循环。
- **How to apply:** 在 kernel 入口放置 `tl.assume(stride_in_d == 1)` 等。

### §2.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树（伪代码）

```python
# 1. 数据准备
input_c = input if input.is_contiguous() else input.contiguous()    # G7
B, H, S, D = input_c.shape
output = torch.empty_like(input_c)

# 2. Broadcast stride 处理（L1.2）
r1_stride = _bc_stride(r1, input_c.shape)
r2_stride = _bc_stride(r2, input_c.shape)

# 3. 编译期常量（T4）
IS_HALF = (rotary_mode == 'half')
IS_FP16 = (dtype == torch.float16)
IS_BF16 = (dtype == torch.bfloat16)
MAX_HALF_D = D // 2          # 作为 tl.constexpr 传入
TILE_S = 16                  # 固定值（L1.5）
assert S % TILE_S == 0

# 4. Grid 计算（G1 动态 num_cores + G4 grid 不超核数）
num_s_tiles = S // TILE_S
num_blocks = B * H * num_s_tiles
def _grid(meta): return (min(num_blocks, VEC_CORE_NUM),)

# 5. Kernel 启动
kernel[_grid](..., num_cores=VEC_CORE_NUM, TILE_S=TILE_S, MAX_HALF_D=MAX_HALF_D, ...)
```

#### L2.2 Kernel 内多核并行骨架（Uniform Grid Splitting）

**核心思想**：将 `num_blocks = B * H * num_s_tiles` 均匀分配到 `num_cores` 个 vector core 上，避免 naive `ceil(num_blocks / num_cores)` 导致的 idle core。

```python
pid = tl.program_id(0)
num_blocks = B * H * num_s_tiles

blocks_per_core = num_blocks // num_cores
remainder = num_blocks % num_cores
block_start = blocks_per_core * pid + tl.minimum(pid, remainder)
block_end = block_start + blocks_per_core + tl.where(pid < remainder, 1, 0)

for block_idx in range(block_start, block_end):
    # 将 block_idx 解码为 (b, h, s_tile)
    tmp = block_idx // num_s_tiles
    s_tile = block_idx - tmp * num_s_tiles
    tmp2 = tmp // H
    h = tmp - tmp2 * H
    b = tmp2
    s_start = s_tile * TILE_S
    # ... load/compute/store ...
```

#### L2.3 2D Tiling 向量加载模式

**模式**：每个 block 处理 `TILE_S` 个连续 S 位置 × `MAX_HALF_D` 个连续 D 位置。

```python
s_offs = tl.arange(0, TILE_S)[:, None]      # [TILE_S, 1]
d_offs = tl.arange(0, MAX_HALF_D)[None, :]  # [1, MAX_HALF_D]

in_base = b * stride_b + h * stride_h + s_start * stride_s
idx1 = in_base + s_offs * stride_s + d_offs * stride_d
idx2 = in_base + s_offs * stride_s + (d_offs + half_d) * stride_d

inp1 = tl.load(input_ptr + idx1)  # 2D vector load
inp2 = tl.load(input_ptr + idx2)
```

#### L2.4 half vs interleave 模式处理

- **half 模式**：将 D 维从中间切分，`[..., :half_d]` 和 `[..., half_d:]` 分别与 r1/r2 的对应半区做旋转乘法。
- **interleave 模式**：将 D 维按奇偶分离，`[..., 0::2]` 和 `[..., 1::2]` 分别做旋转乘法。
- **统一公式**：
  - half: `out1 = x1 * r1_1 - x2 * r2_1`, `out2 = x2 * r1_2 + x1 * r2_2`
  - interleave: `out_even = x1 * r1_e - x2 * r2_e`, `out_odd = x2 * r1_o + x1 * r2_o`

### §2.3 Layer 3: 关键技巧

#### L3.1 从 flat-1D 到 per-position vectorization

**问题**：初始实现使用 `BLOCK_SIZE=1024` 的 flat-1D 循环，每个 thread 处理一个 flat index，通过 `%` 和 `//` 分解坐标，导致标量退化。

**解决**：改为每个 program 处理一个 `(b, h, s)` 位置，D 维用 `tl.arange(0, MAX_HALF_D)` 向量化。关键变化：
- 去掉 `BLOCK_SIZE`，改为 `MAX_HALF_D = D // 2` 作为 `tl.constexpr`
- 去掉 `mask = offsets < total_elements`
- 坐标分解从 per-element 变为 per-position（仅分解 `b, h, s`，`d` 由 `tl.arange` 覆盖）

**可替代方向**: 若 D 不固定，可用 `tl.arange(0, BLOCK_D)` 配合 `for d_tile in range(0, half_d, BLOCK_D)` 做 D 维循环分块。

#### L3.2 Uniform Grid Splitting 消除 idle core（G6 负载均衡的具体实现）

```python
blocks_per_core = num_blocks // num_cores
remainder = num_blocks % num_cores
block_start = blocks_per_core * pid + tl.minimum(pid, remainder)
block_end = block_start + blocks_per_core + tl.where(pid < remainder, 1, 0)
```

前 `remainder` 个 core 各多处理 1 个 block，实现完全均匀分配。

**可替代方向**: 若 block 粒度极不均匀（如不同 block 工作量差异大），可考虑 dynamic work stealing 或按工作量加权分配，但 RotaryMul 中每个 block 工作量相同，uniform splitting 最优。

#### L3.3 2D Tiling (TILE_S=16) 摊平同步开销

**问题**：per-position kernel（每个 program 只处理 1 个 S 位置）在大 S shape 下产生过多 pipeline sync（`hivm.hir.set_flag`, `wait_flag`, `pipe_barrier`），大 S 性能差。

**解决**：每个 program 一次处理 `TILE_S=16` 个连续 S 位置，用 2D `tl.arange` 做 `[TILE_S, MAX_HALF_D]` 的向量化 load/store。

**关键参数选择**：
- `TILE_S = 16` 是经验最优值（非 IR 分析得出）
- `TILE_S = 8` 导致 2D tile 太小（256 elements），vectorization 效果差
- `TILE_S = 32` 导致编译器标量化，性能退化 65x

**可替代方向**: 对于 S 较小（如 S < 16）的 shape，可回退到 TILE_S=1 的 per-position 模式，但 RotaryMul 的 S 通常为 128/256/512/1024+，固定 16 即可。

#### L3.4 避免 Host 侧 dtype 转换和 expand

**问题**：早期实现在 Host 侧将输入 `.to(torch.float32)` 并用 `expand_as()` 处理 broadcast，引入额外内存拷贝和峰值内存占用（16.9MB → 7.44MB）。

**解决**：
- 不在 Host 侧做 dtype 转换，仅在 kernel 内对 fp16/bf16 升精度（T2）
- 不在 Host 侧 `expand` broadcast 张量，而是通过自定义 `_bc_stride` 在 kernel 内用 stride=0 处理 broadcast（L1.2）

### §2.4 RotaryMul 性能基准

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 小 shape [1,1,128,64] | 2.4x | 小 S 高并行度，轻松超越 torch |
| 中 shape [1,8,512,64] | 0.67x | torch aclnn 优化充分，Triton 有 gap |
| 大 shape [1,8,32768,64] | 0.10x | 内存带宽瓶颈，Triton 仍落后 |
| 全量 50 cases | 1.02x | 几何平均刚好达标 |

**关键结论**：
1. RotaryMul 在小 shape 上 Triton 有明显优势（2-3x），但在大 shape / 高并行度场景下，torch aclnn 的 `aclnnRotaryPositionEmbedding` 高度优化，Triton 难以超越。
2. 2D Tiling + Uniform Grid Splitting 是达到目标加速比（0.8x）的关键；去掉任一项都会使几何平均低于目标。
3. 标量退化（flat-1D index 分解）是 Ascend 上最常见的性能陷阱，必须从算法设计阶段避免。

### §2.5 融合搜索基因编码存档（1_RotaryMul, 910b2, 2026-09-01）

> 来源：`triton-optimizer-multi-agent` fusion-search 模式，R2+ 遗传进化（选择/交叉/变异/profiling 驱动新增），10 轮收敛终止。R1 1.0632 → **R4 峰值 1.5261（best）** → R9/R10 1.3816，峰值后未再突破。torch 基线 = `aclnnRotaryPositionEmbedding` msprof Task Duration geomean（50 case）。full 21 基因定义见任务 `opt-round-10/gene_pool.json`。

#### active 基因（收敛后保留）

| 基因 | 类别 | fitness | 编码动作 | 实测结论 |
|------|------|---------|----------|----------|
| G003 | compute | 1.3775 | 旋转直接公式化（math transform）：half 前后半 / interleave 奇偶位公式化，禁显式物化 rotated | R4 胜者核心基因；消除中间 buffer |
| G009 | tiling | 1.3775 | shape-adaptive BLOCK_ROWS：half（D=64）固定 64；interleave 大-S（S>=2048）用 64、否则 32（UB 上限） | R4 胜者；配合 G010 摊薄 scalar 开销 |
| G010 | structural | 1.3775 | multi-head-per-program：每 program 处理跨 (b,h) segment 行块 | R4 胜者；增大 per-program 连续 S 区间工作量 |
| G004 | memory | 1.2792 | 维度合并：展平 [total_rows, D] 2D，每 program 处理多行 | 提高 vector 单元利用率 |
| G014 | compute | 1.2709 | interleave 全寄存器数据流：单次连续 load 完整 D-tile，寄存器内拆奇偶 | 避免重复访存 |
| G008 | config | 1.2167 | 确定性配置（无 @triton.autotune）：固定 BLOCK_ROWS=64, num_stages=1 | Ascend autotuner host-timing 选择近随机，必须显式固定 |
| G012 | compute | 1.1958 | interleave 计算向量化：tl.interleave/tl.join 寄存器内一次构建奇偶输出 | 减少 AIV 逐元素指令数 |
| G002 | tiling | 1.1244 | @triton.autotune 搜 BLOCK_ROWS ∈ {32,64} × num_stages ∈ {1,2} | 配置空间收窄到 {32,64} |
| G001 | memory | 0.9935 | 1D flat 连续 BLOCK_SIZE=1024，手动 grid | 基准基因；多维索引开销避免 |

#### 证伪基因（deprecated/deleted，同窗 A/B 判否）

| 基因 | 类别 | 方向 | 证伪结论 |
|------|------|------|----------|
| G005 | tiling | Block Size Scaling：flat 1D 大 BLOCK_SIZE=8192 | dormant；无提升 |
| G006 | pipeline | multibuffer=True + 核内循环展开 | deprecated；访存延迟隐藏无净收益 |
| G007 | memory | ALIGNED no-mask 快速路径 + 512B 对齐连续访问 | discarded |
| G011 | memory | interleave 连续 tile 加载（MTE2 burst） | deleted；stride-2 消除无收益 |
| G013 | compute | dtype 特化拆分（fp16/bf16/fp32 独立路径） | deleted；统一 upcast 非瓶颈 |
| G015 | pipeline | interleave 大-S AIV 向量发射（num_warps 4/8） | deprecated；AIV 发射非墙 |
| G016 | memory | half 连续全行 store（wider-store） | deleted；case49/48 store-bound 结构性 |
| G017 | pipeline | interleave MTE2-MTE3 overlap（software-pipeline） | deleted；连续三代未打破 case50 AIV wall |
| G018 | compute | interleave FMA 指令收缩（2×2 线性变换合并） | deleted；指令链非墙 |
| G019 | compute | half FMA 指令收缩 | deleted；同 G018 |
| G020 | memory | care_padding=False 屏障消除（MTE2→Vector） | deleted；同步非瓶颈 |
| G021 | memory | interleave store-side MTE3 offload（双 stride-2 store） | deleted；洗牌开销无收益 |

**结构性结论**：interleave/half 大-S case（case49/48 store-bound、case50 AIV wall）为 **store-bound / AIV 指令链 bound**，math-transform（G003）与确定性配置（G008）是核心杠杆；向量发射、内存调度、指令收缩类基因全部证伪，优化已收敛于 noise floor。

### §2.6 本会话实测补充（2026-09-01 triton-op-generator 标准流程，910b2）

#### Layer 1 补充（设计约束）

- **L1.7 bf16 可跳过 fp32 upcast（对 torch_npu reference bit-exact）**
  - 实测 `torch_npu.npu_rotary_mul` 的 **bf16** 路径与 Triton bf16 直接计算 **bit-exact**（matched=1.0 / MERE=0 / maxabs=0，含 mu±100/sigma25 大值输入分布）；fp16 直接计算在大值下 max_error_cap 违例（matched 低至 0.63，MERE 升至 6.4e-3）**必须保留 upcast**。
  - 但跳过 upcast 的实测收益仅 ~1-5%（wall-clock 多 rep），编译器可能已内部升精度——**优先保留 upcast 求稳，跳过仅作为明确测量支持下的备选**。
  - **Why**: 参考实现的低精度路径本身可能与直接低精度计算逐位一致；"一律 fp32 upcast"（T2/L1.4）对 torch_npu 系参考可能过度。

#### Layer 2 补充（算法骨架）

- **网格结构（uniform split vs direct grid）无 kernel 级差异**：wall-clock 多 rep 下两者逐 shape 相等（±5% 噪声内）；benchmark profiler 单 case 显示的大差异（case16 235us→7us、case48 191→325us）为**测量噪声伪影**，非真实差异。任选其一，不必纠结。

#### Layer 3 补充（关键技巧 / 实测结论）

- **2D tile 访存带宽 ≈ 1D flat（拷贝上限 ~640 GB/s @910b2）**：`[BR, D]` 2D `tl.load`/`tl.store` 与 1D flat 拷贝实测同带宽（此前误判 2D tile 慢 10 倍系字节数计算错误）。2D tile 本身不是瓶颈。
- **3-load 合并（6 次半区 load → 3 次全行 load + `tl.trans`/`tl.split` 寄存器拆分）编译不稳定**：仅极小 shape（grid=2）可编译，真实 shape（grid≥2048）MLIRCompilationError；`tl.trans` 在 runtime 循环内触发 BiShengIR 问题。不可用。
- **小 shape 瓶颈 = MTE2 load 延迟主导**（simulator [1,1,128,64] fp16 half：MTE2 ~144K cycles vs VECTOR ~1.1K cycles），大-S 为带宽/store bound——瓶颈随 shape regime 切换，勿用单一结论覆盖全部。
- **bf16 直连与 fp16 upcast 共用 kernel**：`IS_FP16` 编译期分支（仅 fp16 升精度）即可覆盖 fp16/bf16/fp32 三 dtype，避免按 dtype 拆 kernel（G013 证伪方向一致）。

---

### §2.7 2026-09-03 triton-op-generator 标准流程复测（npu_benchmark/level2/1_RotaryMul，910B2）

同语义复测（`torch_npu.npu_rotary_mul`，half×40 / interleave×10，fp16/bf16/fp32，50 case），**50/50 PASS**。

- 实测几何平均加速比（profiler，与 aclnn torch 参考同机对比）：Phase 3 基线 **1.0277x** → opt_iter_0（flat 快速路径）**1.0599x** → opt_iter_1（+ bf16 直连）**1.0735x**。逐轮单调上升、无回退。
- 本轮新增/复验结论：
  1. **contiguous 同 shape 快路径（flat-row 无坐标解码）**：当 r1/r2 与 input 完全同 shape 且连续（本 50 case 全部命中），kernel 用 flat `row×D+col` 连续寻址，去掉 per-block `(b,h,s)` 解码与逐 tensor base/stride 计算，负载均衡仍 uniform grid splitting。较带解码内核约 +3%（1.0277→1.0599）。含 broadcast（shape 不同）时保留解码兜底 kernel（broadcast 维 stride=0），语义不回退。
  2. **bf16 直连（跳过 fp32 upcast）二次复验通过且略有增益**：仅 fp16 走 `IS_FP16` upcast；bf16 输入在 kernel 内直接 bf16 计算 bit-exact 通过，几何平均 1.0599→1.0735（再次印证 L1.7；bf16 大-S case49 亦通过）。
  3. TILE_S=16 固定 + uniform grid splitting + `tl.assume` 仍为稳定配方；小 shape MTE2 延迟主导、大-S store/HBM-bound 的瓶颈分档结论不变，指令级/访存调度类基因（§2.5 G005-G021）继续证伪。

