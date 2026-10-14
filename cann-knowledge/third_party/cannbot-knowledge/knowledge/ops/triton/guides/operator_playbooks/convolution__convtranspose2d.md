---
type: Guide
title: ConvTranspose2d 算子（2D 转置卷积） 实现手册
description: 算子类别：convolutiontranspose2d　典型特征：weight (in_channels, out_channels, KH,
  KW)，输出放大。
tags: [task.implementation, topic.compute.neighborhood.convolution, topic.implementation.memory_management, topic.data_movement.layout_transform, topic.tiling, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# ConvTranspose2d 算子（2D 转置卷积） 实现手册

> 类别约束见 [convolution_constraints.md](../constraints/convolution_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §4 ConvTranspose2d 算子（2D 转置卷积）

**算子类别**：`convolution-transpose-2d`　**典型特征**：weight `(in_channels, out_channels, KH, KW)`，输出放大。
**性能基准**：50/50 pass，**当前最佳 geomean 0.8297x**（target 0.8x **达标**）。Phase 3 基线 0.7783x → Phase 4 连续块划分优化 0.8297x（+6.6%）。历史最佳 ~0.63x（2026-08 前，基线未做连续块划分）。

### §4.1 Layer 1：特化约束（硬规则）

- **等价转换正确性**：ConvTranspose2d ≡ 输入按 stride·dilation 插零 → 按 `pad = K - 1 - P` 补零 → 与**空间翻转**权重做 stride-1 Conv2d。输出 `H_out = (H_in - 1)*S - 2*P + K`，`W_out = (W_in - 1)*S - 2*P + K`。
- **fan_in**（G1，**与标准 Conv 相反**）：`fan_in = out_channels * KH * KW`；weight shape `(in_channels, out_channels, KH, KW)`。
- **权重空间翻转必须在 Triton kernel 内**（按 `KH-1-kh`/`KW-1-kw` 读），**禁止** host 侧 `torch.flip`。
- **host 侧禁止任何 torch 计算/布局操作**：对 weight 禁 `permute/flip/pad/reshape/contiguous`；对输出禁 `permute/reshape`（禁止为 NHWC→NCHW 转换做 host 侧布局操作）。NHWC workspace → NCHW 必须由 Triton kernel 在 device 侧完成。输出通道对齐由 C4d-EXACT-OUTPUT + C4d-STORE-NCHW-DIRECT 处理，**禁止** host 侧切片（见下）。
- **NHWC workspace 连续对齐**：`x_nhwc` shape `(N, H_pad, W_pad, C_in_pad)`，`C_in_pad = ceil(C_in/16)*16`；`H_pad = H_out + KH - 1`，`W_pad = W_out + KW - 1`。
- **weight 转换结果必须缓存**（G9/C5）：首次 forward 调 `weight_prep_kernel` 生成 `(KH*KW*C_in_pad, C_out_pad)` 并缓存；后续复用。**不缓存** geomean 从 ~0.60x 跌至 ~0.24x。
- **C4d-NO-SELF-HELPERS-IN-FORWARD**：`forward()` 内**禁止**调用 `self.xxx(...)` 形式的 helper（`validate_triton_impl.py` 判为 Type3 退化）。workspace cache / weight cache 必须**内联**在 `forward()` 内，通过 `self._workspace_cache` / `self._weight_cache` dict 直接读写。
- **C4d-WEIGHT-INIT-MODULE-LEVEL**：权重/偏置创建必须放在模块级 helper（如 `_create_weight_bias`）。**禁止实例化 `nn.ConvTranspose2d(...)` / `nn.Conv2d(...)` 来"获取权重"**——必须按 `reset_parameters()` 原理手工复刻（同 §1 G1 / 第 48 行）。**禁止**在 `ModelNew.__init__` / `forward()` 内调用 `nn.ConvTranspose2d`。
- **C4d-EXACT-OUTPUT**：输出张量**直接按精确 `out_channels` 分配** `out = torch.empty(N, out_channels, H_out, W_out, dtype=x.dtype, device=x.device)`，conv kernel 内 3D block_ptr store 用 `shape=(C_out, H_out, W_out)` + `boundary_check=(0,1,2)` 掩码最后一个通道 tile 的 padded 列（`C_out_pad` 仍用于 weight 形状/bias pad）。**禁止**分配 padded `out_workspace[N, C_out_pad, ...]` 再 host 侧 `out_workspace[:, :out_channels, ...].contiguous()` 切片——那次全输出张量拷贝是纯开销，大输出时显著拖累延迟。
- **C4d-STORE-NCHW-DIRECT**：conv kernel 输出必须通过 UB 内 `tl.trans(acc, 1, 0)` + 3D `tl.make_block_ptr` 直接写入**精确 `out_channels`** 的输出张量（与 C4d-EXACT-OUTPUT 配套），block_shape `(BLOCK_N, 1, OW_TILE)`，order `(2,1,0)`，`boundary_check=(0,1,2)`。**禁止**先写 NHWC 再调独立 transpose kernel、禁止 host `permute`、禁止 `out.view(...)` + pointer+mask store、禁止 padded workspace + host 切片。
- **C4d-DETERMINISTIC-CONV-CONFIG**：`conv_im2col_kernel` **禁止** `@triton.autotune`，`BLOCK_C` 由 host 侧确定性给出：`BLOCK_C = min(≤C_in_pad 的最大 2 的幂, 64)`（`C_in_pad≥64→64`、`≥32→32`、否则 16），`OW_TILE` 固定 64，二者作为 `tl.constexpr` 传入。`C_in_pad` 恒为 16 的倍数 → 2 的幂 tile 整除无碎片。autotune 对每个 shape key 做 timing 探测会引入运行间不可复现噪声，确定性启发式既稳定又更快。`weight_prep_kernel` 的 autotune 可保留（仅 cache-miss 跑一次，稳态无影响）。

### §4.2 Layer 2：三阶段流水线

1. **weight_prep_kernel**：device 上把 raw weight `(C_in, C_out, KH, KW)` → `(KH*KW*C_in_pad, C_out_pad)`，同时完成空间翻转 + channel padding。加 `@triton.autotune` 覆盖 `(BLOCK_C_IN, BLOCK_C_OUT)`，默认 configs 至少 `{16,32,64}×{16,32}`，`key=["C_in_pad", "C_out", "KH", "KW"]`（此 kernel 仅 cache-miss 跑一次，autotune 探测开销摊销到 warmup、稳态无影响，故保留）。
2. **prep_row_s1_kernel / prep_row_stride_kernel**：NCHW → dilated+zero-padded NHWC workspace。stride=1 与 stride>1 分两个 kernel，均用 `tl.make_block_ptr` 连续 load/store；workspace 通过 `self._workspace_cache` 按 shape 缓存复用（避免每次 forward 重新零填充）。
3. **conv_im2col_kernel**：在 NHWC workspace 上做显式 KH×KW `tl.static_range` 循环，内部 `tl.dot`。**禁止** `@triton.autotune`——`BLOCK_C`/`OW_TILE` 由 host 侧确定性给出（C4d-DETERMINISTIC-CONV-CONFIG），作为 `tl.constexpr` 传入。Host 侧 `BLOCK_N` 按 `out_channels` 选 16/32/64 并 pad `C_out_pad`（weight/bias 维度对齐）；输出直接写入**精确 `out_channels`** 张量（C4d-EXACT-OUTPUT），store 用 §4.3 的 UB transpose + 3D block_ptr + `boundary_check`（C4d-STORE-NCHW-DIRECT）。

### §4.3 Layer 3：关键技巧片段

**weight_prep_kernel 行映射**（raw `(C_in,C_out,KH,KW)` → out `(KH*KW*C_in_pad, C_out_pad)`）：
```python
kh_flip, kw_flip = KH - 1 - kh, KW - 1 - kw
out_row = (kh * KW + kw) * C_in_pad + c_in
# load raw block at (c_in, c_out) with spatial (kh_flip, kw_flip); store out at (out_row, c_out)
```

**UB transpose + NCHW store**（conv kernel 末尾，写入精确 `out_channels` 张量，C4d-STORE-NCHW-DIRECT）：
```python
acc_trans = tl.trans(acc, 1, 0)                 # (BLOCK_N, OW_TILE)
o_ptr = tl.make_block_ptr(
    base=out_ptr + n * stride_o_n,
    shape=(C_out, H_out, W_out),                # C_out = 精确输出通道，非 C_out_pad
    strides=(stride_o_c, stride_o_h, stride_o_w),
    offsets=(c0, oh, ow0),
    block_shape=(BLOCK_N, 1, OW_TILE),
    order=(2, 1, 0),
)
tl.store(
    o_ptr,
    tl.reshape(acc_trans, (BLOCK_N, 1, OW_TILE)).to(out_ptr.dtype.element_ty),
    boundary_check=(0, 1, 2),                    # 掩码最后一个通道 tile 的 padded 列
)
```

### §4.4 性能基准与经验教训

- **当前最佳**：50/50 pass，geomean **0.8297x**（target 0.8x **达标**）。Phase 3 基线 0.7783x（已超旧 ~0.63x 天花板）→ Phase 4 连续块划分 0.8297x（+6.6%）。
- **历史改动链**：① `conv_im2col_kernel` 去 `@triton.autotune` → host 侧确定性 `BLOCK_C = min(pow2≤C_in_pad, 64)`（C4d-DETERMINISTIC-CONV-CONFIG）；② 输出直接分配精确 `out_channels` + store `boundary_check`（C4d-EXACT-OUTPUT）→ 旧基线 ~0.593x 提到 ~0.63x；③ 三阶段 device 流水线（weight_prep / dilate+pad NHWC workspace / fused conv GEMM 直写 NCHW）+ 全缓存（conv_key 缓存 weight/Wmat、data_ptr+shape 缓存 workspace）→ Phase 3 基线 **0.7783x**；④ **连续块划分 + 最内层 ow_blk 解码**（见 §4.3 trick）→ **0.8297x**。
- **Phase 3 基线风险**：若生成时未遵守 C4d-NO-SELF-HELPERS-IN-FORWARD / C4d-STORE-NCHW-DIRECT / C4d-EXACT-OUTPUT / C4d-DETERMINISTIC-CONV-CONFIG，会退化到 pointer+mask + 无 workspace cache + autotune 路线，geomean 仅 **~0.4479x**。
- **Phase 4 已验证无效方向（勿重试）**：每个 program 处理 2 行 oh 并用运行时 `if valid1:` 包绕第二个 `tl.dot` 来复用 `w_blk`（loop-invariant hoisting 思路）→ **MLIRCompilationError 全 case 失败**（见 G13 新增条目）。w_blk 跨 oh 复用本身是对的，但**禁止用运行时谓词包 dot**；若要做跨行复用，须用编译期 `tl.constexpr` 分支或无条件双累加器，且需评估 UB 容量。

### §4.3a Layer 3 补充技巧：连续块划分 + 最内层 ow_blk 解码（Phase 4 验证 +6.6%）

**问题**：交错划分 `for tid in range(pid, total, NUM_CORES)` 把"相同 (n,co_blk,oh) 的相邻输出列 tile"打散到不同核，`w_blk`（仅依赖 kh,kw,ci0,co_blk，与 ow 无关）跨 tile 的 L1 复用被破坏。

**解法**（G2 连续块划分 + 解码顺序）：
```python
# 连续块划分：每个核拥有连续 tile 区间 [start, start+cnt)
tpc_full = total // NUM_CORES
rem = total - tpc_full * NUM_CORES
if pid < rem:
    start = pid * (tpc_full + 1); cnt = tpc_full + 1
else:
    start = rem * (tpc_full + 1) + (pid - rem) * tpc_full; cnt = tpc_full
for off in range(0, cnt):
    tid = start + off
    # 解码顺序：n 外、co_blk、oh、ow_blk 最内层
    n = tid // (CO_BLOCKS * HOUT * OW_BLOCKS); ...
    ow_blk = <最内层余数>
```
- `ow_blk` 作为**最内层**解码维 + 连续块划分 → 一个核连续扫描相同 (n,co_blk,oh) 的相邻 ow_blk → `w_blk` 从 L1 命中复用，减少 Wmat 重读。
- ConvTranspose2d 50-case 实测 geomean **0.7783 → 0.8297**（+6.6%），最差 case 改善显著（case45 0.50→0.61、case34 0.45→0.52）。
- ⚠️ `start/cnt` 计算的 `if pid < rem` 在循环**之前**，不包绕 dot；循环内 `for off in range(0, cnt)` 无运行时谓词包 dot（否则触发 G13 编译崩溃）。
- grid 仍为 `(min(total, NUM_CORES),)`（一维，G2）；`total < NUM_CORES` 时 `tpc_full=0, rem=total`，每核恰好 1 tile。

### §4.5 近期验证记录（2026-08-06）

- **任务**: `10_ConvTranspose2d` 50-case 多 shape 评测，架构 `ascend910B1`，target_speedup=0.8。
- **结果**: 50/50 verify pass，geomean **0.8297x** vs torch，平均实现延迟 0.0108 ms，框架延迟 0.0082 ms，**达到目标**（`target_reached=true`）。
- **Phase 3**（iter_0 一次通过）: 三阶段 device 流水线 + 全缓存 + 确定性 BLOCK_C + 精确 Cout 直写 NCHW，geomean **0.7783x**（已超旧 ~0.63x 天花板）。
- **Phase 4**: opt_iter_0（2-oh w_blk hoist，运行时 if 包 dot）编译失败回退；opt_iter_1（连续块划分 + 最内层 ow_blk 解码，见 §4.3a）成功，geomean **0.8297x**（+6.6%）。
- **瓶颈 case**: 17/3/24/16/34 等为小通道 + 大 K（in/out ≤8、K∈{4,5}），受 Cube C0=16 通道对齐 padding 浪费约束（Cin=1/Cout=1 时最高 ~94% 规约为 padding 零），属该范式天花板；element-wise direct conv 按 §8.1 已验证无效/aicore-timeout 风险，未尝试。

---
