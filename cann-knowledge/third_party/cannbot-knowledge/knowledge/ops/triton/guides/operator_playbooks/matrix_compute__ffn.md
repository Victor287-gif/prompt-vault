---
type: Guide
title: FFN（Feed-Forward Network）两级 GEMM 算子实现手册
description: '算子类别: matrix_compute。out = act(x@W1+b1)@W2+b2，含标准 fp16/bf16、门控激活(geglu/swiglu/reglu)、int8
  量化、MoE expert 四类路径。'
tags: [task.implementation, topic.compute.composite, topic.compute.elementwise.activation, topic.compute.quantization, topic.mixture_of_experts, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-31T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# FFN（Feed-Forward Network）两级 GEMM 算子实现手册

> 类别约束见 [matrix_compute_constraints.md](../constraints/matrix_compute_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §8 FFN 算子（feed-forward-network）

**算子类别**: `matrix_compute`（两级 GEMM + 非线性激活）
**典型特征**: `out[M,N2] = act(x[M,K1]@W1[K1,N1]+b1)@W2[K2,N2]+b2`。53 case 覆盖：
- 标准路径（fp16/bf16，inner_precise=1）：K2 = N1，h 以 x.dtype 物化
- 门控激活（geglu/swiglu/reglu）：K2 = N1//2，h = act(gate)*up
- int8 量化（scale/offset/deq_scale）：int8 dot + int32 acc + 两级 dequant/requant
- MoE 专家（expert_tokens Python list）：W1/W2 3D [E,K,N]，per-expert 复用标准两段
**性能基准**: 53/53 verify pass，几何平均加速比 **1.2847x** vs torch（2026-08-31 首次运行，opt_iter_1）；**1.1884x**（2026-09-01 复跑，opt_iter_5，npu_ffn framework 更快故绝对数值更低，同环境相对基线 +16.7%）。两次均为结构性内存 bound（simulator：MTE1+MTE2 47-62%，Cube 9-11%；MMAD 9-22% + Cube WAIT_FLAG 空等）
**融合搜索最优（fusion-search, 2026-09-01）**: **1.3509x** vs torch（kernel `f419e763`，10 轮进化 C2 停止，基因 G026）；轨迹 0.5042→0.769→0.9471→1.1193→1.1942→1.2584→1.2584→1.2584→1.2845→1.3509。详见 §8.5。

> **硬件绑定**: Ascend 910B2（24 cube / 48 vector）/ CANN 9.1.0。L0C 128KB、UB 192KB。

### §8.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 ★★★ npu_ffn 门控拆分方向 = up 在前
- **必须** gate = W1 后半列 `[K2:2K2]`，up = W1 前半列 `[0:K2]`，`h = act(gate)*up`。
- 与 docstring `gate, up = x.chunk(2)`（gate 在前）**相反**。纯 torch 链 gate-first 对 geglu 有 18.3% 违规，up-first 后 0%。
- reglu 两种方向都过（relu 语义下差异被容差吸收），但为一致统一用 up-first。
- **Why**: npu_ffn 实证口径；bias 对应也要交换（gate 用 bias[K2+n]，up 用 bias[n]）。

#### L1.2 ★★★ int8 量化数值链（与 docstring 完全不同，必须按此链）
- dequant：`out1_fp = acc.to(f32) * deq1.to(f32)`（**fp32 乘积**，非 bf16 乘积）
- deq1 重映射：`c_idx = (n//DEQ_BLOCK)*DEQ_BLOCK + (n%256)`，`DEQ_BLOCK = 768 if N1==8192 else 256`
- activation 在 fp32：fastgelu 常数 **1.702**（`x/(1+exp(-1.702x))`）、silu=`x/(1+exp(-x))`
- requant 全 **fp16 算术**：`t = act.to(fp16)*scale.to(fp16) + offset.to(fp16)` → `tl.extra.cann.libdevice.nearbyint(t.to(f32))`（RNE）→ clamp(-128,127) → int8
- GEMM2 输出：`out_fp = acc.to(f32) * deq2.to(f32)` → cast bf16
- **Why**: 按此链 case 25/26 与 npu_ffn **完全 bit-exact（max_abs=0）**；任何"等价"改写（bf16 乘积、round-half-away、fp32 requant）都引入 1-2 ulp 差异。

#### L1.3 两段 GEMM 结构（禁止单 kernel 融合）
- activation 非线性阻止两级 GEMM 按 K 融合；隐藏维最大 16384 无法驻留 UB(192KB)，必须物化 h 到 HBM。
- hidden kernel（GEMM1 + 激活）+ output kernel（GEMM2）两段；h 以 x.dtype(fp16/bf16) 物化（inner_precise=1）。

#### L1.4 910B2 基线 tiling（激进改动实测回退）
- 标准非门控：**BM=64, BN=256, BK=128**（acc[64,256]fp32=64KB，UB 160KB）
- 门控：BM=32, BN=256, BK=64（双 acc [32,256]×2=64KB）
- int8 量化：BM=64, BN=128, BK=256（epilogue dequant+activation+requant 临时缓冲多，BN=256 触发 UB overflow 2105344 bits）
- ⛔ 已证伪：BM=128/BN=128/BK=64（小 M 反噬，1.28→1.08）；K_ALIGNED mask 消除（编译器已处理，broadcast_to 反而 -0.6%）。
- ✅ **性能升级（2026-09-01 实测）**：标准非门控 M≥128 时可用 **BM=128, BN=128, BK=256**（acc[128,128]fp32=64KB + a[128,256]fp16=64KB + b[256,128]fp16=64KB = 恰 192KB 可编译）：带 epilogue 实测 M256 +8%、M128 +3.5%、M64 持平，全量 geomean +9.7%。配合 **BM 按 M 分派**（M<128→BM64）避免小 M 半块浪费。int8 quant **禁止** BM128（真实 dequant+activation+requant epilogue UB overflow，简化探针会漏判）。

#### L1.5 scale/offset 维度独立判定
- scale 可 [N] 或 [1]，offset 同理；**必须**分别判定（SCALE_PER_COL / OFFSET_PER_COL）。
- ⛔ per-column scale [8192] + scalar offset [1] 时若 offset 按列读 → 越界读垃圾 → 99.4% 违规。

#### L1.6 归约维 K-mask 必加（BK 升级后）
- 任何 load 都要带 `k_mask = offs_k < K`，尤其 BK 不是 K 因子时。
- ⛔ BK=128 时因全部 K 恰为 128 倍数而"侥幸正确"；BK 升 256 后 K=128 的 shape 立即越界读到下一行垃圾 → AccuracyError（matched_ratio=0）。
- 这是 GC-7（归约维 padding 显式排除）的硬要求，不是可选项；aligned 时全 true mask 开销 ~7% 仍划算。

### §8.2 Layer 2: 算法骨架（首次生成就要写对）

1. **host 路径分派**（GC-14）：MoE(expert_tokens) → per-expert 标准两段；quant(scale/deq_scale1) → int8 两段；标准 → fp16/bf16 两段。
2. **hidden kernel**：1D grid + 核内交织；k 循环 native fp16/bf16/int8 dot + fp32/int32 acc；epilogue = matmul 输出按 op dtype 舍入 → bias → activation(fp32) → store h。
3. **output kernel**：h@W2，acc 按 op dtype 舍入 → bias → cast out dtype。
4. **MoE**：per-expert 循环（Python list cumsum offsets），x/h/out 行切片 view + w1[e]/w2[e]；bias 3D [E,N] 时 bias[e]。
5. **inner_precise=1**：h 以 x.dtype 物化；=0 时 fp32（本任务全部为 1）。

### §8.3 Layer 3: 关键技巧

- **load 重排序**（w1 先于 x 发射）微幅提升 +0.22%（掩盖 L1→L0 2D transpose 慢搬运）；全 kernel 应用后全量 ~+0.9%（2026-09-01 实测）。
- **shape 分派（host）**：BM 按 M 分派（M≥128→128 否则 64）；MoE 按 max_nt 分派（≥128→128 否则 64）。合计再 +5~9%。
- **int8 bias 顺序**：bias1/bias2 先加到 int32 acc 再 dequant（`(acc+bias)*deq1`），写成 `acc*deq1+bias` 会有 3~7% 系统误差。
- **RNE 路径**：`tl.extra.cann.libdevice.nearbyint`（kernel 内可用）；`from triton.language.extra import libdevice` 的 `libdevice.nearbyint` 返回 None 不可用；`tl.round` 不存在。
- **round-half-away 负值公式**：`where(x>=0, floor(x+0.5), -floor(0.5-x))`；`floor(x-0.5)` 对负数错。
- **debug 方法**：verify dtype-aware 容差（fp16 atol=9e-2/rtol=2^-10；bf16 atol=1e-1/rtol=2^-7）；中间 h 对比（bit-exact 排除 hidden，聚焦 output）；对 npu_ffn 黑盒做拆分方向/数值链假设测试。

### §8.4 常见陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 门控 gate-first | docstring 误导 | npu_ffn 实证 up-first (L1.1) |
| int8 dequant bf16 乘积 | docstring 说 bf16 | fp32 乘积 + deq 重映射 (L1.2) |
| requant fp32 / round-half-away | 直觉 | fp16 算术 + nearbyint RNE (L1.2) |
| fastgelu 用 1.7 | docstring | 量化路径用 1.702 (L1.2) |
| quant epilogue UB overflow | BN=256 临时缓冲多 | BN=128 (L1.4) |
| output_dtype 当 int 比较 | get_input_groups 已映射 torch dtype | 直接用 output_dtype (L1.x) |
| offset 越界读 | scale per-column + offset scalar | 独立判定维度 (L1.5) |

### §8.5 融合搜索基因经验（fusion-search，2026-09-01）

来源：`triton-optimizer-multi-agent` fusion-search 10 轮进化（基因池 G001–G026）。两段 GEMM 融合后（F1：hidden+output）核心瓶颈为 GEMM1/GEMM2 的 **HBM 重读流量（traffic）** 与 **激活 epilogue 的 UB 占用**，tiling 选择从"固定 shape 经验"进化为"**逐 case 流量最小化 + 硬约束防护**"。

#### G026 ★★★ VALIDATED（R10，+5.2% → 1.3509x，kernel `f419e763`，route-3）
**GEMM1 traffic-optimal adaptive tile + grid-floor guard**：`_pick_tiles_gemm1(M,N,K,fp32_path,chunked,quant,small_m,x_bytes,w1_bytes,traffic_opt=True)` 逐 case 求最小 `T = x_bytes×ceil(N1/BN) + W1_bytes×ceil(M/BM)`，带多层防护：
- **UB 硬约束**：`BM×BN <= 16384`（GEMM1 fp32-acc per-buffer ≤64KB，**不是** 192KB 经验公式）；GEMM1 有激活 epilogue，UB 预算比 GEMM2 紧。
- **grid-floor guard**：非默认候选要求 `grid >= 16`（不劣于默认 tile 的并行度）；默认 tile 恒允许。
- **fp32-mid guard**：`fp32_path && M>=512 && N<8192` → 保持默认 tile（防中尺寸 fp32 反噬）。
- **G024c BN=256 排除**：regression-profile 排除特定 case 的 BN=256。
- **fallback**：默认 tile `(bm_d,bn_d,bk_d,ns_d,nw_d)`，tie-break 默认 tile > 更大 BK > 更大 ns。
- 有效组合：native mid 宽 N `(128,128,128,ns3,nw4)`；native mid M≥512 `(128,128,64,ns2,nw4)` 默认；fp32 mid M<512 `(128,128,128,ns3,nw4)`；fp32 M≥512 N<8192 默认 `(64,128,64,ns2,nw2)`。
- 收益分布：case7 +35.5%、case8 +30.2%、case6 +29.4%、case21 +18.8%、case22 +18.0%（宽 N / cube-mte2 共主导 case）；恢复回归 case 47/50/51/11/20/31/42。53/53 bit-consistent（纯 tiling 无数值风险）。

#### G024 ✓ VALIDATED（R9，→ 1.2845x，GEMM2 侧）
**GEMM2 traffic-optimal tile**：`_pick_tiles_gemm2` 最小化 `T = mid_bytes×ceil(N2/BN) + W2_bytes×ceil(M/BM)`，UB 约束 `4·BM·BN + ns·2·(BM·BK+BK·BN) <= 192KB`（GEMM2 无激活 epilogue，预算宽裕）。mid 主导（M>>N2，重读 6x）选大 BN（128→256，mid 重读减半）；W2 主导（N2 大如 16384）选大 BM。grid-floor 16、BN=256 profile 排除、nw=BN≥256→4 否则 2。前提：GEMM2 cube 66-99% + mte2 0.82-0.96 双饱和（W2-stream 削减穷尽后唯一杠杆），须整组 A/B 对照（禁单参数 tweak）。

#### G025 ✗ REJECTED（R10，-2.3% → 1.2495x，route-2）—— 勿迁移
**把 G024 traffic-tile 族从 GEMM2 移植到 GEMM1 失败**：GEMM1 权重 **L2-resident**，HBM 重读流量模型失效；最坏 case 为 cube/mte2-bound（cube_util 88-99%），流量削减不转 wall-clock 收益。且 GEMM1 激活 epilogue 改变 UB 布局（192KB 预算 vs GEMM2 256KB），relu 下被迫排除 (BM=128,BN=256) 压缩可用 tile 空间。
- **可移植性规则**：G024（no-activation epilogue + 256KB UB + DRAM-resident 权重）**不可**移植到 G026 所在上下文（activation epilogue + 192KB UB + L2-resident 权重）；traffic-tile 家族必须逐 GEMM 独立重求。

#### 关键非显然结论
- **GEMM1 UB 上限是 `BM×BN<=16384`（per-buffer ≤64KB），不是 192KB 公式**——超过则静默丢 UB headroom，且常规 profile 看不出（5b gate 通过但性能停滞）。
- **grid-floor guard**（grid≥16）是 traffic-tile 的必备护栏：否则小 M 大 BN 的"流量最优"在 grid-starved 时反而劣于默认。
- traffic-tile 只在对应 GEMM **DRAM 带宽饱和 + 对应 operand 主导**时生效；先确认 bottleneck 是 mte2 而非 cube/scalar。
### §8.6 2026-09-04 op-generator 复跑补充（Level4 2_FFN，53 case，npu_ffn framework）

- **激活公式探针锁定（单位权重黑盒 W1=W2=I，新增实证，之前 runbook 只对 quant 锁 1.702）**：
  - fastgelu = x·σ(1.702x)（标准 fp16/bf16 路径同用 1.702，非 1.7）
  - gelu = tanh-近似 σ 形式 x·σ(1.5957691x + 0.0713548x³)（erf 偏离 0.0014 > tanh 0.000955，Triton 用 σ/exp 形式避免 tanh）
  - silu/relu 标准。**若后续 FFN/gated 激活精度不过，先按此链核对，勿凭 docstring 用 1.7/erf。**
- **bf16 inner_precise=0 语义**：Level4 JSON 中 bf16 路径全部 inner_precise=0（runbook 旧述"本任务全部为 1"与实际不符）。实测 h 仍以 x.dtype(bf16) 物化即可 53/53 过——inner_precise 只对应累加精度；两层 GEMM acc 均 fp32、act 在 fp32 acc 上算、h/out 存回 bf16。
- **复现收益汇总**（Phase3 基线 1.2622 → opt 1.4732，+16.7%，53/53）：
  1. std hidden GEMM1 按 M 分派（M≥128→BM128/BN128/BK256，否则 BM64） +6.5%
  2. GEMM2 traffic-optimal tile（G024 流量模型，std/gated/MoE output） +7.7%
  3. MoE per-expert hidden M 分派 +0.2%
  4. load 重排（w1 先于 x/h 发射） +1.5%
- **AST wrapper 结构约束**（validate_triton_impl 3.2）：forward() 内 for/while 与 self.* 均被 flag；MoE per-expert 循环与多路径分派必须放模块级 wrapper（直接 launch kernel），forward 只做直线调度。
- simulator 再次确认结构性内存 bound（CUBE 10-12% + WAIT_FLAG；MTE1+MTE2 ~60%），非 MMAD 极限，与 §8 历史一致。
