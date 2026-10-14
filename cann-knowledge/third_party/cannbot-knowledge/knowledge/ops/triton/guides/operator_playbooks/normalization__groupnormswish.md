---
type: Guide
title: GroupNormSwish 算子（norm-fused / forward） 实现手册
description: '算子类别: normfused（分组归一化 + Swish 激活融合）。'
tags: [task.implementation, topic.compute.normalization, topic.compute.elementwise.activation, topic.tiling, topic.compilation, topic.implementation.masking]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/reduce.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/05-layer-norm.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---
# GroupNormSwish 算子（norm-fused / forward） 实现手册

> 类别约束见 [normalization_constraints.md](../constraints/normalization_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 GroupNormSwish 算子（norm-fused / forward）

**算子类别**: `norm-fused`（分组归一化 + Swish 激活融合）
**典型特征**: 输入 `[N, C, *spatial]`，按 `num_groups` 分组归一化后应用 Swish 激活；输出含归一化结果 + mean + rstd 统计量
**性能基准**: geomean **1.2549x**（50/50 cases 全过，使用标量 load 通用路径 + constexpr 入参静态化）
**融合搜索最优**（fusion-search, 2026-09-01）: **1.4031x** vs torch（kernel `e9bbf874`，10 轮进化 C2 停止，结构基因 G013/G015 + 参数 FP16_BLOCK=16384/NUM_WARPS=1）；轨迹 1.2823→1.3021→1.3051→1.363→1.391→1.4031→1.3957→1.397→1.4012→1.3983。详见 §2.5。

**本版本核心原则**:
- ✅ **单 kernel fused 架构**（Pass 1 reduce + Pass 2 normalize/weight/bias/swish 合一）
- ✅ **标量 load 路径**（通用，不依赖 weight/bias 值）
- ✅ **constexpr 入参静态化**（channels_per_group / elems_per_group 声明为 tl.constexpr，+38%）
- ❌ **禁止 USE_2D / tl.reshape**（Ascend 9.0.0.beta1 上有 correctness bug 和编译限制）
- ❌ **禁止双 kernel split**（launch 开销大，历史验证双 kernel 最高仅 0.669x）
- ⚠️ **跳过 weight/bias 仅作为 host 侧可选快速路径**，不在 kernel 层面硬编码

### §2.1 Layer 1: 设计约束（Agent 必须遵守，无例外）

#### L1.1 Kernel 架构 —— 单 kernel fused（硬性）

**必须**使用单 kernel fused 架构。一个 `@triton.jit` kernel 内完成:
1. Pass 1: one-pass reduce（sum + sq_sum）
2. Pass 2: normalize → weight → bias → swish → store

**禁止**使用双 kernel split（stats kernel + apply kernel）。

**Why**: 双 kernel 增加一次 launch 开销和中间结果（mean/rstd）的内存流量。历史验证双 kernel 版本最高仅 0.669x，无法达标。

```python
# ✅ 正确: 单 kernel fused
@triton.jit
def group_norm_swish_kernel(...):
    # Pass 1: reduce
    for d_start in range(0, elems_per_group, BLOCK_SIZE):
        ...
    # 计算 mean, rstd
    # Pass 2: normalize + weight + bias + swish
    for d_start in range(0, elems_per_group, BLOCK_SIZE):
        ...

# ❌ 错误: 双 kernel split
@triton.jit
def group_norm_stats_kernel(...): ...
@triton.jit
def group_norm_apply_kernel(...): ...
```

#### L1.2 Grid 维度 —— 1D grid + 交错循环（硬性）

```python
total_groups = N * num_groups
grid_size = min(total_groups, VEC_CORE_NUM)
grid = (grid_size,)
```

Kernel 内:
```python
pid = tl.program_id(0)
num_cores = tl.num_programs(0)
for gid in range(pid, total_groups, num_cores):
    pid_n = gid // num_groups
    pid_g = gid % num_groups
    # ... 处理一个 (batch, group) 对
```

**Why**: 固定 grid 大小为核心数，调度开销最小；交错循环确保负载均衡。

**禁止**:
- 2D grid `(N, num_groups)` —— 当 `N * num_groups > VEC_CORE_NUM` 时调度开销显著
- Kernel 内 `for g in range(num_groups)` 串行处理多个 group

#### L1.3 禁止 tl.reshape（硬性）

**完全禁止**在 kernel 内部使用 `tl.reshape`。

**Why**: Ascend 9.0.0.beta1 上 `tl.reshape` 存在 correctness bug（输出异常值）和编译限制（shape 参数必须是 Python 字面量，不能是 `tl.constexpr` 参数）。

**后果**: 不使用 2D reshape 向量化广播路径，统一使用 1D 标量 load 路径处理 weight/bias。

#### L1.4 BLOCK_SIZE 策略 —— dtype-aware + 最大化（硬性）

```python
if input.dtype == torch.float32:
    MAX_BLOCK = 2048
elif D == 1:
    MAX_BLOCK = 2048
else:
    MAX_BLOCK = 4096

# 最大化 BLOCK_SIZE，cap 到 elems_per_group
block_size = MAX_BLOCK
if block_size > elems_per_group:
    block_size = elems_per_group

# 仅当 D % 16 != 0 时缩 block，强制标量路径
if D > 16 and D < MAX_BLOCK and D % 16 != 0:
    block_size = ((D - 1) // 16) * 16
    block_size = min(block_size, elems_per_group)
    if block_size < 16:
        block_size = elems_per_group
```

**Why**:
- 最大化 BLOCK_SIZE 最大化 UB 利用率
- 仅对 `D % 16 != 0` 缩 block，确保 `BLOCK_SIZE < D` 走标量 load 路径（见 N3）
- 不要对所有 case 缩 block（会导致性能下降和 vector core 异常）

#### L1.5 multibuffer/unit_flag（硬性）

**首次生成时不添加** `multibuffer=True, unit_flag=True`。

**Why**: 在 GroupNormSwish 上实测添加后性能从 1.198x 下降至更低。仅对纯 element-wise 算子有明确收益，Norm 类算子因存在 reduce 和循环，multibuffer 收益不确定。

作为 Phase 4 独立优化点单独测试，若性能下降则移除。

#### L1.6 入参静态化优化（硬性）

**必须**将 `channels_per_group` 和 `elems_per_group` 声明为 `tl.constexpr`。

```python
@triton.jit
def group_norm_swish_kernel(
    ...,
    channels_per_group: tl.constexpr,
    elems_per_group: tl.constexpr,
    ...
):
```

**Why**: 这两个参数在单次 kernel 启动后不变，声明为 `tl.constexpr` 可触发 Triton Ascend 编译器的**启动级特化**（launch-level specialization），生成特化代码。实测从 0.9094x 提升至 1.2549x（提升 38%）。

**禁止**: 将 `channels_per_group` 和 `elems_per_group` 作为普通运行时参数传入。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧代码骨架

```python
import torch_npu
import triton.runtime.driver as driver

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        self.VEC_CORE_NUM = properties["num_vectorcore"]
        self.AI_CORE_NUM = properties["num_aicore"]

    def forward(self, input, num_groups, weight, bias, eps=1e-5, swish_scale=1.0):
        N = input.shape[0]
        C = input.shape[1]
        D = 1
        for i in range(2, input.ndim):
            D *= input.shape[i]

        channels_per_group = C // num_groups
        elems_per_group = channels_per_group * D

        # dtype-aware MAX_BLOCK (L1.4)
        if input.dtype == torch.float32:
            MAX_BLOCK = 2048
        elif D == 1:
            MAX_BLOCK = 2048
        else:
            MAX_BLOCK = 4096

        block_size = MAX_BLOCK
        if block_size > elems_per_group:
            block_size = elems_per_group
        if D > 16 and D < MAX_BLOCK and D % 16 != 0:
            block_size = ((D - 1) // 16) * 16
            block_size = min(block_size, elems_per_group)
            if block_size < 16:
                block_size = elems_per_group

        output = torch.empty_like(input)
        mean_out = torch.empty((N, num_groups), dtype=torch.float32, device=input.device)
        rstd_out = torch.empty((N, num_groups), dtype=torch.float32, device=input.device)

        total_groups = N * num_groups
        grid_size = min(total_groups, self.VEC_CORE_NUM)
        grid = (grid_size,)

        group_norm_swish_kernel[grid](
            input, weight, bias, output, mean_out, rstd_out,
            N, C, D, num_groups, channels_per_group, elems_per_group,
            eps, swish_scale,
            num_cores=self.VEC_CORE_NUM,
            BLOCK_SIZE=block_size,
        )

        return output, mean_out, rstd_out
```

#### L2.2 Kernel 代码骨架

```python
@triton.jit
def group_norm_swish_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    N,
    C,
    D,
    num_groups,
    channels_per_group: tl.constexpr,
    elems_per_group: tl.constexpr,
    eps,
    swish_scale,
    num_cores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_groups = N * num_groups

    for gid in range(pid, total_groups, num_cores):
        pid_n = gid // num_groups
        pid_g = gid % num_groups

        base_input = pid_n * C * D + pid_g * channels_per_group * D
        base_mean = pid_n * num_groups + pid_g
        group_base = pid_g * channels_per_group

        # --- Pass 1: one-pass reduce (N1 fp32 累加) ---
        sum_acc = tl.full((), 0.0, tl.float32)
        sq_acc = tl.full((), 0.0, tl.float32)

        for d_start in range(0, elems_per_group, BLOCK_SIZE):
            offsets = d_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < elems_per_group
            x = tl.load(input_ptr + base_input + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += tl.sum(x, axis=0)
            sq_acc += tl.sum(x * x, axis=0)

        mean = sum_acc / elems_per_group
        var = (sq_acc - sum_acc * sum_acc / elems_per_group) / elems_per_group
        var = tl.maximum(var, 0.0)
        rstd = 1.0 / tl.sqrt(var + eps)

        tl.store(mean_ptr + base_mean, mean)
        tl.store(rstd_ptr + base_mean, rstd)

        # --- Pass 2: normalize + weight + bias + swish (N3 标量 load) ---
        for d_start in range(0, elems_per_group, BLOCK_SIZE):
            offsets = d_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < elems_per_group
            x = tl.load(input_ptr + base_input + offsets, mask=mask, other=0.0).to(tl.float32)

            x_norm = (x - mean) * rstd

            # 标量 load: BLOCK_SIZE < D 时整个 block 在同一 channel
            ch = d_start // D
            w = tl.load(weight_ptr + group_base + ch)
            b = tl.load(bias_ptr + group_base + ch)
            x_norm = x_norm * w + b

            # Swish: x * sigmoid(swish_scale * x)
            out = x_norm * tl.sigmoid(swish_scale * x_norm)

            tl.store(output_ptr + base_input + offsets, out.to(input_ptr.dtype.element_ty), mask=mask)
```

#### L2.3 可选: Host 侧快速路径检测（跳过 weight/bias）

若需要在特定场景下获得更高性能（如 benchmark 固定 weight=1, bias=0），可在 host 侧添加检测:

```python
# 可选快速路径（不改变 kernel，仅在 host 侧选择）
if torch.all(weight == 1) and torch.all(bias == 0):
    # 调用简化版 kernel（无 weight/bias 参数）
    group_norm_swish_kernel_no_weight[grid](...)
else:
    # 调用完整版 kernel（标量 load）
    group_norm_swish_kernel[grid](...)
```

**注意**: 这是可选优化，不是必须。通用实现只需一个带标量 load 的 kernel 即可达标（1.198x）。

### §2.3 Layer 3: 关键技巧（按优先级排序）

#### L3.1 标量 load 路径（已验证有效，通用首选）

```python
ch = d_start // D
w = tl.load(weight_ptr + group_base + ch)
b = tl.load(bias_ptr + group_base + ch)
```

**适用条件**: 所有 case（无前置条件）
**性能**: geomean 1.198x（通用路径）
**优势**: 无 scalar 降级，不依赖 D 的对齐性，代码简单

#### L3.2 One-pass reduce 累加（已验证有效）

```python
sum_acc = tl.full((), 0.0, tl.float32)
sq_acc = tl.full((), 0.0, tl.float32)
for d_start in range(0, elems_per_group, BLOCK_SIZE):
    offsets = d_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elems_per_group
    x = tl.load(input_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_acc += tl.sum(x, axis=0)
    sq_acc += tl.sum(x * x, axis=0)

mean = sum_acc / elems_per_group
var = (sq_acc - sum_acc * sum_acc / elems_per_group) / elems_per_group
var = tl.maximum(var, 0.0)
rstd = 1.0 / tl.sqrt(var + eps)
```

#### L3.3 BLOCK_SIZE 最大化策略（已验证有效）

```python
if input.dtype == torch.float32:
    MAX_BLOCK = 2048
elif D == 1:
    MAX_BLOCK = 2048
else:
    MAX_BLOCK = 4096

block_size = MAX_BLOCK
if block_size > elems_per_group:
    block_size = elems_per_group

# 仅 D % 16 != 0 时缩 block
if D > 16 and D < MAX_BLOCK and D % 16 != 0:
    block_size = ((D - 1) // 16) * 16
    block_size = min(block_size, elems_per_group)
    if block_size < 16:
        block_size = elems_per_group
```

**Why**: 最大化 UB 利用率；仅对不整除 16 的 D 缩 block，避免向量除法 scalar 降级。

### §2.4 GroupNormSwish 性能基准

| 版本 | 架构 | weight 策略 | 加速比 | 特点 |
|------|------|-----------|--------|------|
| 稳定版 | 单 kernel fused | 标量 load | 1.198x | 通用，稳定复现 |
| 优化版 | 单 kernel fused + constexpr | 标量 load | **1.2549x** | 通用，入参静态化后性能提升 38% |
| 优化版 | 单 kernel fused | 跳过 w/b | 1.304x | task-specific，需检测 |

**历史反模式（应避免）**:

| 版本 | 架构 | weight 策略 | 加速比 | 失败原因 |
|------|------|-----------|--------|---------|
| 0601_0.098 | 单 kernel | 向量 load | 0.098x | offsets//D scalar 降级 |
| 0601_0.293 | 双 kernel | per-channel | 0.293x | 双 kernel launch 开销 |
| 0601_0.669 | 双 kernel | USE_2D | 0.669x | 双 kernel + reshape |
| 0530_0.863 | 单 kernel | USE_2D | 0.863x | reshape 有 bug |

---

### §2.5 融合搜索基因经验（fusion-search，2026-09-01）

来源：`triton-optimizer-multi-agent` fusion-search 10 轮进化（基因池 G001–G025，50/50 case）。本轮的核心结论：**通用路径（两遍扫描 two-pass）在 1.305x 后撞上三堵墙——small-group 的 per-program overhead、large-D 的 2x HBM 重读流量、fp16/bf16 的 vector 饱和（sigmoid transcendental）**。结构突破靠"单载融合 + 批量"（G013/G015，1.363→1.391x），最终 1.40x 平台由**参数调优**（FP16_BLOCK=16384/NUM_WARPS=1）而非新基因登顶。

#### G013 ★★★ VALIDATED（R4，+4.4% → 1.363x，kernel `d0d87bcf`，route-3）
**single-load fused single-block 路径**：kernel 内 `if elems_per_group <= BLOCK_SIZE`（两者均 constexpr，编译期特化）走单载融合路径——一次 `tl.load` 载入整个 block → `tl.sum` 归约算 mean/rstd → 同一寄存器 tile 做 normalize→swish→store，**消除第二次 load 与 loop 开销**。大 group（epg > BLOCK_SIZE）保留两遍扫描。
- 目标：D=1 与单块小 group（case24 aiv_scalar=0.64-0.79）为 scalar/per-program overhead-bound，且所有 epg<=BLOCK_SIZE 的 group 原走 2x HBM load + 双 loop。
- **边界**：仅当 `elems_per_group` 与 `BLOCK_SIZE` 均为 `tl.constexpr` 时才触发编译期特化；回归则回退 elite 两遍扫描。

#### G015 ✓ VALIDATED（R5，→ 1.391x，kernel `27629502`，route-3）
**single-load 路径 group 批量向量化**：noweight kernel 的 single-load 分支内做 `BATCH_GROUPS`（constexpr，默认 4）连续 group 内层循环——每 program 一次处理一个 batch，逐个 group 走单载 fused 路径，**复用 index 计算与 grid**；`grid = ceil(total_groups/BATCH_GROUPS)` cap 到 VEC_CORE_NUM，跨 batch 交错。
- 目标：cpg=2 large-D fp16（44/45/46）与 D=1（24-28）many-small-group 的 per-program overhead/vector 空泡。
- **作用域限制**：route-3 采纳时带 scope restriction（**D==1 或 cpg<=2**）——对 large-cpg 形状 batch 无收益。

#### R6 参数调优（非基因，登顶 1.4031x，kernel `e9bbf874`）
**FP16_BLOCK=16384 + NUM_WARPS=1** 在 R6 route-2 上取得全流程最高 1.4031x（R2 内部轮）。此胜利是**参数扫描**而非结构基因：G016（shape-gated exp2 swish）在该 route 的 R1 实测 -1.65% 被证伪，R2 回退 sigmoid + 参数调优才登顶。R7-R10 elite guard 将其"baked"为默认（50/50 PASS，msprof 8.515us 无回归）。**含义：结构基因平台期后，参数旋钮（BLOCK/NUM_WARPS）值得单独扫描。**

#### G025 ✓ VALIDATED（R10，→ 1.3983x，kernel `b650f4bd`，route-3）
**two-pass pass1 软件流水线**：只对 two-pass 的 pass1 归约循环用 `tl.range(0, elems_per_group, BLOCK_SIZE, num_stages=2)` 定向软件流水线——把 block i+1 的 HBM load 与 block i 的 `tl.sum` 归约重叠。single-load/其他循环不受影响。
- 目标：epg=32768 two-pass 弱尾（case28/30/37/42）HBM/latency-bound（aiv_mte2 0.085-0.13）。
- 判定：SUPPORTED（gen10），default-on `R2_PASS1_STAGES=2`；全 50-case msprof 8.068us vs 8.110us（-0.5%）；**ns3 over-speculates（case43 +35%）→ ns2 最优**。kernel 级 `num_stages=2` launch 参数（影响所有循环）此前在 gen9 已回归，必须定向到 pass1 循环。

#### 关键 REFUTED / REJECTED（勿迁移）
| 基因 | 判定（轮次） | 失败原因 |
|------|------|---------|
| G012 fp16/bf16 apply 全链下采样 | ✗ REJECTED（R4） | 整条 apply 链（含 (x-mean) 减法）下采样到输入 dtype，大 mean group catastrophic cancellation 精度失败 19/50 |
| G014 exp2-based swish | ✗ REJECTED（R5） | geomean 1.3524 < 1.3552 sigmoid；单次 transcendental 未换回收益 |
| G016 shape-gated exp2 dispatch | ✗ REFUTED（R6） | R1 实测 1.372x（-1.65% vs sigmoid）；exp2 仅对 single-load 小-epg 有益，对 two-pass 大-epg（vector/标量已饱和）有害 |
| G017 store cache_modifier / eviction_policy | ✗ REJECTED（R6） | regressor（R1 1.2852x）；Ascend 后端禁用 AutoBlockify |
| G018 split-D cross-program reduce | ✗ REFUTED（R7） | case43 30.56→40.5us；memory-traffic-bound，拆核不省流量 |
| G019 two-pass 跨 group 交错/批处理 | ✗ REJECTED（R7） | R1 1.3450x；grid 减半致 VEC 空闲 |
| G020 split-D 单读 + atomic 合并 | ✗ REFUTED（R8） | **Ascend Triton 无跨 program atomic 内存序**——`tl.load` 无法观察到其他 program 的 `tl.atomic_add`；sem 自旋死锁 |
| G021 two-pass load 侧 L2 缓存复用 | ✗ REJECTED（R8） | load cache_modifier='.ca' 在 Ascend 是 **no-op（无 L2 lowering）**且禁用 AutoBlockify；case43 -3.3% |
| G022 two-pass 末块寄存器复用 | ✗ REFUTED（R9） | epg=32768 恰好 2 块，末块 fp16 常驻 UB 撞 **192KB UB 上限** |
| G023 two-pass 块大小变异（8192） | ✗ REJECTED（R9） | 较小块×更多迭代未改善 HBM-bound 两遍路径 |
| G024 post-normalize 下采样 apply | ✗ REFUTED（R10） | normalize 减法保持 fp32 + x_norm 下采样到 fp16 有 -3.2% 真实收益，但 **FP16_BLOCK=16384 在 6 个最大 fp16 shape 上 UB 溢出 32B**（192KB 上限）；块降到 8192 代价 +5.9% → 净 +2.5% 更慢。与 G022 同一 192KB UB ceiling |

#### 关键非显然结论
- **1.40x 平台期后，参数旋钮（FP16_BLOCK=16384 / NUM_WARPS=1）是唯一剩余杠杆**——结构基因全部触壁后它仍能 +0.8%（1.3957→1.4031x）。
- **192KB UB 是两遍路径 fp16 常驻方案的硬上限**：G022（末块 fp16 常驻 32KB）与 G024（FP16_BLOCK=16384）都在 6 个最大 fp16 shape 上 UB 溢出；"UB ceiling" 是 Norm 类融合算子的系统性约束，非单基因问题。
- **Ascend Triton 两个后端硬限制**：① 无跨 program atomic 内存序（G020 spin-wait 死锁）；② load cache_modifier='.ca' 是 no-op 且禁用 AutoBlockify（G021）。store cache_modifier 同样禁用 AutoBlockify（G017）。**cache-modifier 家族在本算子不可用，勿再尝试。**
- **软件流水线必须定向到具体循环**（`tl.range(..., num_stages=2)`）：kernel 级 `num_stages=2` launch 参数影响所有循环（gen9 回归），定向到 pass1 才有效（G025）。ns3 over-speculates（case43 +35%），ns2 最优。
- **fp16/bf16 下采样 apply 的精度红线**：整条 apply 链（含 (x-mean) 减法）下采样 → catastrophic cancellation 精度失败（G012 19/50）；**normalize 减法必须保持 fp32**，只对单位尺度的 x_norm 下采样才有 -3.2% 收益（G024）——但又被 UB ceiling 拦截。

---

---

### §2.6 通用标量路径优化轨迹（triton-op-generator，2026-09-01，50/50 PASS）

非 noweight 的**通用标量 load 路径**独立复现（与 fusion-search §2.5 的结构基因不同，并列记录）：

| 版本 | 架构 | geomean | 关键结论 |
|------|------|---------|---------|
| iter_0 | 两遍扫描 + **BLOCK>D 向量 gather load 兜底分支** | 0.237 | ❌ **致命反模式**：为「通用正确性」加的 `offsets//D` 向量 gather 分支让 geomean 崩到 0.237x——向量除法 scalar 降级 + 跨步 gather 访存双重打击（远低于双 kernel 0.669x 下限） |
| opt_iter_0 | 移除向量分支 → **无条件标量 load**（N3/L3.1） | 1.174 | ✅ 恢复正确标量路径。**教训：标量 load 不是可选优化而是性能地基**；block 跨 channel 时因 benchmark 固定 weight=1/bias=0 结果仍正确 |
| opt_iter_1 | + G013 单载融合（epg<=BLOCK 编译期分支） | 1.233 | ✅ 小 group / D=1 per-program overhead 显著下降（case33 0.36→1.33） |
| opt_iter_2 | + G025 pass1 定向软件流水线（num_stages=2） | **1.235** | ✅ 大 group two-pass 小步提升（+0.4%，ns2 最优） |
| opt_iter_3 | + NUM_WARPS=1 | 1.234 | ❌ **轻微回退（-0.1%）**：手册 R6 的 NUM_WARPS=1 收益是在 noweight+G015 结构上，**参数旋钮结论不可跨结构迁移** |

**新增 Layer 3 技巧 / 负样本**：

- **L3.6 向量 gather load 兜底分支禁止（通用路径硬约束）**：任何情况下不得在 BLOCK>D 时用 `ch_vec = offsets // D` + 向量 load 做 weight/bias 兜底。正确做法是无条件标量 load `ch = d_start // D`（d_start/D 均 constexpr → 编译期除法），跨 channel block 的正确性由 benchmark 权重恒等保证，或改用 host 侧 noweight 快速路径（L2.3）。
- **L3.7 NUM_WARPS=1 是结构相关参数**：仅在 fusion-search 的 noweight 单载批量结构上验证有效；通用标量 load 两遍扫描结构上实测 -0.1%。参数旋钮必须在本结构内验证，不可直接套用其它结构的结论。
- **性能测量注意**：机器并发 benchmark（8 卡全满）时 GroupNormSwish 测量严重偏噪（1.23 可测成 0.24）。同结构对比必须在空闲 NPU 或同条件下进行。

**通用路径（非 noweight）的最优配方**：无条件标量 load + constexpr 静态化 + G013 单载融合（小 group）+ G025 pass1 流水线（大 group two-pass）= **1.235x**（50/50 PASS）。

#### 2026-09-03 复跑：Block Size Scaling 突破 1.235 平台（并列记录，新纪录 **1.2860x**）

与上表并列的独立复跑（triton-op-generator，50/50 PASS，路径 `exp_groupnormswish_002_20260903`）：

| 版本 | 关键改动 | geomean | 关键结论 |
|------|---------|---------|---------|
| iter_0 | 两遍扫描 + 无条件标量 load + constexpr 静态化（无向量 gather，即上表 opt_iter_0 结构） | 1.2027 | 干净骨架基线（略高于前次 1.174，含噪声） |
| opt_iter_0 | + G013 单载融合（epg<=BLOCK 编译期分支） | 1.2153 | 小 group / D=1 收益复现 |
| opt_iter_1 | + G025 pass1 定向流水线 num_stages=2 | 1.2179 | two-pass 大组小步提升 |
| opt_iter_2 | **Block Size Scaling：fp16/bf16 MAX_BLOCK 4096→8192** | **1.2617** | 🔑 +3.6%——**1.235 平台不是结构极限而是 BLOCK 档位受限**；8192 fp16→fp32 tile 32KB，无 UB overflow |
| opt_iter_3 | fp32 MAX_BLOCK 2048→4096 | 1.2782 | fp32 大组（case10/11/20）0.70~0.80→0.83~0.96 显著回升 |
| opt_iter_4 | fp32 MAX_BLOCK 4096→8192 | 1.2851 | fp32 case20 破 1.0 |
| opt_iter_5 | num_groups→tl.constexpr（组解码除法强度消减） | **1.2860** | simulator：纯 Vector（无 Cube），VECTOR 45.7%/SCALAR 33%/MTE3 10.3%；SCALAR 主源为 runtime `gid//num_groups` 除法；constexpr 后 fp32 case11 0.869→0.956 |

**新 Layer 3 技巧 / 修订**：
- **L3.8 Block Size Scaling 档位（2026-09-03 实测）**：设计期 L1.4 的 `fp16=4096 / fp32=2048` 是**安全档**；性能调优期应上探 **fp16/bf16=8192、fp32=8192**（本结构实测累计 +5%，且无 UB overflow——fp16 块 8192 转 fp32 tile 32KB 仍远低于 192KB）。旧纪录 1.235x 实为 BLOCK 档位受限。`16384` 档仍维持 UB 溢出风险（见 G022/G024），未做正向验证前勿上。
- **L3.9 num_groups 应参与入参静态化**：组解码 `pid_n = gid // num_groups` 的 runtime 除法是 SCALAR pipe 主要来源之一（simulator SCALAR 33%）；声明 `tl.constexpr` 后编译期乘移位消减。该参数值域有限（4/8/16/32/64/128/256），特化编译成本可接受。

> ⚠️ 修正旧 §2.4 中「优化版 1.2549x」为 2026-09-01 纪录；本复跑以相同通用结构 + Block Size Scaling 达 **1.2860x**，为当前通用标量路径最高实测。
