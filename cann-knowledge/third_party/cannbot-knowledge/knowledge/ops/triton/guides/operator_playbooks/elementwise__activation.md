---
type: Guide
title: Activation 算子（GELU / SiLU / SwiGLU） 实现手册
description: '典型特征: 逐元素非线性激活，1D 展平连续访存，fp32 计算存回原 dtype'
tags: [task.implementation, topic.compute.elementwise.activation, topic.tiling, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/sigmoid.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/exp.md
  role: supporting
created_at: '2026-08-26T08:28:46Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---

# Activation 算子（GELU / SiLU / SwiGLU） 实现手册

> 类别约束见 [../elementwise_constraints.md](../constraints/elementwise_constraints.md)（L1.1~L1.5 硬约束在此），
> 全局约束见 [../global_constraints.md](../constraints/global_constraints.md)。

## §1 Activation 算子（elementwise）

**算子类别**: `elementwise`　**典型特征**: 逐元素非线性激活（GELU / SiLU / SwiGLU），1D 展平连续访存，fp32 计算存回原 dtype。
**性能基准**: GELU 30/30、SwiGLU 30/30、SiLU 30/30 精度通过；几何平均 **GELU 0.7969×**（目标 1.2× 未达）/ **SwiGLU 1.3096×**（已达）/ **SiLU 1.2669×**（已达，kernel-only），Ascend950PR / `ascend910_9382`。

### §1.1 Layer 1 设计约束（算子专属）

- **L1.1** 公式严格对齐 torch：GELU **⚠️ NPU golden 契约（910B2 实测）：torch_npu 的 `F.gelu` 忽略 `approximate`，`'none'`/`'tanh'` 均返回 sigmoid 形式 `x/(1+exp(-1.5957691*0.044715*(x/0.044715+x^3)))`，必须用此形式**（erf 公式 'none' case max_abs_diff 4.7e-4 必 fail；tanh 形式小值域 ULP 超阈值）；SiLU `x_f32 * tl.sigmoid(x_f32)`；SwiGLU `silu(a)*b`。
- **L1.2** 一律 1D 展平 + 连续访存，禁 2D tile / 双层 mask（GELU 2D 0.6665× vs 1D 0.7969×）。
- **L1.3** 动态 grid `min(cdiv(numel, BLOCK), num_vectorcore)`，禁固定满核 grid。
- **L1.4** fp32 计算、存回原 dtype。
- **L1.5 ★** compiler hints 逐算子独立：GELU/SwiGLU 加 `multibuffer=True, unit_flag=True`；**SiLU 不加**。

### §1.2 Layer 2 算法骨架（单融合 1D grid-stride）

```python
import triton, triton.language as tl
_NUM_VCORE = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": b}) for b in (512,1024,2048,4096,8192,16384,32768)],
    key=["numel"],
)
@triton.jit
def _activation_fwd_kernel(x_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0); num_progs = tl.num_programs(axis=0)
    for off_start in range(pid * BLOCK_SIZE, numel, num_progs * BLOCK_SIZE):
        off = off_start + tl.arange(0, BLOCK_SIZE)
        mask = off < numel
        x_val = tl.load(x_ptr + off, mask=mask, other=0.0)
        y_f32 = compute_fn(x_val.to(tl.float32))          # GELU / SiLU / SwiGLU 替换此处
        tl.store(y_ptr + off, y_f32.to(x_val.dtype), mask=mask)

def activation_impl(x, *args):
    ori_shape = x.shape
    x_flat = x.reshape(-1) if x.is_contiguous() else x.contiguous().reshape(-1)
    y = torch.empty_like(x_flat)
    grid = (min(triton.cdiv(x_flat.numel(), 512), _NUM_VCORE),)
    _activation_fwd_kernel[grid](x_flat, y, x_flat.numel())  # GELU/SwiGLU 加 multibuffer=True,unit_flag=True；SiLU 不加
    return y.reshape(*ori_shape)
```

### §1.3 Layer 3 关键技巧（可参考，不可照抄）

- **L3.1 动态 grid + compiler hints（GELU +5.4%）**：固定 `grid=(num_vectorcore,)` 换 `min(cdiv(numel, BLOCK), num_cores)` + `multibuffer=True, unit_flag=True`，GELU geomean 0.7561× → 0.7969×。
- **L3.2 1D 展平 + BLOCK autotune + hints（SwiGLU 1.3096×）**：autotune `BLOCK_SIZE=[512-65536]`（8 档）、key=`numel` + hints；中等 shape（999×9999、1024×10240、5336×3584）达 **1.5-2.1×**。**关键差异**：SwiGLU 双输入使 load 带宽翻倍，1D 连续访存收益比单输入更大（SwiGLU 1D/2D 差 0.17× vs SiLU 差 0.02×）。
- **L3.3 去掉 hints 的 SiLU（1.2669×，kernel-only）**：autotune + 无 hints + 缓存 num_vectorcore。op 级 `npu.Event` 显示 0.945×（比原始差），但 **CANN profiler kernel-only 显示 1.2669×**。**教训**：原始 host wrapper 更精简，op 级 Event 误判，必须以 CANN kernel-only 为准。中型 shape fp16/bf16 达 **2.3-2.49×**。
- **L3.4 Tail block 特化 —— 无效**：满 block（无 mask）与 tail block（有 mask）拆分 → GELU 0.7923× < 统一 mask 版。逐元素激活用统一 mask 更简单更快。
- **L3.5 BLOCK_SIZE 扫描 —— 平坦**：BLOCK=4096/8192/16384 几何平均几乎相同，默认 `BLOCK=4096`。
- **L3.6 fp32 路径接近持平**：fp32 shape 普遍 ≈0.98×，fp16/bf16 大 shape 是主要劣化来源；可考虑 dtype 特化路径或 autotune BLOCK。
- **L3.7 dtype-aware BLOCK（910B2，GELU 0.8995→0.9511 +5.7%）**：fp16/bf16 大 tensor（≥4M 元素）BLOCK=8192（case33 100M bf16 0.853→0.963、case34 134M fp16 0.859→0.962，UB 压力小 8192 流水更优）；**fp32 保持 4096**（case35 28M fp32 用 8192 回退 1.02→0.81）。中小 tensor BLOCK 1024-16384 无差异。实现为 host 侧 dtype+numel 分派，无需 runtime autotune。
- **L3.8 care_padding 无收益（910B2）**：已 multibuffer 的 elementwise 再叠 care_padding=False 无增量（6 shape ±3% 噪声内）——multibuffer 已实现 load/compute 重叠；勿叠加。
- **L3.9 小 tensor 紧 tile BLOCK（910B2，50-case GELU 0.9611→0.9746，+1.4%）**：`numel<4096` 时 BLOCK 用 `next_power_of_2(max(128,numel))` 紧 tile，消除"一个 program 用 BLOCK=4096 掩码空算 3-4× lanes"的小 shape 浪费（此类 case 基线 0.71-0.79×）；大 tensor 仍按 L3.7 dtype 分派（fp16/bf16≥4M→8192，fp32→4096）。不影响 ≥4096 的 shape。
- **L3.10 hints(multibuffer/unit_flag) 保留收益实测（910B2 50-case GELU）**：移除 hints 后 0.9611→0.9199 显著回退（与旧结论相反）——910B2 的 fp32/fp16/bf16 混合 50-case 基准下小 shape 仍依赖 hints 的 load/compute 重叠，**不要盲信"hints 对小 kernel 无收益"的早期结论，按本基准保留**。

### §1.4 性能基准 / 优化记录表（Ascend950PR / `ascend910_9382`）

| 算子 | 方案 | 几何平均 | 精度 | 目标 1.2× |
|------|------|---------|------|-----------|
| GELU | 1D 展平 + 动态 grid + hints（L3.1） | **0.7969×** | 30/30 | 未达 |

> 910B2 补充（50-case, npu_benchmark level1 1_GELU, fp32/fp16/bf16×erf/tanh）：dtype-aware BLOCK + hints 基线 **0.9611×** → 叠加 L3.9 小 tensor 紧 tile **0.9746×**（几何平均, 目标 1.2× 仍不可达；elementwise HBM-bound 物理上限 ~1.0×）。
| SwiGLU | 1D 展平 + BLOCK autotune + hints（L3.2） | **1.3096×** | 30/30 | **已达** |
| SiLU | autotune + 去 hints（kernel-only，L3.3） | **1.2669×** | 30/30 | **已达** |

- 1D vs 2D 差距：GELU 0.7969 / 0.6665；SwiGLU 1.3096 / 1.1347。
- 测量口径：**CANN profiler kernel-only**；op 级 npu.Event 会低估小 kernel（SiLU 0.945× 假象）。

### §1.5 陷阱 / 证伪方向（不要重跑）

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 小 tensor 用固定满核 grid | 浪费 launch/scheduling 开销 | L1.3 动态钳制核数 |
| 2D tile 双层 mask | 跨步访存 + 额外 mask；双输入带宽翻倍时更伤 | L1.2 一律 1D 展平 |
| Tail block 特化拆分 | 分支开销抵消向量化收益 | L3.4 统一 mask |
| tanh 近似对 exact erf 参考 | 精度不匹配 | L1.1 仅参考用 tanh 时允许 |
| 统一加 multibuffer/unit_flag | hints 对小 kernel 有 per-launch 开销 | L1.5 逐算子独立实测 |
| 信 op 级 npu.Event 为最终指标 | 含 host wrapper 开销 | L3.3 以 CANN kernel-only 为准 |
| 伪造几何平均聚合数 | per-case 全部 <0.5 却报 1.33× | 必须逐 case 校验一致性 |
