---
type: Guide
title: elementwise 类算子设计约束（逐元素激活）
description: 逐元素非线性激活（GELU / SiLU / SwiGLU）的判别特征与设计硬约束：1D 展平连续访存、动态 grid 钳制核数、fp32
  计算存回原 dtype、compiler hints 逐算子独立判断。
tags: [task.design, topic.compute.elementwise.arithmetic, topic.tiling, topic.precision]
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

# elementwise 类算子设计约束（逐元素激活）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）。

适用范围：**逐元素非线性激活**（GELU / SiLU / SwiGLU），锚定算子 `_gelu_fwd_kernel` (op14) / `_silu_fwd_kernel` (op15) / `_swiglu_fwd_kernel` (op16)。
核心优化哲学：逐元素算子**没有空间局部性收益**，2D tiling 只增加索引开销——一律 **1D 展平 + grid-stride 连续访存**，fp32 计算后存回原 dtype。

## 本类别的算子级实现手册

- [`operators/elementwise__activation.md`](../operator_playbooks/elementwise__activation.md) — Activation（GELU / SiLU / SwiGLU）实现手册：Layer 2 单融合 1D grid-stride 骨架、Layer 3 技巧与性能基准（GELU 0.7969× / SwiGLU 1.3096× / SiLU 1.2669×）。

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

1. 计算为**逐元素**（输出仅依赖同位置输入），且含**非线性激活**：GELU（exact erf / tanh 近似）、SiLU（`x·sigmoid(x)`）、SwiGLU（`silu(a)·b` 双输入）。
2. 参考实现为 `nn.GELU` / `nn.SiLU` / SwiGLU 门控（含 matmul 拆出的激活段）。
3. 输入可 reshape 为 1D、无跨元素依赖 → 命中本文件；否则回退到对应的 matmul / normalization / quantization 类别卡。

> 性能参考：Ascend950PR / `ascend910_9382`，各 30 case（10 shapes × fp16/bf16/fp32）全精度通过：
> GELU 最佳几何平均 **0.7969×**（目标 1.2× **未达**）；SwiGLU **1.3096×**（目标 1.2× **已达**）；SiLU **1.2669×**（目标 1.2× **已达**，CANN kernel-only 复测）。

## §1 通用经验（跨算子，首次生成必须遵守）

- **1D 展平是默认姿态**：输入 `reshape(-1)` + 1D grid + grid-stride；2D tile / 双层 mask / 跨步 tile 循环一律禁止（L1.2）。
- **网格按元素数动态钳制**，小 tensor 不启动满核 grid（L1.3）。
- **fp32 计算、存回原 dtype**（L1.4）；数学公式与 torch 参考严格一致（L1.1）。
- **测量口径**：op 级 `npu.Event` 含 host wrapper 开销会误判小 kernel，**必须以 CANN profiler kernel-only 为准**。
- **BLOCK autotune 512-32768，key=`numel`**；`num_vectorcore` 模块级缓存避免逐次 driver 查询。

## Layer 1 设计约束（硬性规则，GELU / SiLU / SwiGLU 通用）

- **L1.1 必须 数学公式与 torch 参考严格一致**
  - **Why:** 精度测试逐元素与 torch 参考比对，公式偏差直接 fail。
  - **How to apply:**
    - GELU: **⚠️ NPU golden 契约（910B2 实测）**：torch_npu 的 `F.gelu` **忽略 `approximate` 参数**，`'none'` 与 `'tanh'` 输出完全一致（maxdiff=0.0），且均等于 sigmoid 形式 `y = x / (1 + exp(-1.5957691*0.044715*(x/0.044715 + x^3)))`。**必须用此 sigmoid 形式**（`tl.exp`）；用 CPU 语义的 erf 公式 `0.5*x*(1+erf(x*0.7071067811865476))` 时 'none' case max_abs_diff≈4.7e-4 必 fail（MERE 阈值 1.22e-4）。tanh 形式 `0.5*x*(1+tl.math.tanh(t))` 在小值域 ULP 误差 1e-8~1e-7 超 small_value_error（fp32 9.3e-10），也不可用——sigmoid 形式与 golden 逐位一致（小 x maxdiff 0.0）。
    - SiLU: `y = x * sigmoid(x)`，fp32 计算 `x_f32 * tl.sigmoid(x_f32)` 后 cast 回输入 dtype。
    - SwiGLU: `c = silu(a) * b = a * sigmoid(a) * b`。

- **L1.2 必须 逐元素激活 1D 展平 + 连续访存**
  - **Why:** 逐元素算子没有空间局部性收益，2D tiling 只增加索引开销。实测 GELU 2D tiling **0.6665×** vs 1D **0.7969×**；SwiGLU 2D **1.1347×** vs 1D **1.3096×**。
  - **How to apply:** 输入 `reshape(-1)`（非连续先 `contiguous()`），1D grid，每个 program 以 grid-stride 循环处理连续 `BLOCK` 个元素；**禁止** 2D 索引、双层 mask、跨步 tile 循环。

- **L1.3 必须 动态 grid 钳制核数**
  - **Why:** 固定 `grid=(num_vectorcore,)` 在小 tensor 上浪费 launch/scheduling 开销，小 shape 上 baseline 反超。
  - **How to apply:** `grid_size = min(triton.cdiv(numel, BLOCK), num_vectorcore)`；`num_vectorcore` 从 `get_device_properties("npu")["num_vectorcore"]` 模块级缓存读取。

- **L1.4 必须 fp32 计算，存回原始 dtype**
  - **Why:** fp16/bf16 中间精度不足会导致与 torch fp32 参考的累积误差。
  - **How to apply:** 加载后 cast 到 `tl.float32` 再做数学运算，存回前 cast 回输入 dtype。

- **L1.5 ★ 必须 compiler hints 按算子独立判断（不可一刀切）**
  - **Why:** GELU/SwiGLU 用 `multibuffer=True, unit_flag=True` 有收益；SiLU **禁止**使用——实测这些 hints 对小 kernel 的 per-launch 开销反而拖累性能（0.927→0.945× 提升来自**去掉** hints）。
  - **How to apply:** 每个算子独立实测，**不可把 GELU 的 hint 结论直接套到 SiLU/SwiGLU**；单/双输入访存模式不同，hints 对小 kernel 有 per-launch 开销。

## §2 Layer 2 算法骨架要点（1D grid-stride 逐元素激活）

```python
@triton.autotune(configs=[triton.Config({"BLOCK_SIZE": b}) for b in (512,1024,2048,4096,8192,16384,32768)], key=["numel"])
@triton.jit
def _activation_fwd_kernel(x_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0); num_progs = tl.num_programs(0)
    for off_start in range(pid * BLOCK_SIZE, numel, num_progs * BLOCK_SIZE):
        off = off_start + tl.arange(0, BLOCK_SIZE)
        mask = off < numel
        x_val = tl.load(x_ptr + off, mask=mask, other=0.0)
        y_f32 = compute_fn(x_val.to(tl.float32))      # GELU / SiLU / SwiGLU 替换
        tl.store(y_ptr + off, y_f32.to(x_val.dtype), mask=mask)

def activation_impl(x, *args):
    x_flat = x.reshape(-1) if x.is_contiguous() else x.contiguous().reshape(-1)
    grid = (min(triton.cdiv(x_flat.numel(), 512), _NUM_VCORE),)
    _activation_fwd_kernel[grid](x_flat, y, x_flat.numel())   # GELU/SwiGLU 加 multibuffer=True,unit_flag=True；SiLU 不加
```

- BLOCK autotune 范围 512-32768，仅以 `numel` 为 key；`num_vectorcore` 模块级缓存避免逐次 driver 查询。

## §3 证伪方向 / 常见陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 小 tensor 用固定满核 grid | 浪费 launch/scheduling 开销，小 shape 上 baseline 反超 | L1.3 动态钳制核数 |
| 2D tile 双层 mask（含把单输入 2D 模式复制到双输入） | 跨步访存 + 额外 mask 运算；双输入 load 带宽翻倍时更伤 | L1.2 一律 1D 展平 |
| Tail block 特化拆分 | 标量分支开销抵消向量化收益（GELU 0.7923× < 统一 mask 版） | 统一 mask |
| tanh 近似对 exact erf 参考 | 精度不匹配 | L1.1 仅参考用 tanh 时允许 |
| 混用单/双输入算子的 hint 结论、或统一加 multibuffer/unit_flag | 单/双输入访存模式不同；hints 对小 kernel 有 per-launch 开销 | L1.5 逐算子独立实测 |
| 信 op 级 npu.Event 为最终指标 | 含 host wrapper 开销，原始 wrapper 更精简 | 以 CANN kernel-only 为准 |
| 伪造几何平均聚合数 | per-case 全部 <0.5 却报 1.33×（mojo_0729 op15 session 数据作废） | 必须逐 case 校验一致性 |
| GELU 用 erf 公式（910B2） | CPU torch 'none' 语义 ≠ NPU golden；'none' case max_abs_diff 4.7e-4 | 必须用 sigmoid 形式（L1.1 NPU 契约） |
| 大 tensor 统一 BLOCK=8192（910B2） | fp32 大 tensor UB 压力大，8192 反而回退（case35 28M 1.02→0.81） | BLOCK dtype 感知：fp16/bf16≥4M 用 8192，fp32 保持 4096 |
| 对已 multibuffer 的 elementwise 再叠 care_padding=False | multibuffer 已实现 load/compute 重叠，care_padding 无增量（±3% 噪声） | 勿叠加；care_padding 只对无 multibuffer 的 mask load 有效 |
