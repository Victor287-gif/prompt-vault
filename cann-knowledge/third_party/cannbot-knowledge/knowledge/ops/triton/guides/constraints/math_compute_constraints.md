---
type: Guide
title: math_compute 类算子设计约束
description: math_compute 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.compute.reduction, topic.compute.ordering, topic.tiling, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# math_compute 类算子设计约束

> ⚠️ **排序原语写 `tl.sort`**，`ext.sort` 在本环境（Triton 3.2.0 + Ascend）会
> `MLIRCompilationError`。详见 [`operators/math_compute__sort.md`](../operator_playbooks/math_compute__sort.md) 顶部说明。

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## 数学计算类算子优化经验

本文档合并了三类数学计算算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复；与张量变换类共用的通用约束见 GC-1）
- **§2 Sum**（sum，多维 reduce-sum）
- **§3 HyenaFftSizePaddingRfft**（transformation-compute，自定义 RFFT）
- **§4 Sort**（sort-topk，硬件排序 + bitonic merge）
- **§5 各算子常见陷阱**

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/math_compute__hyenafftsizepaddingrfft.md`](../operator_playbooks/math_compute__hyenafftsizepaddingrfft.md) — HyenaFftSizePaddingRfft 算子（transformation-compute）
- [`operators/math_compute__sort.md`](../operator_playbooks/math_compute__sort.md) — Sort 算子（sort-topk）
- [`operators/math_compute__sum.md`](../operator_playbooks/math_compute__sum.md) — Sum 算子（sum）

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| Sum | `sum` | 多维 reduce-sum，支持 dim/keepdim/dim=None 全归约 | 按 (ndim, 归一化 dim) 分派 + fp32 累加 + 输出维分块 |
| HyenaFftSizePaddingRfft | `transformation-compute` | 实数输入 → 复数频谱，输出维度 `seqlen+1`，需 zero-pad 到 `fft_size=2*seqlen` | pow2 走 butterfly / 非 pow2 走 Bluestein + 多核 per-stage + 2D tile + 双缓冲 |
| Sort | `sort-topk` | 沿指定维排序，支持 descending/任意 dim，需 pad 到 2 的幂 | 按 L 分段：tl.sort 硬件 / 软件 bitonic / 三段式 sort-merge / 递归 bitonic |

> ⚠️ **关键区分**：三类算子计算模式差异极大，优化哲学不可混用：
> - Sum 关心 **归约维选择与累加精度** 避免 fp16/bf16 累加溢出和末维归约串行
> - HyenaFftSizePaddingRfft 关心 **fft_size 是否为 2 的幂** 决定 butterfly vs Bluestein 路径，以及 **多核 per-stage 拆分** 避免单核串行
> - Sort 关心 **L 分段策略** 匹配 tl.sort UB 上限，以及 **逐元素 min/max 合并** 消除 gather 随机访存

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 6 条约束是三类数学计算算子**共有**且**未在 GC-1）/ G2（pow2 BLOCK）/ G5（int32 索引）/ G7（contiguous）等通用约束此处不再重复；transformer-inference.md T2（fp16/bf16 升 fp32 计算）/ T3（禁止 forward 中 torch 计算）/ T4（constexpr）等也不再重复，各算子章节引用时标注。

### M1 归一化语义维度到末维（permute→contiguous→处理→permute 回）

- **必须**当算子的目标维度（reduce dim / sort dim）非末维时，在 `forward()` 中先 permute 到末维 → contiguous → 处理 → permute 回，**禁止**在 kernel 内部处理多维非末维语义。
- **Why:** kernel 内部降为 2D `(rows, target_dim)` 逻辑能最大化向量化与并行度；多维 grid 在这类算子上无收益且坐标解码 overhead 大。
- **典型应用**：Sort 的 `dim != x.ndim - 1` 路径（L1.2）；Sum 的 `dim` 归一化（host 侧 `if dim < 0: dim += ndim`）。
- 与 G7 的关系：G7 要求整体 contiguous，M1 进一步要求**目标维归一化到末维**再进 kernel。

### M2 非 2 的幂的语义维度必须 pad 到 next_pow2 + 极值填充 + 截断

- **必须**当算法（bitonic sort / butterfly FFT）要求长度为 2 的幂时，host 侧 pad 到 `L = next_power_of_2(n)`，pad 值取**不影响真实元素相对顺序的极值**，处理完成后截断回 `[:, :n]`。
- **pad 值规则**：
  - Sort 升序 → `+inf`，降序 → `-inf`（保证 pad 元素排到末尾）
  - FFT 类 → `0`（zero-pad，不改变频谱幅度）
- **Why:** 算法前提是长度为 2 的幂；非极值填充会污染真实元素。
- **典型应用**：Sort L1.3 pad 到 `L = next_pow2(n)` 上限 32768；HyenaFft zero-pad 到 `fft_size = 2*seqlen`。
- **上限保护**：pad 长度应有上限（如 Sort cap=32768，FFT 受 UB 约束），避免指数膨胀。

### M3 长度参数必须作为 `tl.constexpr` 传入，禁止运行时变量

- **必须**将 `L`（pad 后长度）、`LOG_L`、`fft_size`、`num_stages`、`half_m`、`m`、`K` 等循环边界 / `tl.arange` 长度作为 `tl.constexpr` 传入 kernel。
- **禁止**在 kernel 内用运行时变量作为 `tl.arange` 参数或循环边界。
- **Why:** Triton Ascend 上 fixed-shape vector load 必须是编译期常量长度；constexpr 让编译器展开循环、生成 vector load/store。Sort 的 bitonic stage、FFT 的 Cooley-Tukey stage 都强依赖编译期展开。
- 与 T4 的差异：T4 强调任何进入 `tl.arange` 的变量都要 constexpr；M3 进一步强调**长度类参数**（L/fft_size/num_stages）的特殊性，因为它们同时决定循环边界和 vector load 长度。
- **局部变量例外**：函数体内**禁止**用 `m: tl.constexpr = 1 << s` 这样的局部 constexpr 声明（会报 `_builder argument must be provided`），局部变量去掉注解即可，编译器会自动常量传播。

### M4 UB（Unified Buffer）占用决定 BLOCK_ROWS / tile 大小，必须按算子特征分别自适应

- **必须**为不同计算特征的 kernel 分别提供 UB-aware 的 BLOCK 选择函数，**禁止**共用单一函数。
- **UB 估算规则**：
  - `BLOCK_ROWS * L * dtype_itemsize * live_buffers_count <= UB_TOTAL`
  - 不同算法 live_buffers_count 差异大（tl.sort ~1 buffer，软件 bitonic ~8 buffers，FFT butterfly 2-3 buffers）
- **目标**：`grid = cdiv(n_rows, BLOCK_ROWS) <= num_cores`，且 `n_rows <= num_cores` 时 `BLOCK_ROWS = 1`。
- **典型应用**：Sort 的 `_block_rows_for_extsort`（UB=8192 fp32/16384 fp16）与 `_block_rows_for_L`（50KB / 8 buffers）分离；FFT 的 `BLOCK_SIZE_B=64`（vs 256 更优，减少寄存器压力）。
- **UB overflow 处理**：循环引入后收紧 UB budget（参见 GC-3 相关的 L1.5 模式，UB 从 128KB 收紧到 48KB）。

### M5 按输出元素数（而非输入元素数）分配 grid，且 grid 钳制到 num_cores

- **必须**按**输出元素总数**（非输入元素）分配核数，每个 program 处理一段连续输出；`grid_size = min(total_blocks, num_cores)`。
- **Why:** 数学计算算子的输入输出规模往往差异大（reduce 类输入 >> 输出，FFT 类输出维度 `seqlen+1` ≠ 输入 `seqlen`）；按输入分配会导致空跑或负载不均。
- **典型应用**：Sum 模式 A `grid = (total_out,)`（输出元素数）；Sort `grid = cdiv(n_rows, BLOCK_ROWS)`；FFT `grid = min(total_seq × tiles_per_seq, num_cores)`。
- 与 G4/G6 的关系：G4 要求 grid 不超核数，G6 给出负载均衡公式；M5 强调数学计算类算子**必须按输出元素数**（含归约后的输出点）分配，而非输入。

### M6 数学归一化与算法前置条件必须显式处理（FFT 归一化 / 方向交替 / 模归约）

- **必须**显式处理算法的数学前置条件和后置归一化，禁止依赖隐式行为。
- **典型场景**：
  - **FFT 归一化**：rfft 输出必须除以 `fft_size`，匹配 PyTorch `torch.fft.rfft` 约定（HyenaFft L1.4）。
  - **chirp 角度模归约**：Bluestein `w[n] = exp(-πi·n²/N)` 必须对角度做 `(n*n) % (2*N)` 模归约，否则大 n 时 NPU cos/sin 精度退化（HyenaFft L1.6）。
  - **bitonic 方向交替**：多段 merge 时左右半必须 `DESCENDING` 与 `not DESCENDING` 交替构成 bitonic 序列（Sort L2.5）。
  - **DC/Nyquist 虚部归零**：实输入 RFFT 的 k=0 和 k=seqlen 虚部数学上严格为 0，必须强制归零避免相对误差放大（HyenaFft L3.5）。
- **Why:** 这类数学约束若遗漏会导致 7+ case 精度失败或逐 bit 不匹配，且问题隐蔽（小 shape 正常、大 shape 才暴露）。

---

## §5 常见陷阱与避免方法

### §5.1 Sum 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 末维归约大 shape 慢于 aclnnReduceSum | reduce_last 沿 N 串行步进，大 N（如 16384）下迭代多、访存量大，8192×16384 仅 0.22x | 引入 split-K 两阶段归约；或增大 BLOCK_SIZE_N；勿照搬当前 16×1024 配置（L3.3） |
| 3D/4D 小 tensor 末维归约 launch 开销主导 | `BLOCK_SIZE_A=B=C=1` 导致 grid=输出元素数，小 tensor 上 im 延迟 6x 于 fw（case13 0.031ms vs 0.0049ms） | 输出维分块（A/B>1）让单 program 处理多输出点；小 tensor 回落 atomic 全归约（L3.4） |
| 低精度直接累加丢精度 | fp16/bf16 不经 fp32 累加，大 tensor verify 失败 | 强制 `data.to(tl.float32)` 后累加，store 前 cast 回原 dtype（L1.1/T2） |
| keepdim 误传入 kernel | 在 kernel 内处理 keepdim 导致输出形状错误 | keepdim 只在 host 侧构造 output 形状，kernel 仅按归约后扁平索引 store（L1.4） |
| dim=None 未用 atomic 累加 | 全归约无独立输出元素，多 block 部分和竞争写同一地址出错 | 预分配标量 output，kernel 内 atomic_add（L1.3） |

### §5.2 HyenaFftSizePaddingRfft 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 忘记除以 `fft_size` | PyTorch RFFT 默认归一化 | 两种 kernel 最后都除以 `fft_size`（L1.4/M6） |
| chirp 角度未模归约 | 大 n 时 NPU cos/sin 精度退化 | `n2_mod = (n*n) % (2*N)` 后再算角度（L1.6/M6） |
| 非 2 的幂 seqlen 用直接 DFT | matched_ratio < 0.9 | 必须用 Bluestein 算法（L1.1） |
| `self._get_xxx()` 缓存方法 | AST 验证器误判 PyTorch 退化 | 缓存逻辑内联到 forward()（L1.7） |
| fp64 cos/sin | Triton Ascend 不支持 | 全程 fp32 + 模归约保精度（L3.7 禁止项） |
| Twiddle LUT | global memory load 慢于硬件 trig，慢 6x | inline `tl.math.cos/sin`（L3.7 禁止项） |
| 向量化 in-place bit-reversal | 数据冒险，5/49 case 精度失败 | out-of-place gather 或保持标量循环（L3.3/L3.7 禁止项） |
| `N.to(tl.float32)`（N 为 constexpr） | constexpr 无 `.to` 方法 | 直接用 N 或创建运行时变量（L3.7 禁止项） |
| 函数体内 `m: tl.constexpr = 1<<s` | 局部变量不能用 tl.constexpr 注解 | 去掉注解，编译器自动常量传播（L3.7 禁止项/M3） |
| 合并 (k,j) tile 的整数除法分解 | NPU 整数除法昂贵 + 每次重算 twiddle，慢 9x | 保持 `for j_start` 外层 + `for k` 内层的双层循环结构（L3.7 禁止项） |
| Pass-merge normalize 到最后 stage store | Bluestein IFFT `num_stages >= 5`，`apply_norm` 分支引入额外乘法 + 寄存器压力，实测负收益 | 保持独立 normalize pass（L3.7 禁止项） |

### §5.3 Sort 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| `tl.sort` 单次排序长度超 UB 上限 | L > 4096（fp32）或 L > 8192（fp16）时 `tl.sort` UB overflow | 按分段策略拆分；L=16384 fp16 用三段式 sort-merge（L1.1/L2.3） |
| 升序大 L 直接用 `tl.sort(descending=False)` | `tl.sort` 8192 长度 asc 路径不稳定 | 用取负 trick `-tl.sort(-X, desc=True)`（L1.7/L3.2） |
| pad 值方向错误 | asc 用 `-inf` pad 会让 pad 元素排到开头 | asc 用 `+inf`，desc 用 `-inf`（L1.3/M2） |
| 两半方向相同就做 min/max 合并 | min/max 拆分要求两半方向相反构成 bitonic；同向合并不会得到有序结果 | Stage 1 两半必须 `DESCENDING` 与 `not DESCENDING` 交替（L2.3/L2.5/M6） |
| 硬编码 `num_cores` | 不同型号 NPU 核数不同 | 动态读取 `num_vectorcore`，兜底 48（L1.5/G1） |
| BLOCK_ROWS 共用单一函数 | tl.sort 与软件 bitonic 的 UB 占用差异大，共用会溢出或浪费并行度 | 分别提供 `_block_rows_for_extsort` / `_block_rows_for_L`（L1.6/L3.7/M4） |
| 非末维排序直接进 kernel | kernel 只处理 2D `(n_rows, n)`，非末维会维度错乱 | forward 中 permute 到末维，排序后 permute 回（L1.2/M1） |
| kernel 内使用 `continue`/`break` | 该后端对非规整控制流支持有限 | 用 `tl.where` 条件赋值替代（L1.8） |
| cross-tile gather merge 极慢 | gather 随机访存 | 用逐元素 min/max 合并替代（L3.3） |
| 为限制 grid 在核数内而对 `tl.sort` 加 program 内循环 | 编译器 multi-buffer 会把循环内 `tl.sort` 的临时 buffer 保留多份，导致 UB overflow（实测 fp32 L=4096 要求 208KB > 192KB） | 对 `tl.sort` 路径直接按 `grid = cdiv(n_rows, BLOCK_ROWS)` 启动，不强制限制在 `num_cores`；或在循环体外手动释放 buffer（若后端支持） |
| `tl.sort` 的 import 路径错误 | `triton.language.extra.cann` 在部分环境（如 Triton 3.2.0）不存在 | 使用 `triton.language.sort` 直接调用 |
---

## §6 GELU / elementwise 激活算子经验（2026-08-13, ascend910b1, 50 cases 实测）

### Layer 1（设计约束，硬性）

- **G1-GELU 公式对齐**：GELU 标杆（torch.nn.functional.gelu 在 NPU 上）对 `approximate='none'` 与 `'tanh'` **均按同一 sigmoid 等价公式**计算：
  `y = x / (1 + e^(-1.5957691 * 0.044715 * (x/0.044715 + x^3)))`
  CANN 无精确 erf 路径，**必须**用此公式对齐，单 kernel 即可覆盖两模式（无需 constexpr 特化分支）。
- **G2-除法不可替换**：`x/0.044715` 常量除法在 BishengIR 生成 `vdiv + temp_buffer` 慢路径，但**禁止**改乘倒数常量——fp32 小值域阈值（small_value_error=9.31e-10）下 1ulp 舍入差击穿 matched_ratio≥0.9（实测 38/50 失败）。标杆数值基准为除法形式，除法是精度必需。
- **G3-BLOCK 上限**：大 shape（16K~134M 元素）带宽受限时 BLOCK 需扩至 UB 安全上限；fp32 下 BLOCK=16384（x 与 x_f32 双 buffer 128KB < 192KB）为安全上限，禁止超过（GC-8）。

### Layer 2（算法骨架）

- 单 kernel 1D flat elementwise：grid = min(cdiv(n, BLOCK), num_cores)（动态读取 num_vectorcore），
  grid 用 meta-lambda 与 autotune 选中 BLOCK 匹配；核内连续块划分（blocks_per_core + remainder 负载均衡，
  禁止交织 grid-stride——checklist Task 6）；循环边界用 tl.num_programs(0) 自算。
- `@triton.autotune(BLOCK_SIZE ∈ {1024, 2048, 4096, 8192, 16384}, key=['n_elements'])`：
  小 shape 自动选小块（launch-bound），大 shape 自动选大块（带宽受限），全 shape 自适应。

### Layer 3（关键技巧）

```python
@triton.autotune(configs=[triton.Config({"BLOCK_SIZE": b}) for b in (1024, 2048, 4096, 8192, 16384)],
                 key=["n_elements"])
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    grid_size = tl.num_programs(0).to(tl.int32)
    total_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    bpc = total_blocks // grid_size
    rem = total_blocks - bpc * grid_size
    if pid < rem:
        my_blocks, start = bpc + 1, pid * (bpc + 1)
    else:
        start = rem * (bpc + 1) + (pid - rem) * bpc
        my_blocks = bpc
    for i in range(my_blocks):
        offs = (start + i) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0)
        x_f32 = x.to(tl.float32)
        x3 = x_f32 * x_f32 * x_f32
        arg = x_f32 / 0.044715 + x3          # 除法保留（G2 精度守卫）
        y = (x_f32 / (1.0 + tl.exp(-1.5957691 * 0.044715 * arg))).to(x.dtype)
        tl.store(y_ptr + offs, y, mask=offs < n_elements)
# host: grid = lambda meta: (min(triton.cdiv(n, meta["BLOCK_SIZE"]), num_cores),)
```
- 实测性能：50 cases 几何平均 speedup 1.1276（baseline 0.8814，+27.9%）；小 shape（≤4K 元素）1.4-1.9x（launch-bound），大 shape 带宽极限 ~0.8-1.0x。
- 可替代方向：fp16/bf16 下可进一步试 BLOCK=32768（UB 192KB 边界，需实测 UB overflow 风险）。
