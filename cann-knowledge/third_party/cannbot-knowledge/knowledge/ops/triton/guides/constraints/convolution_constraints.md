---
type: Guide
title: convolution 类算子设计约束
description: convolution 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.compute.neighborhood.convolution, topic.data_movement.layout_transform, topic.data_movement.memory_access, topic.compute.composite, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---

# convolution 类算子设计约束

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## 卷积类算子优化经验

本文档合并卷积类算子的优化经验，按以下结构组织：

- **§1 通用经验**：跨算子共有的工程约束（已提取，各算子章节不再重复）+ 通用算法骨架
- **§2 ConvStandard1d**（1D 标准卷积）
- **§3 ConvDepthwise2d**（2D depthwise）
- **§4 ConvTranspose2d**（2D 转置卷积）
- **§5 ConvStandard3d**（3D 标准 / grouped，`kernel_size=(k,k,1)`）
- **§6 ConvStandard2d**（2D 标准 + grouped + depthwise-like）
- **§7 常见陷阱**（按算子分小节）
- **§8 已验证无效方向**

> ⚠️ **关键区分**：卷积类的核心优化哲学是 **"按 stride/groups/kernel 特化多路径分派 + K-tap→1×1 分解 + reduction 维对齐 Cube C0=16"**。生成时先读 §1 通用经验（硬规则），再读本算子章节；**禁止混用**非卷积类别的经验。

> ⚠️ 本文件只记录**设计约束、算法骨架与关键技巧片段**，不保留可直接复制的完整 kernel；生成时应根据约束重新设计变量名与实现结构。

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/convolution__convdepthwise2d.md`](../operator_playbooks/convolution__convdepthwise2d.md) — ConvDepthwise2d 算子（2D depthwise）
- [`operators/convolution__convstandard1d.md`](../operator_playbooks/convolution__convstandard1d.md) — ConvStandard1d 算子（1D 标准卷积）
- [`operators/convolution__convstandard3d.md`](../operator_playbooks/convolution__convstandard3d.md) — ConvStandard3d 算子（3D 标准 / grouped，源 benchmark 实际 `kernel_size=(k,k,1)`）
- [`operators/convolution__convtranspose2d.md`](../operator_playbooks/convolution__convtranspose2d.md) — ConvTranspose2d 算子（2D 转置卷积）

## §0 适用范围与算子分类

| 算子 | 计算特征 | 核心优化策略 | 性能基准 (geomean vs torch) |
|------|---------|-------------|---------------------------|
| ConvStandard1d | 1D 标准卷积，K-tap | stride==1→shifted 1×1 GEMM（groups>1 用 general shifted）；stride>1→Triton im2col+GEMM 直写+bias 融合；padding 用 `torch.zeros`+`copy_`（禁止 unfold/permute）；动态 BLOCK 分档；im2col BLOCK_L 自适应 | **~0.84x**（target 0.8x，达标） |
| ConvDepthwise2d | 2D depthwise，无跨通道规约 | element-wise mul + 3D/5D block_ptr；k=5/7 → split-KHW+atomic | **~0.90x**（达标） |
| ConvTranspose2d | 2D 转置卷积 | dilate+pad → flipped-weight stride-1 Conv2d 等价转换；确定性 BLOCK_C 启发式；连续块划分 + 最内层 ow_blk 解码复用 w_blk 缓存 | **0.8297x**（target 0.8x，**达标**） |
| ConvStandard3d | 3D 标准 + grouped（源 benchmark 实际 `kernel_size=(k,k,1)`） | NC1DHW0 + 1×1 分解（主）；**P3 no-permute 布局 + HW 展平 dot**（主力）；Cout_pg<16 → groups-batched M=16 | **1.1332x**（**达标** target 0.8x，60/60 pass） |

**归类规则**：算子名含 `Conv1d`/`ConvStandard1d` → §2；`ConvDepthwise2d` → §3；`ConvTranspose2d` → §4；`ConvStandard3d` → §5；`Conv2d`/`ConvStandard2d` → §6。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束是卷积类算子**共有**的工程规则。各算子章节只补充**特化**部分（fan_in 口径、路径选择、门控阈值等）。

### Layer 1：通用设计约束

#### G1 手工 CPU 初始化权重/偏置
- **必须**：按 `nn.ConvNd.reset_parameters` 原理生成；CPU 上 `empty + uniform_` 后再 `.to(device)`；per-config 种子 `torch.manual_seed(hash(key) & 0xFFFFFFFF)`；顺序**先 weight 全量，再 bias 全量**。
- **禁止**：调用 `nn.ConvNd`；在 NPU 上 `uniform_`；在 `__init__` 中做权重预计算（必须移到 `forward()` 内）。
- **Why**：NPU RNG 与 CPU 同种子序列不同 → 精度失配。
- **fan_in 口径（各算子不同，见对应章节）**：
  - 标准卷积（1d/2d/3d）：`fan_in = (C_in // groups) * Π(kernel_size)`
  - depthwise（groups==C_in，即 C_in//groups=1）：`fan_in = Π(kernel_size)`
  - ConvTranspose2d：`fan_in = out_channels * Π(kernel_size)`（用 **out_channels**，与标准 Conv 相反）
  - 统一：`bound = 1 / sqrt(fan_in)`，`uniform_(-bound, bound)`。

#### G2 一维 grid + 核数上限
- `grid = (min(total_tiles, num_cores),)`；kernel 内 `for tile_idx in range(pid, total_tiles, NUM_CORES)` 覆盖。
- **禁止** 3D grid；**禁止**交织划分 `range(core_id, total, NUM_CORES)`（小 shape 延迟偏高，改连续块划分）。
- **Why**：Ascend 仅支持一维 grid；任务数超核数必须程序内循环覆盖。

#### G3 禁止 `%`（减法代模）
- tile / oh / ow / kk / group 解码全部用 `a - (a // b) * b`；比较用 int32。

#### G4 block_ptr 向量化访存 >> pointer-tensor gather
- 输入/输出加载优先 `tl.make_block_ptr` + `boundary_check`。
- **Why**：`tl.load(ptr + computed_offsets)` 走 gather（实测 ~0.8 GB/s）；`tl.make_block_ptr` 触发 `DataCopy`/`DataCopyPad`（实测 ~40 GB/s）。

#### G5 block_ptr 负 offset 必须 host 预补零
- pad>0 时先用 Triton pad kernel 把输入补成 `x_padded`，使 kernel 内 offset 恒 ≥ 0。
- **Why**：`boundary_check` 对负 offset 返回**整个 block 全零**（非逐元素零）→ 边界区域精度失败。

#### G6 forward() 禁止 torch/F 计算接口
- **禁止**：`F.pad`、`F.conv*`、`torch.unfold`（计算用）、`permute`（计算/拷贝重排）、切片赋值、`min()/max()`（validator 视为计算操作）。
- **允许**（纯布局/元信息）：`view / reshape / permute(纯布局) / contiguous / squeeze / .to`。
- padding / layout 转换 / 转置必须封装进 `@triton.jit` kernel。
- **Why**：validator Type3 退化检测；host 侧 torch 计算引入大拷贝与额外 launch。

#### G7 大 K_dim 保留 ieee 精度
- `tl.dot(..., input_precision="ieee")` 在 Ascend 触发原生 f32 Cube 路径。
- **Why**：大 reduction 维 tf32 精度不足。

#### G8 block_ptr 连续轴必须是单位 stride（⚠️强约束）
- `order` 中连续轴（stride=1）若实际 stride≠1，**K≥3 起读到数据错位**（K=1 因只取单 slice 不触发）。
- **解法**：device 转置使 reduction 轴 stride=1（如 weight 物理布局 `[C_OUT, C_IN, K]` 中 C_IN stride=K≠1，须转置为 `[C_OUT, K, C_IN]`）。
- **Why**：Ascend block_ptr 对非单位 stride 连续轴 lowering 有误；`boundary_check` 不救（只截 shape 边界，不截 stride 语义）。

#### G9 device 侧权重变换缓存键必须随源权重唯一（⚠️强约束）
- 任何对源权重做 device 变换（transpose/pack/reshape）后的缓存，键必须随源权重唯一。
- **推荐键**：`weight.data_ptr()`（源权重由 `_CONV_CACHE` 持久持有、跨 config 不释放，data_ptr 稳定唯一）或完整 conv_key `(in,out,K,stride,pad,dil,groups,bias) + variant tag`。
- **Why**：两个 config 共享 `(cin,cout,K)` 但权重张量不同 → 命中陈旧变换 → 该 config 全 shape 精度失配（rel err 1~2x，非精度问题）。

#### G10 reduction 维对齐 Cube C0=16
- `K_dim` pad 到 16 倍数；`BLOCK_K ≥ 16`（小 K case 不退化到 8）。
- **Why**：Cube 原生宽度 C0=16，未对齐走非 Cube 路径或精度异常。

#### G11 输出并行 tile + 禁止无保护 atomic_add
- 每个 program 负责一个输出 tile，内部规约后一次 `tl.store`。
- atomic_add 仅在受控 split 优化中使用：输出先 zero、全部 chunk 用 atomic_add、bias 仅 `k_split==0` 时加；**禁止** chunk 0 用 `tl.store` + 后续 atomic_add（race → err 2~3）。

#### G12 groups > 1 按组处理
- `g = co // C_out_per_group`；输入用组内相对索引 + 组起始 `g * C_in_per_group`；权重用 co_global + ci_local。
- **禁止** 用统一 GEMM 导致输出通道混组（精度全错）。

#### G13 Triton-Ascend 平台硬限制
- **禁止** kernel 内嵌函数定义（全部内联）。
- **禁止** 3D grid（见 G2）。
- **禁止** 对已加载的局部张量做切片 `tensor[a:b, c:d]` → `unsupported tensor index: slice`，编译期失败。
- **禁止** 把 `reshape`/`trans` 的输出直接喂 `tl.dot`（布局元数据被忽略，store 取值对但 dot 全错）；dot 两操作数必须来自 genuine 2D `block_ptr`/`tl.load`。
- **禁止** 3D batched `tl.dot`（编译/执行挂死 >10min）。
- **禁止** 用运行时谓词包绕 `tl.dot`（`if runtime_cond: acc = tl.dot(...)`）：Ascend `linalg_to_bin` / bishengir-compile pass 崩溃，全 case `MLIRCompilationError`（ConvTranspose2d opt_iter_0 "2-oh w_blk hoist" 实测 50/50 编译失败）。需用**编译期 `tl.constexpr` 分支**或**双命名累加器 + 无条件 dot**规避；运行时边界只能靠 `tl.load` 的 `mask`/`boundary_check` 处理，不能用 `if` 包 dot。

---

### Layer 2：通用算法骨架

#### A1 K-tap → K 个 shifted 1×1 block_ptr dot 分解（跨 1d/2d/3d 标准卷积的算法突破）
- 把 K-tap 卷积拆为 **K 个 shifted 1×1 卷积**，reduction 维从 `C_in·K` 降到 `C0=16`（Cube 原生宽度），K 次 `tl.dot` 累加进同一 accumulator。
- stride==1 时每个 tap 的输入列连续，**一次 block_ptr load 取整块**，无需 host im2col 物化；dilation 仅改变 tap 起点 `l_start + k*DIL`。
- 权重需 device 转置使 reduction 轴 stride=1（见 G8）。
- **适用**：groups≥1 标准卷积 + stride==1（Conv1d 主路径 / ConvStandard3d T7 NC1DHW0 / NCHWc+5D）。
- **收益**：单 shape 1.5–8x；reduction 维对齐 Cube C0=16 是关键（见 [[feedback_nc1hwc0_breakthrough]]）。

#### A2 NC1xC0 打包（通道内层 C0=16）
- pack x 为 `[N, groups, C1_pg, <spatial>, C0]`（2D=NCHWc / 3D=NC1DHW0 / 1D=NC1LC0），让 16 通道占 1 cacheline，利用率 100%。
- 权重按 `[groups, Cout_pg, C1_pg, <K>, C0]` 打包；配合 A1，每个 tap 读 `w[...k...]` + `x[...c0...]` 做 Cube dot。
- stride==1 可保留 `[N,g,C1,C0,<spatial>]`（W 内层连续，no-permute）；stride>1 必须 permute 到 C0 内层，否则 block_ptr 非法/gather 回退。

#### A3 im2col + GEMM（带 K_dim·M 门控）
- 两阶段：im2col kernel 把 x scatter-gather 为连续 `x_col`（输出端 block_ptr 连续 store），再 `W @ x_col` block_ptr Cube dot。
- **必须** host 门控（按 `K_dim * M` 阈值，需按数据集校准）：小 case Python/内存开销抵消收益。
- 仅当 A1 不适用（stride>1 列不连续）时使用。
- `x_col` reduction 顺序须与 weight 展平顺序一致（如 `(C, kd, kh)`）。

#### A4 depthwise element-wise vector 路径（无跨通道规约）
- depthwise 每个输出通道只依赖 1 输入通道（groups==C_in），**无跨通道规约**，KHW≤49。
- **禁止** `tl.dot`（Cube 16³ 阵列浪费 15/16 行；ascend910B1 cube:vec=1:2 下 Vector 路径更优）。
- 采用 Python 常量 KHW 循环 + `[BLOCK_M, BLOCK_N]` 向量化 element-wise mul + accumulate。

#### A5 多路径 hybrid 分派
- host 按 `(stride, padding, groups, kernel_size, 通道数)` 选路径；每条路径独立准备 pack 与 kernel，避免单 kernel fuse 导致 UB 放不下常驻累加器。

---

## §7 常见陷阱与避免方法

> 仅列出 §1 G1–G13 之外、带具体失败症状的陷阱；通用陷阱（NPU 权重初始化、`%`、3D grid、负 offset 未补零、autotune symbol 重复传入、kernel 内嵌函数定义、`reshape/trans` 喂 dot）见对应 G 编号。

### §7.1 ConvStandard1d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| 对非连续输出 `out.view(C_out, N*L_out)` | 精度全错、数值巨大 | 用 T7 直写 `out[N,Cout,Lout]`；确需中间 buffer 先 `empty` 计算再 reshape/copy_ |
| group weight tap-k base 漏 `+k*Ci_pg` | 所有 k 读同一 slice，全错 | 打包布局 `[g,co,k,ci]` 下 base 必须含 per-k 偏移（T6） |
| autotune 引入错误配置 | 验证大量失败 | 小范围手动调参；autotune 后必须全量验证 |

### §7.2 ConvDepthwise2d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| 全场景用 5D block_ptr | stride=1 被迫小 tile，整体退步 | stride=1 保留 3D KHW-loop block_ptr |
| stride≥2 用 3D block_ptr + H/W stride trick | 仍有 KHW 次冗余 DRAM 读 | 升级到 5D block_ptr 单次 load 整个 rf |
| stride=1 走 NCHWc | tile 小，并行度下降 | 仅 stride≥2 用 NCHWc |
| k=5/k=7 UB-limited 不做 split | tile 被迫很小 | 按 §3.2 阈值启用 split-KHW + atomic_add |
| chunk 0 用 `tl.store` 替代 atomic_add | race 覆盖，err 2~3 | 全部 chunk atomic_add，输出先 zero（G11） |

### §7.3 ConvTranspose2d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| 权重未 flip | 等价转换错误 | `weight_prep_kernel` 内按 `KH-1-kh`/`KW-1-kw` 读 |
| pad 公式用 Conv2d 口径 | shape/精度错 | `pad = K - 1 - P` |
| weight bound 用 in_channels | fan_in 错 | `bound = 1/sqrt(out_channels * KH * KW)` |
| weight shape 维度搞反 | (in,out,KH,KW) 搞错 | ConvTranspose2d weight 为 `(in_channels, out_channels, KH, KW)` |
| conv kernel 直接写 NCHW 不做 UB transpose | 单 case 0.01ms→0.18ms | 先 UB 内 `tl.trans` accumulator，再 3D block_ptr 写 NCHW |
| 对 UB 中加载的 rf 张量切片 | `unsupported tensor index: slice` | 改用全局内存侧按 (kh,kw) 分别 block_ptr load |
| conv kernel 用 `@triton.autotune` 选 `BLOCK_C` | 运行间 geomean ±0.02~0.03、单 case ±0.27x，结论不可复现 | host 侧确定性启发式 `BLOCK_C = min(pow2≤C_in_pad, 64)`（C4d-DETERMINISTIC-CONV-CONFIG） |
| padded `out_workspace` + host `[:, :out_channels].contiguous()` 切片 | 全输出张量拷贝纯开销，大输出拖累延迟 | 直接分配精确 `out_channels` + store `boundary_check` 掩码（C4d-EXACT-OUTPUT） |
| 直接从原始 NCHW 做动态 clamp block_ptr load | aicore timeout / scalar 降级 | 先 prep kernel 生成 padded NHWC workspace 再读 |
| 返回 `out_workspace` 未切片 | 输出 shape 为 `(N, C_out_pad, H_out, W_out)`，verify 报 shape mismatch | 返回 `out_workspace[:, :out_channels, :, :].contiguous()` |
| 未缓存 device 转换后 weight | 0.60x→0.24x | `_weight_cache` 按 G9 键复用 |
| 未缓存 NHWC workspace | 0.60x→~0.45x，prep kernel 重复 alloc | `_workspace_cache` 按 shape/dtype/device 键复用 |
| 交错划分 `range(pid,total,NUM_CORES)` + ow_blk 非最内层解码 | 小/中 shape geomean ~0.78x，w_blk 跨 tile 无 L1 复用 | 连续块划分（start/cnt 形式）+ ow_blk 作最内层解码维（§4.3a，+6.6%） |
| 运行时 `if cond: tl.dot(...)`（如 2-oh hoist 的 `if valid1:`） | 全 case `MLIRCompilationError`（bishengir-compile 崩溃） | 运行时谓词不得包绕 dot；用 `mask`/`boundary_check` 处理边界，或编译期 constexpr 分支（G13） |

### §7.4 ConvStandard3d

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| kernel_size 维度顺序错 | 输出 shape 错 | 元组顺序 (D, H, W) |
| `for c1_inner in range(...)` | 非 static_range 结果错 | 用 `tl.static_range`（T7） |
| weight c1_inner stride 写错 | C1_pg=1 掩盖，>1 全失败 | stride=`Kd*Kh*C0`（T7） |
| block_ptr order 与 strides 不匹配 | ow=0 偶然正确，ow>0 全错 | order 按 stride 从小到大排列 |
| Cout_pg<16 降 BLOCK_COUT 到 4/8 | Cube 16×16 下限，收益噪声 | 改 groups-batched M=16（T9） |
| `tl.arange(GROUPS_PER_M*COUT_PG)` 非 constexpr 报错 | 编译错 | 显式 `M_FLAT` 作为 constexpr kernel 参数（T9） |
| NC1DHW0 stride>1 尝试 no-permute | gather 回退 -26%~-50% | stride>1 必须 permute 到 C0 内层（T14） |
| block_ptr store 跨 group 溢出 | store 用绝对地址 `shape=(C_out, HW_out)` 只截总 C_out 不截 group 边界，`Cout_pg<BLOCK_COUT` 时写零污染下一组 | constexpr `FULL_COUT=(BLOCK_COUT==Cout_pg)` 分支：满 tile 才 block store，否则 masked scalar store；不加 `tl.where` 钳位（T20） |
| im2col kernel 用 `n * NCDHW` 当 batch 偏移 | batch>1 后续 batch 全零/越界 | 用 `n * CDHW`（§5.1） |
| HW-flat 无门控全量 dispatch | K≥7 或 C 超限时 timeout / impl 慢 10x | 仅 `C_in_pg≤16、C_out_pg≤64、K≤5` stride==1 启用（T20） |
| per-kd 合并 dot（reshape 3D load 扩大 K-dim） | dot 消费 reshape/trans 数值全错 | 保持 per-(kd,kh) K-dim=C0 小 dot（G13） |
| block_ptr `order=(0,1)`（最内轴非 stride=1） | **静默错值**：impl 全零 / 常数（不报错，36 case 全 fail） | Ascend 强约束：block_ptr 永远 `order=(1,0)`，布局让最后一维 stride=1 连续。改 pack 布局使最内轴 stride=1：P3 no-permute `[N,g,C1,C0,Dp,Hp,Wp]` 使 W stride=1、flat HW 轴连续（T20）；旧 C0-HW 交错 `[N,g,C1,Dp,C0,HWpad]` 同理使 HW 展平轴 stride=1 |
| gather+static_range kernel 的 `HAVE_BIAS=False` 特化 | **挂死 aicore**（507014 timeout / 507057 SUSPECT REMOTE ERROR）；同参数 bias=True 正常 | 该 constexpr 取值组合触发编译 bug；bias=None 时建零向量 `torch.zeros(Cout,...)` + 强制 `HAVE_BIAS=True` 绕过（零偏置不影响精度）。判据：两 case 仅一 constexpr 不同、一个挂死一个正常 → 嫌疑特化 bug |
| 隔离诊断 harness 忘 `x.to('npu')` | NPU kernel 读 CPU 内存报 507057（伪故障，误判 kernel 死） | `get_input_groups()` 返回的 tensor 常在 CPU；自写 harness 必须 `x.to('npu:0')` + `torch.npu.set_device`；verify.py 内部已搬 NPU 故其结果有效 |
| **P2 `xm` block_ptr order 与 strides 不匹配（packed block_ptr）** | 全 case 静默错值（ow=0 可能偶然对，ow>0 错） | `xm` 内存为 `[..., Wp, C0]`，读 `shape=(C0,Wp)` 时须 `strides=(1,C0), order=(0,1)`；若误用 `(C0,1), order=(1,0)` 则 W 维 stride 实际为 C0 而非 1，导致跨 W 错位（§5.3 T9 / §5.5 历史里程碑 opt_iter_7） |
| **P2 M-block group 偏移未局部化** | groups>GROUPS_PER_M（case19/44 g=8）时大块数值错误，g≤GROUPS_PER_M 时可能碰巧对 | `repack_weight_m_kernel` 必须用 `g_in_block = g - (g//GROUPS_PER_M)*GROUPS_PER_M` 计算 c0 偏移，与 `pack_input_m_kernel` 每 M-block 重置对齐（§5.3 T9） |
| **缓存分支内定义布局常量** | verify 单轮通过，benchmark 多轮调用报 `UnboundLocalError`（如 `WM_M_STRIDE`） | 由 tensor 布局决定的 strides（`WM_M_STRIDE`、`WM_KK_STRIDE` 等）必须在 `if cache is None:` 分支之前赋值，保证 cache 命中分支也能访问（§5.3 T9） |
| **P3 kernel 形参含 `stride` 但 `_launch_hwflat` 不传** | `NameError: name 'stride' is not defined`（34 case 全 fail，仅 P3 路径） | P3 门控恒 stride==1，host 显式传 `1`。旧 hwflat kernel 隐式假设 stride==1 不接 stride 参数；迁移到带 `stride: tl.constexpr` 形参的 kernel 时易漏（T20，opt_iter_26 首轮踩坑） |
| **P3 no-permute 布局但改用分离 W 分块** | 0.4632x 回退（`BLOCK_W≤64`、tile 数 `Hout×` 增加、丢失大块连续 Cube load） | no-permute 必须**叠加 flat-HW dot**（保留大 `BLOCK_HW≤256`），二者缺一不可。仅换布局不换 tile 策略会退化（opt_iter_25 已证伪，见 §8.1） |
| **P3 packer gather 直读（旧 C0-HW 布局）** | pack pass 偏慢，整体 0.6745x 未达标 | 改连续 4D block-copy packer（`_pack_x_nc1dhw0_grouped` + `pad_copy_4d_kernel`，T21 实例），pack 纯连续拷贝 → 1.1332x 达标 |

---

## §8 已验证无效方向 / 未验证方向

### §8.1 已验证无效 / 需避免方向

**ConvStandard3d（ascend910b1，60 cases，kernel_size 实际生效为 (k,k,1)）**：
1. 4D block_ptr depthwise fast path：在该 60-case 集合上 geomean 从 0.2976x 跌至 ~0.2745x，必须回退 scalar direct-conv fallback。
2. `CHANNELS_PER_BLOCK = groups_per_m * C_in_pg` 作为 `tl.constexpr`：非 16 的 channel block 被编译器 scalarize，整体下降 ~10–15%。
3. 受控 im2col + GEMM（groups==1，S>1/Dil>1，K_dim≥64，Co≥32）：geomean 从 0.3109x 崩塌至 0.2310x，必须关闭。
4. no-permute 对 stride==1 中小 C case 有稳定增益，但无法弥补整体与 aclnn 的差距。
5. per-kd 合并 dot（reshape 3D load 扩大 K-dim）：dot 消费 reshape/trans 数值全错。
6. 单 dot 加载完整 `K_dim > 128`：触发 `MLIRCompilationError`。
7. 专用 1×1 纯 matmul：K 极小，launch/overhead 主导，geomean 下降。
8. 两阶段路径加大 `BLOCK_KK=64` 或 `BLOCK_HW=128`：UB overflow。
9. **stride==1 no-permute 路径的 host copy 融合**（`torch.zeros + copy_` → device `pad_input_kernel`）：2026-08-05 验证，no-permute case latency 下降 35–60%，但整体 geomean 从 0.7789× 微跌至 0.7694×，未成为有效全局优化方向。
10. **P4 `BLOCK_COUT=256` for Cout_pg>128**：单 case latency 可能微降，但 60-case geomean 从 0.6745x 降至 0.6366x，整体回退。
11. **P3 `BLOCK_HW=512`**：aicore timeout / 编译极慢，必须保持上限 256。
12. **§4.3a 连续块划分 + 最内层 tile 维度复用 `w_blk`**：当前 tile 已内嵌完整 `ow`/`HW` 循环，w_blk 已复用；改为连续块划分后 geomean 从 0.6745x 降至 0.6299x，不适用。
13. **收紧 P2 门控**（如要求 `groups*Cin_pg==16` 或排除更多 depthwise）：会把 case 20/45 等挤到 P3，整体 geomean 下降。
14. **P4 K=7 / P5 stride=2 K=7 路由到更大 BLOCK 或回退 P5**：case 48 等 bottleneck 无改善，整体 regress。
15. **opt_iter_24：depthwise s1d1 专用 scalar kernel（无 tl.dot）**：把 `stride==1 && dilation==1` 的 depthwise case（`Cin_pg==Cout_pg==1`）从 P2 mgrouped 改走专用 scalar depthwise kernel。geomean 0.6745x→**0.5610x** 回退。原因：depthwise 无跨通道规约，scalar kernel 无法饱满 Cube；P2 mgrouped（block-diagonal weight + Cube dot）对 depthwise 更优。结论：**depthwise 留在 P2，不要单开 scalar 路径**。
16. **opt_iter_25：P3 no-permute 布局 + 分离 W 分块**：把 P3 改 no-permute 布局但 compute 用 per-`(od,oh)` tile + 内层 `ow` 循环（`BLOCK_W≤64`），而非 flat-HW dot。geomean 0.6745x→**0.4632x** 回退。原因：丢失 flat-HW 的大 `BLOCK_HW` 连续 Cube load，且 tile 数 `Hout×` 增加。结论：**no-permute 布局必须叠加 flat-HW dot**（opt_iter_26 正解，见 §5.3 T20），二者缺一不可。
17. **（有效方向，勿与 4/16 混淆）P3 no-permute 布局 + flat-HW dot（opt_iter_26）**：no-permute 物理布局 `[N,g,C1,C0,Dp,Hp,Wp]`（连续 4D block-copy packer）**叠加** flat-HW dot（大 `BLOCK_HW≤256`），geomean 0.6745x→**1.1332x 达标**。与失效方向 4（P4 standalone no-permute ~0.33x）、16（no-permute+分离 W）的区别在于：本方案把 no-permute 当**物理布局**用，compute 仍走 flat-HW 大块 dot。
18. **旧 C0-HW 交错布局 P3（`[N,g,C1,Dp,C0,HWpad]` + `pack_input_hwflat_kernel`/`conv3d_hwflat_kernel`）**：0.6745x，**已被 opt_iter_26 取代**（pack 是 gather 非连续、compute 未利用 no-permute 的连续拷贝优势）。除非复现历史，新实现请用 §5.3 T20 方案。

**ConvTranspose2d**：
- RF block load + UB 内切片：不支持对加载后局部张量切片，编译期失败。
- K=4/5 stride=1 直接 NCHW scalar-load：触发 aicore timeout 或严重 scalar 降级。
- device-side im2col + 单大 GEMM：额外 global-memory round-trip，geomean 降至 ~0.36x。
- Winograd F(2×2, 3×3)：实现复杂、kernel 输出 NaN、触发 NPU 设备级异常（507057），风险不可控。
- 单 kernel fuse prep+conv：UB 放不下常驻累加器，性能显著退步。
- 对 K=4/5 继续调 BLOCK 参数：瓶颈在 KH×KW 循环调用次数与规约维对齐，调参收益已耗尽。
- 2-oh w_blk hoist（每 program 处理 2 行 oh，用 `if valid1:` 包第二个 `tl.dot` 复用 w_blk）：全 case `MLIRCompilationError`（运行时谓词包 dot，见 G13）。w_blk 跨行复用方向本身合理，但禁止运行时谓词包 dot；若要复用须改编译期 constexpr 分支 + 无条件双累加器。

**ConvStandard1d**（最终 geomean **0.8382x**，达标 0.8x；约束 C1d-NO-LAYOUT-OPS 内已无有效方向，有效技巧见 §2.3 T5–T9）：
1. **大 GEMM tile 调优**（opt_iter_15，放宽 acc 上限到 8192，dot 调用 −75%）：idx25 分量 profile 显示 im2col gather 占 impl 97%、GEMM <3%，tile 改了仅 −2%，geomean 持平。
2. **`make_block_ptr(strides=(STRIDE,))` strided load 加速 gather**：与 pointer-tensor `tl.load(ptr+arange*STRIDE)` 同速（0.98x，maxerr=0），Ascend 不把 1D strided block_ptr lower 成更快 DMA。
3. **im2col 按 K 维 2D tile load `[K_PAD,BLOCK_L]`**（想降 load 指令数 N·C·K→N·C）：A/B 实测 case25/36/6/42 慢 2-3×，BL=256 时 UB 溢出（multi-buffer 需 232KB>192KB）；Ascend 把 2D strided 计算指针 load 分解成逐元素，有效指令数不降反升。
4. **Path A（shifted GEMM）tile 调优**：case24 全 tile ±5%，且大 GEMM(case24) 与 tiny(case2) 核时间相近 → launch/计算地板 bound，非 GEMM 效率 bound。
5. **groups>1 / stride>1 直接 Vector element-wise conv（无 im2col）**：per-output gather `Cin_pg·K` 是 im2col 的 ~160×（case36），gather 已是地板，省下的 workspace 往返无法抵消。
- **结论**：stride>1 gather 是 Ascend 指令吞吐地板（case 25 ~1.1ms 占 impl 97%）；任何形式 strided 2D 访存（block_ptr strided / 显式 2D 计算指针）都被分解为逐元素，**约束内唯一有效手段是 1D BLOCK_L 放大（T9）**。唯一能进一步破地板的是放宽约束回 host `unfold/permute`（约束前 1.06x），需用户显式同意。

## 并入：convolution__convstandard2d

## §6 ConvStandard2d（2D 标准 + grouped + depthwise-like）

待归档

---

## 并入：op-coding 的 conv 编写注意事项

原文——

> 注意conv类卷积算子编写：
torch module中的卷积算子生成会包含一个随机权重weight，为保证我们生成的triton实现也具有相同的结果，我们可以在triton的host侧代码中生成对应的weight，例如：
```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triton_kernel():
    pass
def triton_host():
    args = ...
    weight = nn.conv(**args).weight.to(device)
```
具体的参数和''nn''中调用的module要与torch保持一致，device与传入的backend保持一致（如"npu"），我会在调用triton之前固定相同的随机种子，所以在这里只需要正确地创建类的实例，并导出权重

---
