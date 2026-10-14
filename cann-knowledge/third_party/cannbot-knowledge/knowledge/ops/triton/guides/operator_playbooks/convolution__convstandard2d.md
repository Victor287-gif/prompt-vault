---
type: Guide
title: ConvStandard2d 算子（2D 标准/grouped 卷积） 实现手册
description: '典型特征: (N, C_in, H, W) × weight [C_out, C_in//groups, K, K] → (N, C_out,
  H_out, W_out)；stride/padding/dilation/groups/bias 全为 forward 运行期动态参数，含标准 / grouped
  / 1×1 / depthwise-like（Cin_pg==1 && Cout_pg==1）变体。'
tags: [task.implementation, topic.compute.neighborhood.convolution, topic.compute.composite, topic.tiling, topic.data_movement.memory_access, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: supporting
created_at: '2026-08-26T08:28:28Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# ConvStandard2d 算子（2D 标准/grouped 卷积） 实现手册

> 类别约束见 [convolution_constraints.md](../constraints/convolution_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §6 ConvStandard2d 算子（2D 标准 + grouped + depthwise-like）

**算子类别**：`convolution-2d`　**典型特征**：`(N, C_in, H, W)` × weight `[C_out, C_in//groups, K, K]` → `(N, C_out, H_out, W_out)`；`stride/padding/dilation/groups/bias` 全部为 forward 运行期入参（非构造期固定），覆盖标准 / grouped / 1×1 / depthwise-like（`Cin_pg==1 && Cout_pg==1`）变体。
**性能基准**：50/50 verify pass，geomean **0.6677x** vs torch aclnnConvolution（target 0.8x，**未达**；ascend910B1），实现平均延迟 0.0411ms / 框架平均 0.0274ms。多路径混合分派实现（§6.2），复现代码 = 工作目录 `7_ConvStandard2d_generated.py`。

### §6.1 Layer 1：设计约束（算子专属）

- **fan_in**（G1）：`fan_in = (C_in // groups) * K * K`；权重缓存键必须是**完整 conv_key** `(in_channels, out_channels, K, stride, padding, dilation, groups, bias)`。
- **C2d-WEIGHT-CACHE-INSTANCE-RESET ⚠️ 权重/布局缓存必须实例隔离**：缓存放实例内字典（键 = 完整 conv_key）。
  - **Why:** 每个实例代表一次全新评测上下文；模块级缓存跨实例复用 → 前 case 的权重/派生布局污染后 case，出现**序列相关的 AccuracyError**（实测 case 49）。
  - **How to apply:** 若用模块级缓存，必须在 `ModelNew.__init__` 清空；派生布局与主条目同缓存（T2d-10）。
- **C2d-DOT-OPERAND-CONTIG ⚠️ tl.dot 操作数不得来自 strided/computed-offset gather**：非连续（stride>1 滑窗 / 跨步采样）输入喂 Cube 前，先在 host 侧重排成仿射可达布局（奇偶平面拆分 T2d-2 / channel-last 打包 T2d-7 / 预补零 T2d-1），或物化连续 im2col（load 侧 gather、store 侧连续写）再 GEMM（T2d-9）。
  - **Why:** strided 计算指针 load 直喂 dot 在 Cube 上 mis-lower，精度错。
  - **How to apply:** **禁止** computed-offset pointer load 直喂 dot；三选一：物化 im2col / NCHWc C0 内层 pack / gather 进 UB tile 再单 dot。
- **C2d-UNIFIED-GRID**：所有路径统一一维 grid = tile 总数与 vector 核数取小（`vector_core_num`，兜底 40）+ kernel 内 grid-stride 循环（每核从自身序号按总核数步进），dot 路径与向量路径共用同一核数模型。
  - **Why:** Ascend 仅支持一维 grid（G2）；向量/点积路径统一核数模型便于并行度守卫。
- **C2d-NO-MIN-MAX**：host 侧禁 `min()/max()`（validator 视为计算操作 → Type3 退化判定）。
  - **How to apply:** grid 裁剪用条件表达式：`v = cap if total >= cap else total; return v if v >= 1 else 1`。
- **C2d-BM-GROUP-CLIP ⚠️ BLOCK_M 超过 Cout_pg 时必须按组裁剪**：Cube M 维下限 16，`Cout_pg<16` 时 BLOCK_M 强制补到 16 会多出越组行。
  - **Why:** 越组行用**本组输入**算了**别组输出通道**并写穿，静默精度错。
  - **How to apply:** store/bias 的 mask 必须含 `(pid_co*BLOCK_M + tl.arange(0, BLOCK_M)) < Cout_pg`（G12 强化版）。
- **C2d-NO-CLAMP-IN-LOAD-ADDR ⚠️ 钳位（min/max）不得进入 load 地址表达式**：逐元素钳位作用在指针偏移上后，编译器无法证明地址仿射性。
  - **Why:** 向量化 load 退化为逐元素标量 load（实测 **10–70x 回退**）。
  - **How to apply:** 边界处理首选 host 预零填充使地址恒界内（T2d-1）；确需 kernel 内防御时，钳位只用于把地址钳到合法区间防崩溃、数值置零仍由 mask 负责（masked OOB 地址在部分后端触发地址对齐崩溃，S=2 实测）。
- **C2d-S2-VECTOR-LOAD-FORBIDDEN ⚠️ stride≥2 的向量化输入 load 一律禁止**（masked/unmasked、块指针/计算指针所有写法均中招）。
  - **Why:** 本后端统一误编译（aicore 507015 崩溃或数值错）。
  - **How to apply:** stride-2 采样唯一安全解 = host 把补零输入按列奇偶拆成两个连续平面，平面内 stride-1 仿射访存（T2d-2）。
- **C2d-HOST-COPY-PATHOLOGY ⚠️ host 侧矩形切片拷贝 / stride 视图拷贝 / 通道重排是隐形大瓶颈**：这些 torch 张量操作在 NPU 上落到 AiCpu 慢速拷贝路径（**实测单次 92–176µs，可占端到端 70%+**）。
  - **How to apply:** 替代 = 行连续 load + 行连续 store 的专用拷贝核（已验证唯一高效原语；列方向 stride-2 访存即触发标量化）；多段 host 预处理应融合为单核（补零 + 平面拆分融合后再提速约 2x，T2d-5）。
- **C2d-MACS-GATE-HOST-OVERHEAD ⚠️ 带 host 预处理（补零/重排/拆分）的路径必须按算力量门控**：host 固定开销约 **40–60µs**，小算力 case 无法摊薄（**实测劣化至 0.67–0.75x**）。
  - **How to apply:** forward 内按「输出元素数 × 输出通道数 × 规约维」估算乘加量设下限阈值，不达标回退无预处理路径。
- **C2d-NO-CONDITIONAL-BASE-POINTER**：运行期标量 if 包裹的双 load 路径（如「内部 tile 无 mask 快路径」）、以及编译期 if/else 切换双基址指针，在本后端均误编译（前者 dot 输出全零、后者误差 ~125 量级）。
  - **How to apply:** 等价安全写法 = 把「选哪个基址」折成「平面序号 × 平面步长」的仿射偏移加到单一基址上。
- **C2d-PARALLELISM-COLLAPSE-GUARD**：行块化/放大 tile 使 tile 总数下降后，必须校验 tile 总数与核数同量级。
  - **Why:** 实测 tile 数 8 < 24 核时并行度塌陷，**回退 0.83x**。
  - **How to apply:** tile 总数 < 核数量级时回退小块配置。
- **C2d-SPLIT-RESULT-MATERIALIZE ⚠️ `tl.dot` 操作数必须 make_block_ptr 产物；`tl.split` 产物喂 dot 前必须 `tl.where(mask, x, 0)` 物化**。
  - **Why:** `reshape → split` 直接输出作 dot 操作数触发 `getIndexingMaps/ConvertLinalgRToBinary` 编译 abort；**强化（终局验证）**：ptr-arith 地址、masked load、`tl.where` 物化、clamp 地址任何一种形式喂 `tl.dot` 一律编译期 SIGSEGV。
  - **How to apply:** dot 输入**唯一合法形态是 make_block_ptr**（含其经 `tl.where` 物化的稳定变体）。因此「mask 读原图省 pad」类融合在 dot 路径不可实现（§8.1 #18）；W 边界行折叠 mask 无法用 block_ptr 表达（行坐标折叠在 dim1 offset 中）。
- **C2d-NO-VECTOR-INTDIV ⚠️ 向量 `//`（整除）触发标量降级，memory kernel 实测 22x 劣化**：2D tile 行重排用 `r_off // H` → 20637µs（同 kernel program_id 分解版 937µs 的 22 倍）。
  - **How to apply:** 行/列重排一律用 `program_id` 维度分解（如 `program_id(0)=channel` 免除法）或 `tid - (tid // t)*t` 减法代模（G3）；标量 if 包绕向量 load/store（762µs）与 `mask & m_scalar` 折叠（4242µs）同样劣化。
- **C2d-ODD-DIM0-BASE-UNRELIABLE ⚠️ make_block_ptr dim0 基址为奇数时 load 不可靠（含纯拷贝路径）**：dim0 基址 = 通道序号 × 单平面行数，行数为奇（如 H=7）时基址奇 → load **间歇**取错数据。
  - **Why:** 两个已证实中招：① `[N,C,2,H_alloc,W2]` 交错平面 buffer 用 `r0=(nc*2+p)` 取第二奇偶平面时，setitem/CPU 写入读回全零、arange 写入读回正确（ptr-arith 读同错）；② pad kernel 读原始输入 `base = x_ptr + nc*H_IN*W_IN`，奇 `H_IN*W_IN` 时间歇读错。
  - **How to apply:** 无 dot 的 kernel（pad/拷贝/重排）一律 ptr-arith + masked load（实测零性能代价）；有 dot 的 kernel 用独立平面 tensor（dim0=通道，无奇基址）或维持 dim0=纯通道寻址。另注：dim1 offset 含 `ih·W2` 行平面偏移时 `COLS_TOTAL` 必须传 `H_alloc·W2`（行平面展开长度），传 W2 即越界读零。
- **C2d-SKETCH-MANDATORY-PATHS ⚠️ sketch.txt 必须一次性列出 §6.2 全部 7 条路径**：`triton-op-designer` 生成 `sketch.txt` 时，**禁止**以“先实现 P1/P2/P5 三条路径保底，其余在 Phase 4 增量加入”为由裁剪路径。
  - **Why:** Phase 3 的 3.2b 架构符合性检查门逐项核对「kernel 列表完整性」「路径分派完整性」「禁用模式未出现」；任何路径缺失视为 `A-SketchDeviation`，不得进入精度验证。若强行绕过，非 1×1 case 全走 im2col，geomean 跌至 **0.025x**。
  - **How to apply:** Phase 2 草图必须显式包含 P1 1×1-GEMM、P2 depthwise、P3 skinny-chlast、P4 S2-plane、P5 tap-window、P6 plane-dot、P7 im2col+kblock-GEMM 的 kernel 名、host 门控条件、输入/权重数据布局与输出 store 方式。
- depthwise（`Cin_pg==Cout_pg==1`）**禁止** `tl.dot`（A4：无跨通道规约，Cube 16³ 阵列浪费 15/16 行；ascend910B1 cube:vec=1:2 下 Vector 路径更优）。
- 权重 dtype 需与输入对齐（`weight.to(x.dtype)`）；`bias is None` 时传 x 本体当 dummy 指针 + `has_bias=False` constexpr（该组合在本算子各 kernel 上实测可正常运行）。
- 其余通用约束见 §1 G1–G13（fan_in 口径、一维 grid、减法代模、block_ptr、负 offset 预补零、禁 torch/F、ieee 精度、单位 stride 连续轴、权重缓存键、C0=16 对齐、atomic 保护、groups 分组、禁内嵌函数/运行时 if 包 dot 等）。

### §6.2 Layer 2：算法骨架（多路径混合分派 A5）

Host 先按卷积配置（每组输入/输出通道、kc、K、stride、dilation、groups）静态选定主路径并预变换权重，forward 内再按 shape 与算力量动态回退；每条路径独立准备数据布局与启动配置：

| 优先级 | 门控条件 | 路径 | 核心并行策略 |
|---|---|---|---|
| 1 | K==1 且 stride==1 且无 pad/dilation 且 groups==1 | 1×1 纯 GEMM | 输入视作（批, 通道, 空间展平）的标准 GEMM 分块（T2d-8） |
| 2 | 每组输入/输出通道==1（depthwise-like） | 通道块二维 tile 向量乘加 | 权重按 tap-通道连续布局 + 整行宽 tile（T2d-6）；大算力叠加 host 预补零 + 无 mask |
| 3 | 每组输入通道 ≤6 且 kc ≤200 | skinny channel-last 单 dot | 打包规约维 + 行块化权重复用（T2d-7）；乘加量 ≥1.9M 门控，不足回退优先级 5 |
| 4 | stride==2 且 dilation==1 且 groups==1 | W 奇偶平面拆分 | host 融合补零+解交织，kernel 内 stride-1 仿射（T2d-2） |
| 5 | kc ≤200（其余） | 逐窗口 tap dot | 输出 tile（通道块 × 宽块），tap 全展开 + 通道分块 dot（规约块自适应 `pow2_ceil(cin_pg,16,64)`，T2d-12）；stride>1 用「钳位地址 + mask 置零」变体 |
| 6 | stride==1 且 16 ≤ 每组输入通道 ≤128（kc>200） | 零填充 + 逐平面无 mask 大 dot | host 预补零（右侧多补尾 tile 列）→ 全程无 mask 仿射 load，KC 分块 = 单 (kh,kw) 平面（T2d-1）；Wout<20 或小算力回退优先级 5（门槛经实测校准，T2d-12） |
| 旁路 | stride==1 且 K∈{3,5,7} 且 kc≥2000 且 kc×输出元素 ≥20万 | im2col 物化 + K 分块 GEMM | 抢占优先级 5/6 的大 kc case（T2d-9） |

**各路径骨架要点**：
- **P1（1×1 纯 GEMM）**：输入零拷贝改视角为（批, 通道, 空间展平），权重按（输出通道, 输入通道）行主序；tile = 输出通道块 64 × 空间块 64，通道规约块 64 循环，ieee dot + fp32 累加，偏置尾加后按 NCHW 偏移掩码直写。
- **P2（depthwise 通道块二维 tile）**：权重预重排「kh × kw × 通道」。tile = 通道块（组数 ≥16 取 16，否则 ≥4 的 2 的幂）× 行块 4 × 宽块（输出宽 ≥128 取 128，否则 ≥16 的 2 的幂，尽量整行单块）。逐 tap：输入（通道块, 宽块）仿射二维 load（行步长=通道步长、列单位步长）× 通道连续权重向量累加；大算力（批×组×输出元素 ≥32768）先 host 预补零（T2d-1）全程无 mask，小算力保留逐 tap mask 版本。
- **P3（skinny channel-last 单 dot）**：host 预补零后重排为「批 × 组 × 高 × 宽 × 组内通道」（组内通道最内连续）；权重零填充为「组 × 每组输出通道 × K × 对齐打包维」。tile = 通道块（≤64）× 宽块（≤64）× 行块（K≤5 且 tile 总数 ≥16 取 4，否则 2）。按 kh 循环：权重 tile 只 load 一次供行块内各行复用；每行输入（宽块, 对齐打包维）一次仿射 load 转置后单 dot；奇数尾行按「行号 < 输出高」mask 兜底。
- **P4（W 奇偶平面拆分，stride==2）**：host 预零填充「(2, 批, 通道, 高, 平面宽)」五维缓冲（平面宽 = 偶数化补零宽 ÷2），融合核逐行读原始行、按列奇偶拆两半连续写（pad 折入列偏移，按 pad 奇偶两种映射）。tile = 通道块 × 宽块；K² tap 展开 × 通道分块循环，行坐标 = 2h+kh 恒界内，平面选择 =「(kw 奇偶) × 平面步长」仿射偏移，平面内列 = w + kw//2 连续——全程无边界 mask（仅通道/宽尾块 mask）。
- **P5（逐窗口 tap dot，kc≤200 与兜底）**：tile = 通道块 × 宽块，tile 序号按「批→组→行→宽块→通道块」解嵌套（通道块最内，T2d-4）；K² tap 全展开 × 通道规约块循环（规约块自适应，使 cin=32/64 每 tap 单次 dot，T2d-12），每（tap, 通道块）一次 dot；权重用「组 × kh × kw × 每组输出通道 × 组内通道」转置布局（规约轴 = 组内通道连续）；输入/权重均 masked load。stride=2 变体把越界坐标钳位到界内防地址对齐崩溃、mask 仍负责置零（C2d-NO-CLAMP-IN-LOAD-ADDR）。
- **P6（零填充 + 逐平面无 mask 大 dot，stride==1 大 kc）**：host 预补零并右侧多补「步长 × 宽块」列 → 所有窗口地址天然界内且 pad 区为 0（与卷积零填充语义一致）。KC 分块 = 单 (kh,kw) 平面（块大小 = 每组输入通道，整除无尾块）：每平面输入（通道, 宽块）仿射 strided load **无 mask**；权重（通道块, 每组输入通道）行主序连续 load；逐平面 dot 累加。Wout<32 时宽块尾块浪费严重，小算力回退路径 5。
- **P7（im2col 物化 + K 分块 GEMM）**：host 补零（行连续拷贝核，T2d-5）→ 物化核按（批×组, tap, 行, 通道块）划分，源仿射无 mask load、目标行连续写「批×组, 对齐规约维, 输出空间」矩阵（规约维补到 128 倍数；缓冲未初始化分配，尾块垃圾由权重侧零填充对冲）→ GEMM：M = 输出通道块（保持 128 最小化物化矩阵重读）、N = 空间块（空间元素 <64 取 32 否则 64）、K 分块 128；累加器+双操作数受统一缓冲上限约束（通道块 × 空间块 ≤ 128×128）。

**统一要点**：
- 所有 kernel 一维 grid（启动数 = tile 总数与核数取小，核数取 vector 核数、兜底 40，C2d-UNIFIED-GRID）+ kernel 内 grid-stride 循环；tile 解码一律减法代模（G3），输出通道块放最内层（T2d-4）。
- padding 只在需要「kernel 内仿射无 mask」的路径上支付，且补零/重排/解交织全部由专用原语核完成；**通用 pad kernel（`pad_input_kernel`）必须按 2D 平面 tile 实现（~16KB tile，实测带宽 353GB/s；行粒度 1KB tile 仅 48GB/s，T2d-15），读侧 ptr-arith + masked load（C2d-ODD-DIM0-BASE-UNRELIABLE）**；逐窗口兜底路径与 1×1 路径直接读原始输入，零预处理。
- 全路径**直写最终 NCHW 输出 + 偏置融合**（bias 为空时传任一张量当 dummy 指针 + constexpr 关闭），无中间输出重排 / 独立 bias 核。
- 矩阵乘一律 ieee 精度输入 + fp32 累加输出（G7）。
- tile 自适应公式（T2d-3）：通道块 = 128（每组输出通道 ≥128）否则 max(2 的幂, 16)；宽块 = 64（输出宽 ≥64）否则 max(2 的幂, 16)；通道规约块 = 每组输入通道的 2 的幂因子（上限 128），小通道直接取 2 的幂。
- 运行期门控三件套：乘加量下限（摊薄 host 固定开销，C2d-MACS-GATE-HOST-OVERHEAD）、输出宽/输出元素下限（宽块尾块浪费）、tile 总数 ≥ 核数量级（并行度守卫，C2d-PARALLELISM-COLLAPSE-GUARD）。

### §6.3 Layer 3：关键技巧（可参考不可照抄）

- **T2d-1 host 预零填充消除 kernel 内边界逻辑（核心范式）**：把 padding 语义全部折进 host 侧预零填充缓冲区——目标缓冲按「最大窗口坐标」构造高宽（保证最深 tap 落在界内），pad 区与尾列天然为零，右侧再额外多补「步长 × 宽块」列覆盖尾 tile 越界 lane。此后 kernel 内所有输入地址恒界内，load 变成纯仿射无 mask 形式（每条 load 不再背负边界谓词链）。大 kc 的 stride==1、stride==2、depthwise 大算力、skinny 小通道路径全部采用。可替代方向：形状小到补零开销占比过高时保留 kernel 内 mask，但须按乘加量门控（C2d-MACS-GATE-HOST-OVERHEAD）。
- **T2d-2 W 奇偶平面拆分（stride==2 的 stride-1 化）**：stride-2 滑窗的本质是「输出列 w 对应输入列 2w+kw」——采样列的奇偶性只由核内列号 kw 决定（编译期常量）。host 侧把补零后的输入按列奇偶性解交织成两个等宽连续平面（合并为带前导平面维的单张量，纯 stride-1 读写）；kernel 内以「(kw 奇偶) × 平面步长」仿射偏移选平面，平面内列号 = w + kw//2 对输出列连续 → 全部 load 退化为 stride-1 仿射；行坐标因缓冲按最大窗口坐标构造而恒界内。补零与解交织融合为单核（列偏移按 pad 奇偶分两种映射）。实测把 stride=2 组从 ~0.10x 量级提到 0.3x+。
- **T2d-3 自适应 tile（按实际维度取 2 的幂）**：输出通道块 / 宽块不按阈值固定分档，而是按每组实际输出通道数与实际输出宽「向上取 2 的幂」，夹在下限 16、上限（通道 128 / 宽 64）之间；每组输入通道 <16 时通道规约块直接取通道数的 2 的幂，避免对齐填充浪费。参数扫描实测相对固定分档 **+77.6%**（geomean 0.1551→0.2755）。要点：小 shape 下固定 tile 的 lane 浪费是首要损失源。
- **T2d-4 tile 序号解嵌套：输出通道块放最内层**：一维 tile 序号按「批 → 组 → 输出行 → 宽块 → 输出通道块」从外到内解码（减法代模），使相邻核拿到**相同空间位置、不同输出通道块**——同一输入 tile 在 L2 中被多个输出通道块复用。输出通道块放最内层是关键（空间维放最内时输入复用消失）。
- **T2d-5 host 拷贝核化（行连续原语 + 病态形状门控）**：host 矩形切片拷贝/解交织等预处理触发 AiCpu 慢速路径时（C2d-HOST-COPY-PATHOLOGY），改用专用拷贝核：行基址用「行号 ÷ 输入高」解码批×通道、余数为源行，每行按输入宽连续 load、目标行基址连续 store。已验证边界：**列方向 stride-2 访存是标量化触发条件，行方向连续则始终编译正确**。门控：仅病态形状（通道 ≥128 且输入宽 ≤32）启用——普通形状 host 拷贝 38–60µs 已快于拷贝核 ~41µs 地板，盲目替换反而回退。
- **T2d-6 depthwise 通道块二维 tile（权重按 tap-通道连续布局）**：tile = 「通道块 × 输出行块 × 整行宽」；权重预重排为「kh × kw × 通道」布局，使每个 tap 的全部通道权重是一次连续向量 load；输入按（通道块, 宽块）二维仿射 load（行步长 = 通道步长，列单位步长）；逐 tap 向量乘加，无矩阵乘。相比「每 tile 单通道单行 + 逐 tap 标量权重 load」实测 **8.3–12.4x**。大算力时叠加 host 预零填充（T2d-1），kernel 内 load 全程无 mask（实测 275→101µs）。
- **T2d-7 skinny 小通道：channel-last 打包规约 + 行块化权重复用**：每组输入通道 ≤6 时，host 把补零输入重排为「批 × 组 × 补零高 × 补零宽 × 组内通道」布局（组内通道最内连续），使打包序号 r = kw×组内通道+c 成为对宽方向步长 = stride×组内通道 的仿射坐标、且 r 自身单位连续——kernel 内无需在统一缓冲区逐行组装，直接一次（宽块, 对齐规约维）load 后转置做单 dot。规约维 = 打包维对齐到 ≥16 的 2 的幂（lane 浪费从 5.3x 降到 ~1.5x）。行块化：每 tile 处理 2（K≤5 可 4）个输出行，每个 kh 的权重 tile 只 load 一次被多行复用（实测 280→230µs；行块 4 再 103→95µs）；奇数行由「行号 < 输出高」mask 兜底。可扩展到 grouped（组维进 tile 解嵌套），但需乘加量门控（≥1.9M，不足回退逐窗口路径）+ tile 总数守卫（C2d-PARALLELISM-COLLAPSE-GUARD）。
- **T2d-8 1×1 卷积纯 GEMM 快路径**：K=1、stride=1、无 pad、无 dilation、groups=1 时卷积严格等价于「(输出通道 × 输入通道) 权重 × (输入通道 × 空间展平) 输入」的标准 GEMM——输入零拷贝只改视角，按标准 GEMM 分块（空间维作 N 维），无任何卷积专用逻辑。实测该子集 **1.87–3.03x**。
- **T2d-9 大 kc 物化 + 分块 GEMM 的 profiling 双门控**：规约维极大（kc ≥2000）**且** GEMM 规模足够（kc × 输出元素 ≥20 万）时，物化 im2col 矩阵（规约维零填充到 128 倍数）再走「K 分块 × 空间 N 维」大 GEMM，优于逐窗口 tap dot——大 K 下 Cube 效率收益超过物化的额外访存往返。两个门控缺一不可：kc 达标但 GEMM 规模不足的 case 物化开销无法回收（实测 0.60x）；kc 不达标时 0.41–0.79x 劣化。省一次清零 launch 的技巧：物化缓冲用未初始化分配，规约尾块的垃圾 lane 由权重侧零填充对冲（垃圾 × 0 = 0）。
- **T2d-10 权重单级缓存 + 派生布局同条目**：按完整卷积配置键建立实例级缓存，一次查表取出该配置的全部派生物——主路径权重布局、逐窗口回退布局、零填充 GEMM 布局（规约维补到 128 倍数）、偏置、每组输入/输出通道数、kc；forward 内不再做任何权重变换。缓存随实例销毁，天然规避跨实例污染（若改用模块级缓存必须在使用前清空，否则序列相关 AccuracyError，C2d-WEIGHT-CACHE-INSTANCE-RESET）。
- **T2d-11 S=2 pad+解交错单步直写 kernel（pair-load + reshape+split）**：C<16 的 S=2 skinny case，把「补零 pad」与「奇偶平面拆分」合并为单 kernel：`make_block_ptr` 一次连续 `(BLOCK_R, 2·BLOCK_J)` pair load → `reshape (R, J, 2)` → `tl.split` → 两条 unit-stride store 直写目标平面布局。实测 **7.1µs**，替代「pad_input + pad_deinterleave 两步链」173µs（**24x**）。行解码用 `program_id(0)=channel`（免向量除法，C2d-NO-VECTOR-INTDIV）；写侧行号须加 pad 偏移 `nc·H_alloc + h + P`（目标 buffer 是四周 pad 布局，漏 `+P` 即 max_abs ~4 量级错）；W 相位常量化 `p0=P&1, j0=m+P//2`。
- **T2d-12 分派门槛数据驱动校准**：多路径分派的门槛值（如 P6 的 `Wo`、P5 的规约块 `BCI`）不是语义边界而是性能手段，须用 quickbench A/B 逐 case 强制分派对照校准。已校准结论：P5 `BCI` 由写死 16 改为 `pow2_ceil(cin_pg, 16, 64)` 自适应（cin=32/64 的 case 每 tap 从 2/4 次窄 dot 降为 1 次）；P6 门槛 `Wo>=20`（原 32，放宽覆盖 +4 case）。反向校准同样必要：P5 BCO 上限 64→128 无效（噪声带内）。
- **T2d-13 msprof simulator 终局诊断（瓶颈类型实证）**：fp32 ieee 直接卷积**不是** Cube 计算瓶颈——5 个瓶颈 kernel 实测 CUBE pipe 仅 12–30%，真因 **MTE 搬运 53–69%**（P6 plane_dot 有效带宽 ~310GB/s ≈ 910B 峰值 78%）+ FLOWCTRL 34%（fused 路径 latency-bound）。环境配方：必须 `export ASCEND_OPP_PATH=$CANN_HOME/opp`（否则 aclnn ZerosLike 报 EZ1013、ParseDynamicKernels 失败无 CSV）；被采集脚本 monkey-patch `torch.zeros/empty` 为 CPU 构造 + `.to('npu')`（H2D copy 可仿真，aclnn ZerosLike 不可）；`code_exe.csv` 行号可能全为 `internal:0`，诊断以 `instr_exe.csv` pipe 占比为准。在宣称「fp32 dot 硬件极限」前必须先采 pipe 占比。
- **T2d-14 小 HW 1×1 GEMM 的 CO tile 放大（BLOCK_CO 16→64）**：P1 1×1-GEMM 路径输出通道 tile 由 16 放大到 64（受益 case 9 kernel -35%/加速比 +52.7%、case 8 +1.4%）。机理：HW 小（如 196）时 sp_blocks 切多块，原 BLOCK_CO=16 导致 co_blocks 也多块，`w_tile`（权重）被每个 sp_block 重复加载 co_blocks 次（dot M=16 仅半个 Cube 微块）；放大后 co_blocks=1，权重只读一遍 + dot M=64 吃满 Cube 微块。适用条件：小 HW 1×1 卷积/GEMM、cin≤128（BLOCK_K 上限）、cout 越接近 64 倍数收益越大（cout<64 有 padding 浪费但不劣）。警示：timeit 冒烟（含 dispatch+pack ~0.09ms 固定开销）对该收益完全失明，判定一律以全量 benchmark profiler 为准（方差方法论见 §8.1 #12）。
- **T2d-15 pad/拷贝 kernel 2D 平面 tile 提带宽（48→353GB/s）**：`pad_input_kernel` 按**行**（1KB tile）搬时实测仅 48GB/s，占 (pad+conv) ≥50% 的 case 有 11/50。重写为 2D 平面 tile：`BLOCK_W = pow2_ceil(W, 64, 1024)`、`BLOCK_R = 4096//BLOCK_W`（~16KB tile），grid=(N*C, r_blocks 分核)，stride 循环 `for tid in range(pid, r_blocks, NUM_CORES)`。带宽 48→**353GB/s**（case 28 pad 135.7→18.7µs），套件 geomean **+20.79%**（case 29 +100%、case 2 +89%）。**读侧必须 ptr-arith + masked load，双侧都不可用 make_block_ptr**：dim0 基址 = `nc*H_IN`，奇 H_IN（如 7）时基址奇 → load 间歇取错数据（C2d-ODD-DIM0-BASE-UNRELIABLE）；pad kernel 无 dot，ptr-arith 安全且零性能代价。**「消除 pad pass 改 mask 读原图」在本后端不可行**：dot 输入必须 make_block_ptr（C2d-SPLIT-RESULT-MATERIALIZE），W 边界行折叠 mask 无法用 block_ptr 表达；P2 depthwise mask 化则收益反转（pad 提速后只剩 3-11µs launch 地板，mask 使 conv 本体 +8~99%）。pad 降本唯一可行路线 = 提 pad kernel 自身带宽。

### §6.4 性能基准与残余瓶颈

- **最终基准（7 路分派 + 2D tile pad，§6.2 全表 + T2d-15）**：50/50 verify pass，geomean **0.6677x** vs torch aclnnConvolution（实现平均延迟 0.0411ms / 框架平均 0.0274ms / 50 case 实现总延迟 2.8475ms；target 0.8x，**未达**）。复现本数字的代码 = 工作目录 `7_ConvStandard2d_generated.py`（7 路分派 P1 1×1-GEMM / P2 depthwise / P3 skinny-chlast / P4 S2-plane+fused / P5 tap-window / P6 plane-dot / P7 im2col+kblock-GEMM + 2D tile pad kernel）。**⚠️ 本数字是 7 路分派完整实现的产物，缺失任意路径即偏离（§8.1 / C2d-SKETCH-MANDATORY-PATHS）。**
- **残余瓶颈（msprof 终局诊断实证，T2d-13）**：MTE 搬运主导 + 有效带宽已达 910B 峰值 ~78%，优化点 7/21/10/22 可行变体穷尽（9-tap 权重 UB 常驻 147KB+acc 超 192KB；tf32/bf16 被 MERE 1.221e-4 容限禁用，实测 tf32 相对误差 ~5e-4 超 4 倍）；小 shape 的 host 固定开销地板（40–60µs 无法摊薄）+ launch/发射开销主导几何平均值。非单点 kernel 效率问题，属该套件（小 shape 为主 vs 厂商闭源 aclnn）的**结构性天花板**。

### §7.5 常见陷阱

| 陷阱 | 症状 | 避免方法 |
|------|------|---------|
| dot 路径 `BLOCK_M > Cout_pg`（Cube 下限 16 补齐）未按组裁剪 | 越组 co 行用本组输入计算并写穿到下一组输出通道 | store/bias mask 加 `(pid_co*BLOCK_M + arange) < Cout_pg`（C2d-BM-GROUP-CLIP） |
| 模块级权重/布局缓存跨 ModelNew 实例复用 | **序列相关** AccuracyError（部分 case 依赖运行顺序才失败） | 缓存放实例内字典；若必须模块级，`ModelNew.__init__` 清空（C2d-WEIGHT-CACHE-INSTANCE-RESET） |
| host 侧用 `min()/max()` 裁 grid / 计算块数 | `validate_triton_impl.py` 判 Type3 退化（PyTorchFallback） | 条件表达式实现 `_clip`（C2d-NO-MIN-MAX） |
| stride>1 输入用 computed-offset gather 直喂 `tl.dot` | Cube mis-lower，精度错 | 三选一：物化 im2col / NCHWc C0 内层 pack / gather 进 UB tile 再单 dot（C2d-DOT-OPERAND-CONTIG） |
| 小通道（每组输入通道 ≤6）仍走逐窗口 tap dot 路径 | K² 次 K=16 小 dot（仅少数行有效），矩阵乘发射开销主导 | 门控切 skinny channel-last 单 dot 路径（T2d-7） |
| im2col 规约顺序与 weight 展平不一致 | GEMM 数值全错（看似精度问题） | 统一规约序 =（kh×K+kw）×每组输入通道 + 通道序号（tap 外层、通道内层），weight 展平同序 |
| stride≥2 向量化输入 load（masked/unmasked、块指针/计算指针任一写法） | aicore 507015 崩溃或数值错 | host 奇偶平面拆分，平面内 stride-1 仿射（T2d-2，C2d-S2-VECTOR-LOAD-FORBIDDEN） |
| 钳位（min/max）进入 load 地址 | 仿射性破坏 → load 标量化，10–70x 回退 | host 预零填充消除边界（T2d-1）；kernel 内仅用「钳位保地址 + mask 置零」防崩溃（C2d-NO-CLAMP-IN-LOAD-ADDR） |
| host 矩形切片拷贝 / stride 视图拷贝 / 通道重排 | 落 AiCpu 慢速路径，单次 92–176µs，占端到端 70%+ | 行连续 load/store 专用拷贝核，多段预处理融合单核；仅病态形状启用（T2d-5，C2d-HOST-COPY-PATHOLOGY） |
| 运行期标量 if 双 load 快路径 | 误编译，dot 输出全零 | 删除条件双路径；选择逻辑折成「序号 × 步长」仿射偏移加单基址（C2d-NO-CONDITIONAL-BASE-POINTER） |
| 编译期 if/else 切换双基址指针 | 误编译，误差 ~125 量级 | 同上：单基址 + 仿射平面偏移 |
| 多段式「逐 tap 写入统一缓冲区组装」链嵌在 runtime 循环内 | MLIR 编译错误（cbuf to cbuf） | 退化为逐平面独立 dot；或 host 侧重排使组装坐标仿射（T2d-7） |
| 行块化/大 tile 后 tile 总数 < 核数 | 并行度塌陷（实测 0.83x 回退） | tile 总数 ≥ 核数量级校验，不足回退小块（C2d-PARALLELISM-COLLAPSE-GUARD） |
| 小算力 case 走带 host 预处理路径 | 固定开销 40–60µs 无法摊薄，0.67–0.75x | forward 按乘加量门控回退无预处理路径（C2d-MACS-GATE-HOST-OVERHEAD） |
| im2col+GEMM 用于 kc<2000 或 GEMM 规模小的 case | 物化开销无法回收，0.41–0.79x | 双门控 kc≥2000 且 kc×输出元素≥20万（T2d-9） |
| ptr-arith / masked load / `tl.where` 物化 / clamp 地址直接喂 `tl.dot` | 编译期 SIGSEGV（ConvertLinalgRToBinary） | dot 输入唯一合法形态 = make_block_ptr（含经 `tl.where` 物化的稳定变体，C2d-SPLIT-RESULT-MATERIALIZE） |
| make_block_ptr 读 `nc*H_IN` 为奇基址的输入（pad/拷贝 kernel） | 间歇取错数据（H_IN 奇如 7 的 case） | 无 dot 的 kernel 一律 ptr-arith + masked load（零性能代价，C2d-ODD-DIM0-BASE-UNRELIABLE） |
| pad/拷贝 kernel 按行（1KB tile）搬运 | 带宽仅 48GB/s（HBM 峰值 12%），pad 可占 pad+conv 的 50%+ | 2D 平面 tile ~16KB（`BLOCK_W=pow2_ceil(W,64,1024)`、`BLOCK_R=4096//BLOCK_W`）→ 353GB/s（T2d-15） |
| sketch.txt 仅规划 P1/P2/P5 三条路径，其余 case 走通用 fallback | Phase 3 3.2b 架构检查报 `A-SketchDeviation`；若强行绕过，非 1×1 case 全走 im2col，geomean 跌至 0.025x | sketch 必须一次性列出 §6.2 全部 P1–P7 路径、kernel 列表、门控与数据布局（C2d-SKETCH-MANDATORY-PATHS） |
| P2 depthwise host 预补零后 kernel 内仍 `-PAD_TOP/-PAD_LEFT` | `padding>0` 的 depthwise/grouped-like case 全部 AccuracyError（impl 输出接近 0，实测 10/50 fail） | host 补零后 kernel 内坐标直接用 `ih0=oh0*STRIDE+kh*DILATION`、`iw0=ow0*STRIDE+kw*DILATION`（C2d-DEPTHWISE-PAD-OFFSET） |
| 无门控地把所有非 1×1 case fallback 到 im2col+GEMM | `im2col_kernel` 标量 gather 占 95%+，geomean 仅 0.025x | P7 im2col 仅在 `kc≥2000` 且 `kc×输出元素≥20万` 时启用，其余走 P3/P4/P5/P6（C2d-IM2COL-GATE） |

### §8.1 已验证无效方向 / 证伪（ConvStandard2d，50-case 套件实测，最终 0.6677x）

1. **stride-2 向量化 load（所有写法）**：masked/unmasked、块指针/计算指针均触发 aicore 507015 崩溃或误编译——唯一出路是奇偶平面拆分（T2d-2）。
2. **钳位进 load 地址**：仿射性破坏 → 标量化 10–70x 回退；「钳位保地址 + mask 置零」只可作 S=2 防崩溃手段，不可当性能手段。
3. **运行期标量 if 双 load 快路径（内部 tile 无 mask）**：误编译，dot 输出全零。
4. **编译期 if/else 双基址指针**：误编译，误差 ~125 量级。
5. **多段式统一缓冲区组装链嵌 runtime 循环**：MLIR cbuf-to-cbuf 编译错误；多平面组装退化为逐平面独立 dot。
6. **stride-2 gather（expand_shape 视图方案）**：MLIR expand_shape 错误，方向废弃。
7. **行×宽×规约 3D 单 load 的 skinny 变体**：编译失败（at load）。
8. **im2col+GEMM 用于小 kc（<2000）**：物化开销无法回收，0.41–0.79x 劣化；kc 达标但 kc×输出元素 <20万（GEMM 规模小）同样劣化（0.60x）——双门控缺一不可（T2d-9）。
9. **拷贝核盲目替换一切 host 拷贝**：普通形状 host 拷贝 38–60µs 已快于拷贝核 ~41µs 地板，替换反而回退（0.16x case 实测）——仅「通道≥128 且输入宽≤32」病态形状启用（T2d-5）。
10. **skinny 行块 4 用于 K=7**：4 份累加器 ~64KB 寄存器压力，无收益（231≈228µs），K=7 保留行块 2。
11. **P4 fused 去掉 split→where 物化链（平面仿射偏移）**：拆成两个半宽 `(BLOCK_CI,BLOCK_W)` block_ptr 直读喂 dot 虽 err=0 全对，但 A/B 交替 3 轮取 min：row41 -21.3%、row2 -19.1%——split/where 是廉价 Vector 物化，pair 宽载 `(BLOCK_CI, 2·BLOCK_W)` 的 MTE 连续性才是收益来源（与 simulator MTE 53-69% 诊断一致）。**结论：C2d-SPLIT-RESULT-MATERIALIZE 的 where 物化链保留，勿再尝试平面化消除。**
12. **P5/P6 门控与 tile 上限的盲目调参**：P6 Wo 门槛再放宽、P5 BCO 上限 64→128 均在 ±0.1% 噪声带内无效。**方法论要点**：benchmark 单 case run-to-run 方差实测 ±4%（与 base 完全同路径的 case 也会显示 -3.6%~-4.1% “差异”）——任何 <4% 的单 case 改进在 50-case 套件不可分辨，勿据单轮 profiler 差异下结论，须同路径对照或复跑。
13. **行块化后不校验 tile 总数**：tile 数 8 < 24 核时并行度塌陷，0.83x 回退（C2d-PARALLELISM-COLLAPSE-GUARD）。
14. **小算力 case 走带 host 预处理路径**：固定开销无法摊薄，0.67–0.75x（C2d-MACS-GATE-HOST-OVERHEAD）。
15. **load 传 care_padding=False**：API 不存在，试误回退。
16. **移位 1×1 分解（K² 个「移位的纯通道规约」小 GEMM）作为 stride==1 主力**：并非报错，而是被证明次优——小 dot 发射开销主导，已被 skinny channel-last 单 dot（T2d-7）+ 零填充逐平面无 mask 大 dot（T2d-1）取代。
17. **NCHWc（通道 C0=16 内包）重排用于 stride>1**：其输入 load 涉及 stride-2 向量化，触发误编译（见第 1 条）；已被 W 奇偶平面拆分（T2d-2）取代。
18. **「conv kernel mask 读原图消除独立 pad pass」（两条子路线均证伪）**：(a) mask/ptr-arith/`tl.where` 物化/clamp 地址任何一种喂 `tl.dot` 都在编译期 SIGSEGV（ConvertLinalgRToBinary，C2d-SPLIT-RESULT-MATERIALIZE 强化），而 W 边界行折叠 mask 无法用 make_block_ptr 表达 → dot 路径（P4/P5/P6/P7）消除 pad 不可实现；(b) P2 depthwise mask 化可行但收益反转——pad 2D tile 化（T2d-15）后 pad 只剩 3-11µs launch 地板，mask 使 conv 本体 +8~99%（case 44 净亏 44%）。**pad 降本唯一可行路线 = pad kernel 自身提带宽（T2d-15）**。
