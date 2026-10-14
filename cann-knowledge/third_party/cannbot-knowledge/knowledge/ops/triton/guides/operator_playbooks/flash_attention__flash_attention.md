---
type: Guide
title: Flash Attention 实现手册
description: FA 形态的设计约束、算法骨架与关键技巧；含与朴素 attention 的差异声明。
tags: [task.implementation, topic.attention, topic.compute.normalization, topic.compute.composite, topic.data_movement.layout_transform, topic.tiling]
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
# Flash Attention 实现手册

> 类别约束见 [flash_attention_constraints.md](../constraints/flash_attention_constraints.md)。

> 类别约束见 [flash_attention_constraints.md](../constraints/flash_attention_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

> **本节共 16 条（L1.1 ~ L1.16），全部是 Phase 2 的硬性边界。**
> Phase 2 Step 1 产出 `precheck.json` 时**必须逐条载入全部 16 条**，不得只取前 N 条或按篇幅截断——
> 实测出现过只载入到 L1.12、把 L1.13（BLOCK 面积预算）漏掉，导致 `BLOCK` 保持硬编码、
> 性能与不带该约束时完全相同的情况。
> 带 ★★★ 的四条（**L1.12 投影权重布局**、**L1.13 BLOCK 面积预算**、**L1.15 输入变换外提**）是收益最大的几笔，尤其不能漏。

### L1.1 ★ `tl.dot` 契约：原生 dtype 操作数 + fp32 累加器；**fp32 输入同样必须用 `tl.dot`**

- **必须** `tl.dot(q, k_t)`，两操作数**保持原生 dtype**，累加器 fp32（`tl.dot` 默认 `out_dtype=tl.float32`）。
- **禁止** 把操作数升到 fp32 再 dot（`tl.dot(q.to(fp32), k.to(fp32))`）——那会走异构路径，是"fp32 禁用 `tl.dot`"这条流传约束的**真正来源**。
- **禁止** 用逐维标量循环（`for d in range(D): scores += q[:,None]*k[None,:]`）或 `tl.where(d_offs==d, ...)` 逐维散写 `acc` 代替矩阵乘。
- **Why**：逐维外积让 Cube 完全空转（`aic_mac_ratio` 7.5%），实测 benchmark 直接超时无有效数据；换 `tl.dot` 后**不可测量 → 2.3554**，是全链路最大的一笔。fp32 输入下精度实测 **50/50 全过**。
- **判别信号**：`aic_mac_ratio < 10%` 且代码里有 `for d in range(head_dim)` ⇒ 立即重写，不用扫参。

### L1.2 `tl.dot` 两操作数必须显式 cast 到同一 dtype

- **必须** 两侧都 `.to(<统一 dtype>)`——`tl.load(..., other=0.0)` 会把结果推成 fp32。
- **Why**：否则 `semantic.dot` 断言 `Both operands must be same dtype. Got fp32 and fp16`。实测该错误导致 33/50 个非 fp32 case 全部编译失败。

### L1.3 online softmax 必须同步缩放 `acc` 与 `l`

- **必须** 每轮 KV tile：`m_new = max(m, rowmax(scores))`；`alpha = exp(m - m_new)`；`l = l*alpha + sum(p)`；`acc = acc*alpha + tl.dot(p, v)`；`m = m_new`。
- **禁止** 漏掉 `acc *= alpha`（结果偏大）或漏掉 `m = m_new`（下一轮基准错）。
- `m` 用**有限极小值**初始化（`-1e30`），不用 `-inf`（见 F2）。

### L1.4 ★ 精度契约先钉死，再谈性能

- **必须** 在写 kernel 前先确认**参考实现的算术路径**：attention 是写作 `q.float() @ k.float().T`（fp32 归约），还是在**原生 dtype** 上算。
- **两套契约不可混用**：
  - 参考在 fp32 上算 ⇒ `scores`/`p`/`acc`/`l` 全程 fp32，最后 cast 回输出 dtype；
  - 参考在原生 dtype 上算 ⇒ **必须逐位复刻低精度路径**，"算得更准"是错的。四条逐位契约见 `@../../../ops/triton-precision-debug/references/attention-lowprec-contract.md`。
- **Why**：某算子上游一次提交把参考从 fp32 改回原生 dtype，旧契约下 3.6766 的交付版对新参考**只过 20/50**。**参考的算术路径一变，精度类结论全部作废、性能类结论也要重测。**

### L1.5 `p`（softmax 输出）不能直接降到输入 dtype

- **禁止** `tl.dot(p.to(in_dtype), v)`：`p` 是真 fp32 量，降到 fp16 相对误差 ~5e-4，顶穿 fp16 阈值 9.77e-4。实测 27/50 越阈，**失败集合严格等于全部 fp16/bf16 case**。
- **可选替代**（仅低精度输入）：二段拆分 ~22 bit，见 §4.6。**fp32 输入不要开。**

### L1.6 host 端不做 permute / contiguous / pad

- **必须** 直接把原始布局的 stride 传进 kernel（最内轴 D 本来就连续），输出按原布局分配；D 的补齐用 `BLOCK_D=ceil16(D)` 的 masked load（`other=0.0`）。
- **禁止** `q.permute(...).contiguous()`、`torch.zeros`+`copy_` 补齐、输出再 permute 回来。
- **Why**：实测每次调用多出 ~4 个纯搬移 kernel（`Transpose` 占 14.39%），删掉后 **+16%（保守值）**，小 shape 从 2~3x 直接跳到 10~14x。
- **判别**：profiling 时间归属里只要出现 `Transpose`/`ViewCopy`/`ZerosLike`/`Slice`，直接删，不用扫参。

### L1.7 mask 属性作为 `tl.constexpr` 传入，禁止运行时标量分支

- **必须** `HAS_UPPER` / `HAS_LOWER` / `HAS_SOFTCAP` / `IS_CAUSAL` 以 `tl.constexpr` 传入，让编译器把不适用的整段 DCE 掉。
- **禁止** 把 `causal` / `window_left` / `window_right` / `softcap` 作为运行时标量在 kernel 内构造多个 `[BQ,BKV]` 比较再 `or`。
- **Why**：实测 **+21%**。代价是每种组合一个特化版本（~8 种），编译只发生一次且不计入 benchmark。

### L1.8 `kv_lo` **不做** `BLOCK_KV` 对齐

- **禁止** `kv_lo = ... // BLOCK_KV * BLOCK_KV`。
- **Why**：扫描区间常只有 1~3 个 block，对齐平均多出 `BKV/2` 个 key ⇒ **凭空多一整次迭代**，每次迭代要付 8 个跨核握手。实测与 tile 放大**成对**做时 +21%，单独放大 tile 只有 +1.2%。

### L1.9 Grid 收缩到 `min(核数, tasks)` + 核内步长循环，且 grid 与步长必须一致

```python
# host
NUM_CORES = 24                                            # 必须是常量 constexpr
total_tasks = B * H * cdiv(S_Q, BLOCK_Q)
grid = (min(NUM_CORES, total_tasks),)                     # ✅ 两头都对
# kernel
for task in range(tl.program_id(0), total_tasks, NUM_PROG): ...
```

三种写法的实测差异：`grid=(total_tasks,)` 在 `total > 核数` 时**任务被重复计算**（实测某 case 的 task 被 3 个 pid 各算一遍）；
`grid=(核数,)` 在 `total < 核数` 时小 shape 白付启动开销；只有 `min()` 两头都对，实测 **+5.8%**。

- ⚠️ `NUM_CORES` 必须保持**常量 `constexpr`**——做成随 case 变化的值会触发编译爆炸（实测 verify 超时、0 个用例启动）。`grid` 是启动参数，随 case 变化无妨。
- ⚠️ 任何改变任务空间的改动（如压平循环维）**必须同步改 grid**，否则表现为**静默算少 → AccuracyError 而非报错**。

- **Why**：原始 grid 大 shape 下可达数百，每个 program 都要付一遍循环外的同步收尾。收益小（+0.5%），但属基础写法。
- **禁止** grid 超订（如 2 倍核数）：MIX kernel 超订不会带来同步延迟的重叠，实测无收益。

#### ★ 本条适用于算子内的**每一个** `@triton.jit` kernel，不只 attention 主 kernel

**禁止**把辅助 kernel（RoPE / 位置编码 / bias 预处理 / layout 变换 / 归一化等）的 grid
按元素数或行数**直接展开**——`grid = (B*N*S,)` 这类写法在大 shape 上会撞硬上限：

```
RuntimeError: grid should be less than 65536!
You can try "export TRITON_ALL_BLOCKS_PARALLEL=1" to avoid this problem.
```

**必须**所有 kernel 统一用同一套写法：

```python
total_tasks = <该 kernel 的任务数>                  # 如 B*N*S 行、或 B*N*cdiv(S,BLOCK)
grid = (min(NUM_CORES, total_tasks),)              # ✅ 永远不会越界
# kernel 内
for task in range(tl.program_id(0), total_tasks, NUM_PROG): ...
```

- **判别信号**：失败集合**严格按 `B*N*S`（或该 kernel 的 grid 维）是否 ≥ 65536 划分**，
  且报的是 `RuntimeError` 而**不是**精度不匹配 ⇒ 直接查 grid，不要去查 dtype 或数值路径。
- **实测**：某带 RoPE 的 FA 算子 50 个 case 里失败 8 个，全部满足 `B*N*S ≥ 65536`
  （65536 / 131072 / 262144）；attention 主 kernel 遵守了本条，而另开的 `rope_kernel` 按 `B*N*S` 展开 grid，是唯一的失败源。
- ⚠️ **不要用 `export TRITON_ALL_BLOCKS_PARALLEL=1` 绕过**：那只是放开上限，grid 远超核数仍会带来调度开销，与 grid 收缩初衷相反。

### L1.10 张量的运行时索引与元素赋值均不支持

- **禁止** `q_tile[:, d]`（`d` 为循环变量）→ `ValueError: unsupported tensor index: int32[]`；**禁止** `acc[:, d] = x`。
- **禁止** 用运行期变量做 `tl.arange` 的参数 → `ValueError: arange's arguments must be of type tl.constexpr`。
  由运行期 shape 推出的值（如 `BLOCK_D // 2`、`ceil16(D) // 2`）**必须先声明为 `tl.constexpr` 参数从 host 传入**。
- 需要按维度取向量时直接从 GM 按 `d * stride` load。（若已按 L1.1 用 `tl.dot`，通常不会遇到这条。）

#### ★ 交错重排（rotate-half / interleaved RoPE）的可执行配方

参考实现里 `torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)` 这类**偶/奇元素交错**，
在 Triton 里**不要**用「切出偶/奇两半 → reshape 回去」的写法——那条路上必然要用运行期值构造 `tl.arange` 或做动态 `tl.reshape`，直接编译失败。

**正解：用 constexpr 的静态索引 + `tl.where` 就地选择，全程不改形状**：

```python
# host 侧：BLOCK_D 必须是 tl.constexpr
offs_d  = tl.arange(0, BLOCK_D)                 # ✅ 参数是 constexpr
is_odd  = (offs_d % 2) == 1                     # 静态奇偶掩码
partner = tl.where(is_odd, offs_d - 1, offs_d + 1)   # 配对元素的下标
x       = tl.load(x_ptr + row * stride + offs_d, mask=offs_d < D, other=0.0)
x_pair  = tl.load(x_ptr + row * stride + partner, mask=partner < D, other=0.0)
x_rot   = tl.where(is_odd, x_pair, -x_pair)     # 奇位取前一个、偶位取后一个并取负
out     = x * cos + x_rot * sin                 # 形状自始至终是 [.., BLOCK_D]
```

- **判别信号**：`CompilationError` + `arange's arguments must be of type tl.constexpr`，
  且**失败集合严格按某个布尔属性划分**（如"全部且仅仅是 `use_rope=True` 的 case"）
  ⇒ 直接查该分支里的 `tl.arange` 参数与动态 `tl.reshape`，**不要去查 shape/dtype/数值路径**。
- **实测**：某带 RoPE 的 FA 算子 50 个 case 挂 25 个，全部是 `use_rope=True` 的奇数号 case，连续 4 轮迭代都没自己定位到。
- 同族约束：位置编码的 `cos`/`sin` 表若在 kernel 内生成，其索引同样只能用 constexpr `tl.arange`；
  表本身可在 host 侧算好传入（与 L1.6「host 端不做 permute/pad」不冲突——那禁的是**数据布局搬运**，不是**常量表预计算**）。

### L1.11 `fa-mha` 专属：先对齐 host 侧语义，再写 kernel

含投影 GEMM 的算子，最致命的两条缺陷**不在 kernel 里**，而在 host 侧那二十行胶水代码，且二者各自独立致命、互相掩盖——修好任意一条 `passed_cases` 仍为 0，迭代循环拿不到梯度信号。

**必须按此顺序**：① 权重与参考逐位一致 → ② `y = x @ Wᵀ` 语义正确 → ③ 再写 attention kernel。

| 位置 | 要求 |
|------|------|
| 权重初始化 | **CPU** 上、**先 fp32 采样再 `.to(dtype)`**、构造顺序与次数一致、rng save/restore 一致。四处缺一即全错 |
| 投影 GEMM | `nn.Linear.weight` 是 `[out, in]`，`y = x @ Wᵀ`。方阵下 shape 检查会放行错误写法 ⇒ 交换 stride：`w.stride(1), w.stride(0)` |
| dtype 时序 | 不要直接以 fp16/bf16 采样权重 |

### L1.12 ★★★ `fa-mha` 专属：投影权重**禁止**以 `nn.Linear` 原始 `[out, in]` 布局喂给 GEMM kernel

**这是含投影 GEMM 形态的头号杠杆，首次生成就必须做对，不是 Phase 4 的可选优化。**

- **禁止** 把 `nn.Linear.weight`（`[out_features, in_features]`）原样传进 GEMM kernel 并用 `w.stride(0), w.stride(1)`。
- **必须** 在 `__init__` / 权重准备阶段做 `w.t().contiguous()`（或等价的预重排），kernel 侧 stride 实参相应改为 `w.stride(1), w.stride(0)`。**一次性 host 开销，数值逐位不变。**

```python
# ❌ b tile 的最内轴 stride = in_features ⇒ 离散 gather，无法向量化访存
#    kernel 注释形如 "y = x @ w^T，w 形状 [N, K]（nn.Linear.weight 布局）" 即是此坏味道
proj_kernel[grid](x, w_q, out, ..., w_q.stride(0), w_q.stride(1), ...)

# ✅ 预转置成 [in, out] 连续
self.w_q_t = w_q.t().contiguous()          # host 侧一次性
proj_kernel[grid](x, self.w_q_t, out, ..., self.w_q_t.stride(0), self.w_q_t.stride(1), ...)
```

- **Why（实测）**：同类算子上 `linear_kernel` 6469.7us → 35.6us（**182x**），端到端 0.1052 → 0.6253（**5.94x**），是该算子全程最大的单项收益，且**零精度代价**。
- **判别信号**：profiling 里 `aic_mte2_ratio > 60%`；或扫 tile 时出现"缩小 `BLOCK_N` 反而更快"——那是在绕开离散访存，不是真优化。
- **★ 配套硬约束（与转置同等重要，首次生成就要做对）：布局一改，投影 GEMM 的分块必须重扫**

  **禁止** 沿用 `(BLOCK_M, BLOCK_N, BLOCK_K) = (64, 64, 32)` 这类小分块。
  **必须** 让 a tile 沿 K 的连续字节数 ≥ **512B**：

  ```python
  # a tile 沿 K 的字节数 = BLOCK_K * elem_size，必须 >= 512
  BLOCK_K = max(256, 512 // elem_size)      # fp16: 256 ⇒ 512B ✅   32 ⇒ 64B ❌（对齐线的 1/8）
  BLOCK_M, BLOCK_N = 128, 256               # A、B 均不转置一档的经验值
  # 若 UB 编译失败再逐级减半；并按 L1.13 同样的思路做并行度保护（块数 < 核数/2 就缩 tile）
  ```

  - **Why**：权重转置把算子从"A 不转置、B 转置"变成"**A、B 都不转置**"，**推荐分块随之改变**。
    沿用旧分块会让转置的收益拿不满——同类算子上补这一步再拿 **+13.2%**，
    `MAC` 从 17.9% 升到 40.3%（由**访存受限**转为**算力受限**）。
  - **判别**：profiling 里投影 kernel 的 `aic_mte2_ratio` 仍然很高（实测 94.9%）⇒ 分块没跟上布局。
    该洞在同类算子上**挂了二十多轮**才被发现，因为转置本身已带来数量级收益、容易误以为"这条已经做完了"。
- **进阶（可选，+5.0%）**：在转置基础上再按 tile 形状重排成 `[NBN, K_pad, BLOCK_N]`，见 §4.4。**但两级里第一级才是数量级收益，不做第一级直接做第二级没有意义。**

### L1.13 ★★★ `BLOCK_Q` / `BLOCK_KV` 必须按面积预算取到上限，**禁止硬编码小值**

> ★ **面积预算是 16384 元素（64KB fp32），不是 8192。**
> 早期版本写 8192 是因为**输入变换还留在 KV 循环里占着 UB**；按 **L1.15** 把 RoPE 等
> 变换外提之后，UB 腾出来，`128×128` 的 score tile 可以正常编译并跑通。
> op54 实测：仅把这一处 `8192 → 16384`（`BLOCK_Q×BLOCK_KV` 由 64×128 变 128×128），
> 几何平均 **1.7264 → 2.1048**，精度仍 50/50，`<1.0x` 的用例从 16 个降到 6 个。
> ⇒ **L1.13 与 L1.15 是耦合的**：没做外提就放大 BLOCK 会 `MLIRCompilationError`（实测），
> 做了外提就必须把预算提到 16384，否则白白浪费一半 UB。

**这是本类算子最大的一笔收益，且必须在首次生成就做对——它决定 KV 迭代总数，而迭代数就是代价本身。**

- **禁止** 写死 `BLOCK_Q = 32` / `BLOCK_KV = 64` 这类小常量。实测 `32×64` 的 tile 面积只有预算上限的 **1/4**，
  直接意味着约 **4 倍**的 KV 迭代数与 4 倍的跨核握手。
- **必须** 按下式由 host 侧算出，取到面积预算允许的最大值：

```python
BLOCK_D  = ceil16(head_dim)
BLOCK_KV = min(128, ceil16(S_K))                      # 尽量大，但不超过实际 key 数
BLOCK_Q  = 满足 BLOCK_Q * BLOCK_KV <= 16384 的最大 2 的幂  # scores tile fp32 = 64KB（见上方★）
BLOCK_Q  = min(BLOCK_Q, max(32, ceil32(S_Q)))         # 不超过实际 query 数
while BLOCK_Q * BLOCK_D * elem_size > 64 * 1024:      # acc tile 约束
    BLOCK_Q //= 2
```

- **Why（实测）**：迭代总数 = `B*H*ceil(S/BLOCK_Q) × ceil(span/BLOCK_KV)`，两个 BLOCK 同时翻倍 ⇒ 迭代数变 **1/4**，
  整套 `pipe_barrier` 直接少一半以上。同类算子实测：

| SQ/SKV | 64/64 | 128/64 | 64/128 | **128/128** | 256/* |
|---|---|---|---|---|---|
| 子集几何平均 | 0.697 | 1.129 | 1.155 | **1.42** | 编译失败 |

- **面积上限 16384 是算出来的不是拍脑袋**：`scores` tile 是 `[BQ, BKV]` fp32 = 64KB；再叠加 `acc [BQ, BLOCK_D]` fp32。早期 8192 是输入变换未外提时的旧值（见上方★）。
- **放大到编译失败为止**：UB 上限由编译器直接报数字（`ub overflow, requires N bits while 1572864 bits available`），
  看错误信息就知道差多少。**上限不可用简单阈值预测**——同为 `128×128`，D=96 编得过、D=128 失败、D=160 反而编得过且快 22%。
- ⚠️ **小 shape 要按实际尺寸收缩**（`min(..., ceil16(S_K))` / `min(..., ceil32(S_Q))`）：固定大 tile 在 `S < BLOCK` 时纯空转，
  实测某算子 27/50 个 case 中招、最小 case 空转 85 倍。**"开到上限"与"按 shape 收缩"是同一条规则的两面。**

### L1.14 ★ `tl.maximum` 在多 KV 块路径上必须用 `propagate_nan=ALL`（单块路径必须关）

- **必须** host 侧算出 `MULTI_BLOCK = (max_kvblk > 1)` 作为 `tl.constexpr` 传入，按它分档：

```python
if MULTI_BLOCK:
    m_new = tl.maximum(m, rowmax, propagate_nan=tl.PropagateNan.ALL)
else:
    m_new = tl.maximum(m, rowmax)
```

- **Why**：默认 `propagate_nan=NONE` 在 IR 里被展开成 **7 条向量指令**（2×isnan + vmax + 2×vsel），`ALL` 直接对应硬件单条 `vmax`。
- ⚠️ **禁止全局开启**：实测全局开是 **−9%**，按块数分档才是 +2.7%；另一算子上单块 case 退化 25~45%。
- 语义等价性依赖 F2（掩码用有限极小值、无 NaN），必须在注释里写明论证。

### L1.15 ★★★ 输入变换（RoPE / 位置编码）**必须在 attention 主循环之外**完成

- **禁止** 把 Q/K 的 RoPE、位置编码、bias 变换放进 **KV 循环内**逐块重算。
- **必须** 用**独立的 Triton kernel** 预处理 q/k（写到 workspace），attention kernel 只读取变换后的张量。
  该预处理 kernel 的 grid 同样遵守 **L1.9**（`min(核数, tasks)` + 核内步长循环）。
- **禁止** 改用 host 侧 torch 做这个变换（`torch.einsum` / `stack` / `repeat_interleave` 等）——
  会被退化检测判为主链计算落在 PyTorch 上。**它必须是 Triton kernel，只是不能在 KV 循环里。**

**实测（同一算子、只改这一处）**：

| 配置 | `[8,8,2048,64]` | `[8,8,4096,64]` | `[1,8,2048,64]` | 每次 KV 迭代成本 |
|---|---|---|---|---|
| RoPE 在 KV 循环内（❌） | 0.005x | 0.003x | 0.013x | **630 us** |
| RoPE 外提 + BLOCK 64 | 0.747x | 1.022x | 1.566x | **4.4 us** |
| RoPE 外提 + BLOCK 128（✅） | **1.386x** | **1.985x** | **2.509x** | — |

**三级因果链，缺一环都解释不通**：① K 的 RoPE 每个 KV 块重算一次（含 `partner = offs_d±1` 的元素级 gather）⇒ 每迭代 630us（正常区间 4~10us 的 60~100 倍）；② 外提后回落到 4.4us ⇒ 大 shape 快 114~144 倍；③ **外提还释放了 UB** ⇒ 原本编译不过的 `BLOCK=128` 变得可编译 ⇒ 迭代数减半 ⇒ 再快约 2 倍。

⚠️ **第 3 点是最容易漏的联动**：看到 agent 写下 `BLOCK_Q = min(BLOCK_Q, 64)  # avoid UB overflow with RoPE/ALiBi intermediates`
这类**预防性封顶**时，**不要直接删封顶**（删了确实编译失败），
而要**先把占 UB 的那部分计算移出 kernel**，再重扫 BLOCK。

- ⛔⛔⛔ **交错重排（rotate-half RoPE）禁止用「算出来的伙伴下标」做 gather load。**
  必须用 `tl.split` / `tl.join` 从**一次连续 load** 里拆偶/奇。实测 **70 倍**差距：

  ❌ 错误写法（数据相关寻址 ⇒ 退化成逐元素 gather）：`partner = tl.where(is_odd, offs_d-1, offs_d+1)` 再按 partner 做 `tl.load`。
  ✅ 正确写法（一次连续 load + `tl.split`）：
  ```python
  HD: tl.constexpr = BLOCK_D // 2
  x  = tl.load(p + offs_s[:,None]*si_s + offs_d[None,:]*si_d, mask=m, other=0.)  # 一次连续 load
  xe, xo = tl.split(tl.reshape(x, (BLOCK_S, HD, 2)))        # 偶/奇两半
  j   = tl.arange(0, HD)
  inv = 1.0 / tl.exp(((j*2).to(tl.float32) / D) * 9.210340371976184)
  ang = offs_s.to(tl.float32)[:, None] * inv[None, :]
  c, sn = tl.cos(ang), tl.sin(ang)
  y = tl.reshape(tl.join(xe*c - xo*sn, xo*c + xe*sn), (BLOCK_S, BLOCK_D))   # 交错回去
  tl.store(o_ptr + ... + offs_d[None,:]*so_d, y, mask=m)
  ```
  实测（`[32,8,1024,64]`，NPU=910B2C）：partner gather 74.35ms vs 连续二次 load 1.04ms vs `tl.split`/`tl.join` 1.257ms（位精确）——**gather 寻址本身约 70x**（71.3x/68.7x 两 shape 实测）。该 kernel 用 gather 时单独 152ms，而 torch 整个算子才 10.5ms；`use_rope=True` 用例 speedup 会掉到 0.03~0.10x。`tl.split`/`tl.join`/`tl.interleave` 在 triton-ascend 3.2.2 上均可用。

- ⛔⛔ **变换 kernel 必须为「输入」和「输出」分别传 stride，禁止共用一套。**
  真实输入常是**非连续的转置视图**（`x.view(B,S,N,D).transpose(1,2)`），workspace 用 `torch.empty_like(x)` 建出来**连续**——两者 stride 不同，共用一套会散射到错误地址。实测：连续张量自测 `max|diff|=0.0`（假象），非连续转置视图 `6.65`。该错误是量级性的（~1e0）、且只在需要变换的分支出现，极易误判为 kernel 数学写错。
  ```python
  rope_kernel[grid](query, q_in, B, N, S, D,
      query.stride(0), query.stride(1), query.stride(2), query.stride(3),   # 输入
      q_in.stride(0),  q_in.stride(1),  q_in.stride(2),  q_in.stride(3),    # 输出，必须独立
      ...)
  ```
  ⇒ **自测变换 kernel 时必须用非连续输入**。这与 L1.6「host 端不做 permute/contiguous」配套：正因为不许 `.contiguous()` 归一 stride，kernel 就必须显式吃下两套 stride。

- ✅ **长序列 RoPE：`inv_freq` 必须与参考实现「逐位相同」，禁止代数化简。**
  `angle = position × inv_freq` 中 `inv_freq` 的 1 ulp 偏差被线性放大 S 倍；`S=8192` 时可把输出顶到 ~2e-3。实测各写法与 torch 逐位一致性（`d_k=64`，32 元素）：`tl.exp(-e*ln1e4)` 22/32、`1.0/tl.exp2(e*log2_1e4)` 19/32、**`1.0/tl.exp(e*ln1e4)`（先正指数后取倒数，保留参考运算结构）32/32**。对齐后误差回落到 `tl.sin` vs `torch.sin` 的 1.19e-07。
  ```python
  e   = (tl.arange(0, BLOCK_D) // 2 * 2).to(tl.float32) / D
  inv = 1.0 / tl.exp(e * 9.210340371976184)   # ln(10000)，先正指数、后取倒数
  ang = s_offs.to(tl.float32)[:, None] * inv[None, :]
  ```
  - ⛔ **不要用 `tl.float64` 提精度**——Ascend 后端不支持（`LLVM ERROR: unsupported datatype for arith::TruncFOp`），只能靠对齐参考运算结构。
  - ⛔ **hi+lo 二段拆分对「大角度三角函数」无效**（对 `p` 降精度喂 `tl.dot` 有效，别混淆）：`a_lo` 可达数弧度，一阶展开前提 `|a_lo|≪1` 不成立，实测保留 8~12 位尾数时误差 0.24~14.6。
  - ⛔ **禁止手工范围规约**：`tl.sin`/`tl.cos` 自带大幅角规约，手工 `a - floor(a/2π)*2π` 因 2π 在 fp32 下不可精确表示反而更差。实测 `S=8192`（14.7% 元素 >1000 弧度）：`tl.sin(角度)` 1.19e-07（fp32 极限）vs `tl.sin(fmod 后)` 4.40e-04（差 3700 倍）。

- **判别信号**：按每次迭代成本反推，**远超 4~10us 正常区间**（如数百 us）⇒ 立刻查 KV 循环体内有无本可外提的输入变换，**不要去查 tiling 或访存**。另一信号是逐 case 分布呈**小 shape 尚可、大 shape 崩塌**。

### ★ 跨算子结论冲突的统一判据（三条，照搬会互相矛盾）

同一改动在不同 FA 算子上给出过**相反**结论。下面给出判别前置条件，**不要只看结论**。

| # | 冲突 | 甲方结论 | 乙方结论 | **统一判据** |
|---|---|---|---|---|
| 1 | `scale` 写法 | host 侧 fp32 倒数**乘法**（−36% 时延；低精度下还是逐位契约的硬要求） | 保持 `/ sqrt_dh` **除法**形式（`x/s ≠ x*(1/s)`，数值对齐要求） | **先看参考实现怎么写**：参考用 `aclnnDivs`/除法且判据按 ulp 卡 ⇒ 必须复刻除法语义（见 L1.4 与精度契约卡）；参考对 scale 无逐位要求 ⇒ 看 `aiv_vec_ratio`：**≳0.35 才做倒数乘法**，否则向量核不在关键路径、改了也是打平 |
| 2 | `scf.for` 层数 | 合并 KV 整块/尾块 **+11.7%**、压平循环维 **+3.7%**（每层循环自带一套同步） | 去掉单块路径的 `scf.for` **−10%**（编译器失去 multi-buffer 流水机会），另一算子上三种写法**全部 aicore exception 507015** | **"少一层 `scf.for` 必然更快"不成立**。判据：合并的是**同层级的冗余分支**（整块+尾块两个循环 → 一个）⇒ 值得做；去掉的是**唯一的那层循环**（单块特化）⇒ 大概率负收益或崩编译器，投入前先用单 case 脚本试崩 2~3 个 shape（每次 ~1 分钟） |
| 3 | QKV 三投影融合成一次 GEMM | **无收益**——三次调用读的同一份 x 已命中 L2，"3 次读"实际只有 1 次 GM 流量 | 是**真实收益**（x 只读一遍），但会让 benchmark 口径分数变差 | 判据是 **x 是否真的走 GM**：`aic_mte2_ratio` 高且 x 被重复读时才有收益；x 较小、已驻 L2 时是零收益。⚠️ 无论哪种，**都不要因为口径分数而做或不做**（§7.2） |

> **通例**：跨算子搬运结论前，先确认三件事——**参考实现的算术路径**（决定精度类结论）、
> **profiling 画像**（决定该优化落不落在关键路径）、**编译器/CANN 版本**（决定结构类结论是否仍成立，见 §7.3）。

---

### L1.16 ★★★ fp16 / bf16 路径必须**逐步舍入回输入 dtype**，禁止全程 fp32

低精度用例的 golden 是参考实现在 **fp16 / bf16** 下算出来的：`torch.matmul` 对
fp16 输入返回 fp16、`F.softmax` 返回 fp16、下一个 `matmul` 又在 fp16 上做。
**每一步都有一次舍入。** 实现若全程用 fp32，会「比参考更准」，反而不达标。

判据：`matched_ratio >= 0.9`，逐元素相对阈值 `rel_thr = 2^-10`（fp16）/ `2^-7`（bf16），小值域 `sv_thr = 2^-11 / 2^-8`。

实测（全程 fp32 的实现 vs 参考，四组 shape/dtype）：`d_model=48` fp16 0.7648、`d_model=60` bf16 0.7733、
`d_model=512` fp16 0.7597、`d_model=768` bf16 0.7631——全程 fp32 **稳定落在 0.76 左右**，与 shape 无关。

正确写法 —— 为低精度单开一条路径，在**参考会舍入的每个位置**都 cast 回 `ELEM_TY`：
```python
# fp32 路径：累加器 fp32，中间不降精度
# 低精度路径（ELEM_TY = tl.float16 / tl.bfloat16）：
s_r = tl.cast(tl.dot(q_tile, k_t), ELEM_TY)     # ← 参考的 matmul 返回低精度
s_r = tl.cast(s_r * inv_sqrt, ELEM_TY)          # ← 参考的除法也在低精度上做
...                                              #   softmax 同理
acc += tl.cast(tl.dot(w, v_tile), tl.float32)   # ← PV 在低精度上做，再累加
```

⚠️ 这与 **L1.1**（`tl.dot` 用 fp32 累加器）**不矛盾**：累加器仍是 fp32，
但**每次 `tl.dot` 的输出**要按参考的行为舍回低精度。区别在于"累加用什么精度"
和"中间结果以什么精度落地"是两件事。判断顺序永远是：先问 golden 是用什么精度算的，再决定实现该用什么精度。

#### ⭐ fp32 镜像：cast 分支必须 **fp16 / bf16 / fp32 三档写全，默认兜底 fp32**

从零生成容易写出"else = 低精度"的反向 bug：`else: E = exp_val.to(tl.bfloat16)` 会把 fp32 输入
无条件 cast 成 bf16，丢 ~3 个数量级精度（实测 `max_abs_diff ~5e-3`）。
```python
if is_fp16:
    E = exp_val.to(tl.float16)
elif is_bf16:
    E = exp_val.to(tl.bfloat16)
else:
    E = exp_val                       # fp32 路径任何中间量都不许落低精度 ← 兜底必须是 fp32
```

- **判别信号**：失败集合**严格按 dtype 划分**（全部且仅仅是 fp32 case），且 `max_abs_diff ~1e-3~5e-3` ⇒ 直接查 `else` 分支有没有把 fp32 中间量 cast 到低精度。
- **规则一句话**：L1.16 的镜像——低精度"参考会舍入的每个位置都舍"，fp32"参考不舍入的每个位置都不许舍"。

---
---

> 类别约束见 [flash_attention_constraints.md](../constraints/flash_attention_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 Layer 2: 算法骨架（参考方向，输出必须是全新草图）

### §3.1 主骨架

```
host:
  inv_sqrt = 1.0 / sqrt(head_dim)          # ★ host 侧算好倒数，kernel 内不做除法（§4.1）
  BLOCK_KV = min(128, ceil16(S_K))
  BLOCK_Q  = 满足 BQ*BKV <= 16384 的最大 2 的幂, 再 min(max(32, ceil32(S_Q)))
  while BQ * BLOCK_D * esz > 64KB: BQ //= 2
  BLOCK_D  = ceil16(D)
  MULTI_BLOCK = (max_kvblk > 1)            # ★ 分档开关（§4.2）
  HAS_UPPER/HAS_LOWER/HAS_SOFTCAP          # ★ constexpr 特化（L1.7）
  grid = min(NUM_CORES, B*H*cdiv(S_Q,BQ))

kernel(每个 task = 一个 (b, h, q_block)):
  加载 Q tile 一次，常驻
  m = -1e30; l = 0; acc = 0
  kv_lo, kv_hi = 区间收缩(§3.2)            # ★ 不对齐（L1.8）
  for kv in range(kv_lo, kv_hi, BLOCK_KV):
      scores = tl.dot(Q, Kt) * inv_sqrt    # 原生 dtype 操作数 + fp32 累加
      scores = 掩码(constexpr 分支)         # 有限极小值
      m_new  = tl.maximum(m, rowmax, propagate_nan=ALL if MULTI_BLOCK)
      p      = exp(scores - m_new)
      pv     = tl.dot(p, V)                # ★ Cube 提前发射（§4.7）
      alpha  = exp(m - m_new)
      acc    = acc * alpha + pv
      l      = l * alpha + sum(p)
      m      = m_new
  store(acc / l)                           # 循环外的除法不必改倒数（§4.1 失效边界）
```

### §3.2 KV 扫描区间上下界折叠（`fa-fwd` / `fa-gqa` 核心）

```python
# host 侧：causal 与 window_right 折叠成单一上界，window_left 折叠成下界
upper = min(0 if causal else INF, window_right if window_right >= 0 else INF)
lower = -window_left if window_left >= 0 else -INF
# kernel 内：对 q-block [q0, q0+BQ) 取并集
kv_lo = max(q0 + delta + lower, 0)                       # delta = S_K - S_Q
kv_hi = min(q0 + BQ - 1 + delta + upper + 1, S_K)
```

区间外整块被掩，对 online softmax 无贡献，跳过**数值等价**。实测 **+30%**，冗余倍数最高 70x。
**留一个不可裁剪的 case 当对照组**，改动前后应当不变——立刻确认收益来源正确。

> ⛔ **适用边界：golden 必须真的按 causal/window 收缩。**
> 当 golden 是 CANN op 且它**忽略窗口**（`npu_fusion_attention` 实测 sparse_mode 0/4 的
> pre/next 全被忽略 == 全可见，见 [§6.1](flash_attention__precision_gate_and_measurement.md)）时，
> 收缩会把**窗口外"本来就有贡献"的列**漏掉，直接算错——此时**禁止**做区间收缩。
> 判别：先探针 golden 对 `pre_tockens / next_tockens` 的敏感性，它忽略窗口 ⇒ §3.2 整个不适用。
> 这条对 GQA / SparseFlashAttention 复用同样踩。

### §3.3 趟数结构（由精度契约倒逼，见 §6.3）

| 条件 | 结构 |
|------|------|
| `S ≤ BLOCK_KV`（单块） | **一趟**，scores 常驻片上 |
| `S > BLOCK_KV` + 参考要求"先归一化再舍入" | **延迟归一化**（首选）或两趟（见下，⚠️ 在线两趟易踩坑） |
| 参考在 fp32 上算 | 标准一趟 online softmax |

**趟数判据（scale 阈值）**：`scale` 显著 **< 1/√d**（如 0.08838 / 0.125）⇒ softmax 近均匀 ⇒ online rescaling 的滚动 alpha 累积 fp32 舍入，MERE 顶穿 fp16 阈值。实测（FusionAttention，fp16，scale=0.08838）：单趟 MERE 1.187e-3 > 阈值 9.77e-4。**第一次生成时看到小 scale 就直接选多趟/延迟归一化，不要撞一次失败再改。**

#### ★★★ 精度全过首选结构：延迟归一化（0811/3 61/61 实锤）

**softmax 输出「未归一化 E」（不除以 l），最后 `Out = E @ V / softmax_sum`。**
从结构上**绕开 online rescaling 的滚动 alpha**——不存在跨 KV 块的 alpha 累积舍入，是 FP 精度全过的根本方案（0811/3 FusionAttention 61/61，MERE 对齐到 ~1.8e-4）。

- **必须**：softmax kernel 写 `exp_buf`（未归一化）+ `softmax_max` + `softmax_sum`；再单独 `output_kernel` 做 `Out = (E @ V) / softmax_sum`（最后一步才归一化）。
- **为什么比"在线两趟"可靠**：在线两趟的"趟 2 归一化"仍基于趟 1 的滚动 m/l，滚动 alpha 的 fp32 舍入并未真正消除；延迟归一化则完全不 rescale，无此累积。
- **性能兼顾**：三段式多 kernel 会带来 GM 往返。用 **score+softmax 融合**（QK dot 后立即算 E/max/sum 写 GM，不写完整 scores）把 GM 往返减半；PV 再读 E。这是"精度全过 + 性能好"的平衡点（0811/3 用此结构 0.89x，且在融合后已接近单 kernel 性能）。

#### ⚠️ 反例（0818/3 实锤）："在线 softmax + two_pass" ≠ 延迟归一化，4 例精度失败

0818/3 按"在线 softmax 加个 `two_pass` 开关"实现，仍 **57/61**（case 20/36/54/55，全是 fp16 大 S，MERE 9.9e-4~1.59e-3 超阈）。根因：它的 two_pass 只是**在线 rescaling 内多算一次 max**，滚动 alpha 的舍入没消除。判别：`two_pass` 后 MERE 仍在 1e-3 附近而非 ~1.8e-4 ⇒ 结构没对齐，**改成延迟归一化**，不要用"golden 噪声"当台阶。

> ⚠️ **D≥256/512：CANN 内部按 K-chunk 累加 QK，triton 复刻只能三段式（拆半核 launch）。**
> golden 是 CANN 时，D=512 的 QK 在 CANN 里是 K=256 分块累加，与整段单 `tl.dot` 差 ~5 ULP，exp 放大进 ssum 顶穿 fp32 阈值（实测 1.267e-3）。⚠️ **triton-ascend 编译器必然把单 kernel 内多个 `tl.dot` 融合成单个 K 归约**——runtime 循环 / debug_barrier / 分离变量 / 显式两半**全部无效**（已实证）⇒ 只能**三段式**：2 次独立 score 半核 launch（各 K=256 dot 写/累加到 GM workspace）+ softmax 核。chunk-k=256 后 ssum 违例 1→0。判别信号：bf16/fp32 大 D case 单独失败、`ssum` 差 ~1e-3 量级。

⚠️ **不要用"减小 `BLOCK_Q` 换单趟"**：总迭代数 = `ceil(S/BQ)·ceil(S/BKV)·趟数`，BQ 减半使第一项翻倍、趟数 2→1，**正好抵消**。已算账否决。

### §3.4 `fa-mha` 的两段结构

投影段（GEMM）与 attention 段是**两个独立的优化对象**，profiling 里分别看 `Duration` 占比，主攻目标每轮会换（实测从 attention 64%/linear 36% 变到 46%/54%）。
⚠️ **不要**把 4 次投影和 1 次 attention 合并进同一个 `@triton.jit`（用 `MODE: tl.constexpr` 分支）——见 §7.2。

---
---

> 类别约束见 [flash_attention_constraints.md](../constraints/flash_attention_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §4 Layer 3: 关键技巧（技巧可参考，变量名/结构必须重新设计）

### §4.1 ★ host 侧倒数乘法替代 kernel 内 fp32 张量除法

```python
# ❌ 每个 KV 迭代对 [BQ, BKV] 做一次 fp32 除法
scores = tl.dot(q, tl.trans(k)) / tl.sqrt(tl.cast(Dh, tl.float32))
# ✅ host 侧算好 head_dim ** -0.5
scores = tl.cast(tl.dot(q, tl.trans(k)), tl.float32) * inv_sqrt
```

- **适用判据**：向量核在关键路径（`aiv_vec_ratio ≳ 0.35`）。实测一个算子 **−36%**，另一个算子上（`aiv_vec_ratio` 仅 0.25~0.32、向量核 2/3 时间在等 Cube）**两次证伪**。
- **失效边界**：**不能外推到循环外**。末尾 `acc / l[:,None]`（每 task 一次）改倒数广播乘实测无收益。判据是「该除法是否在内层循环里、作用于二维 tile」。

### §4.2 ★ `tl.maximum(propagate_nan=ALL)` —— **必须按 KV 块数分档**

```python
if MULTI_BLOCK:      # host 侧算出的 max_kvblk > 1
    m_new = tl.maximum(m, rowmax, propagate_nan=tl.PropagateNan.ALL)
else:
    m_new = tl.maximum(m, rowmax)
```

- 默认 `propagate_nan=NONE` 在 IR 里被展开成 **7 条向量指令**（2×isnan + vmax + 2×vsel）；`ALL` 直接对应硬件单条 `vmax`。
- ⚠️ **全局开启是负收益**：实测 6.073 → 5.5302（**−9%**）；按 `max_kvblk>1` 分档 → 6.2362（+2.7%）。另一算子上单块 case 退化 25~45%（两次重复一致）。
- **语义等价性论证必须写进注释**（依赖 F2：无 NaN）。

### §4.3 ★ BLOCK 开到 UB 上限（本类算子最大的一笔）

迭代总数 = `B*H*ceil(S/BQ) × ceil(span/BKV)`，两个 BLOCK 同时翻倍 ⇒ 迭代数变 1/4，**整套 `pipe_barrier` 直接少一半**。

| SQ/SKV | 64/64 | 128/64 | 64/128 | **128/128** | 256/* |
|--------|-------|--------|--------|-------------|-------|
| 子集几何平均 | 0.697 | 1.129 | 1.155 | **1.42** | 编译失败 |

**UB 上限由编译器直接报数字**（`requires N bits while 1572864 bits available`），看错误信息就知道差多少 ⇒ **放大到编译失败为止**。
⚠️ 上限**不可用简单阈值预测**：同为 `128x128`，D=96 编得过、D=128 失败、**D=160 反而编得过且快 22%**——占用还受 constexpr 分支组合影响。

### §4.4 权重按 tile 形状预重排（**仅 `fa-mha`**，判据 `MTE2>60% 且 MAC<50%`）

投影权重是常量，可在 `__init__` 里用 CPU 任意重排，运行期零开销。

```python
# [out,in] → 转置成 [in,out]（数值逐位不变）→ 补零到 [K_pad, NBN*BLOCK_N]
# → 拆成 [NBN, K_pad, BLOCK_N] 连续
t = torch.zeros(kp, nbn * blk_n, dtype=dtype)
t[:d_model, :d_model] = w.t()
packed = t.view(kp, nbn, blk_n).permute(1, 0, 2).contiguous()
```

b tile 基址 `w_ptr + block_n*(K_pad*BLOCK_N)`，行跨度恰为 `BLOCK_N` ⇒ 整块连续，且补零后**不再需要掩码**。
- **两级递进**：先 `[out,in]→[in,out]`（**数量级收益**），再按 tile 重排（+5.0%，**无退化 case**）。
- ⚠️ **同一思路搬到 attention 的 k/v 上完全无效**（§5.2）。

### §4.5 投影分块：dtype 分档 + **并行度保护**（两半缺一不可，**仅 `fa-mha`**）

```python
par_thr = num_cores // 2            # 阈值扫参：24→3.6633、16→3.8283、12→3.8576
while nblocks(blk_m, blk_n) < par_thr and (blk_n > 64 or blk_m > 16):
    交替 blk_n //= 2 / blk_m //= 2  # 先缩 N 再缩 M
```

只放大不保护 = **−3.7%（拒绝）**；加保护 = **+2.7%**。根因是**块数塌陷**（`M=89` 时 `BLOCK_M=128` 只剩 3 个块，24 核只用 3 个）。
**fp32 必须单独一档**（元素大一倍，同 BLOCK 撑爆 UB）：实测 fp32 最优 `(64,128,256)`，16-bit 最优 `(128,256,256)`。

### §4.6 低精度 `p` 二段拆分（**仅 fp16/bf16 输入**）

```python
if PV_SPLIT:                       # = (dtype is not float32)
    p1 = p.to(in_dtype); p2 = (p - p1.to(tl.float32)).to(in_dtype)
    pv = tl.dot(p1, V_tile) + tl.dot(p2, V_tile)     # ~22 bit
else:
    pv = tl.dot(p, V_tile.to(tl.float32))            # fp32：本就是 no-op，别动
```

省掉 UB 里 `[BLOCK_KV, BLOCK_D]` 的 fp32 副本（64x128 时 **64KB，UB 里最大的一块**）。实测 **+2.7%，50/50 全过**。
**fp32 输入不要开**（无 buffer 可省，且阈值最紧、22 位没有余量）。

### §4.7 CV 指令重排（成本极低，试一次，但**不要预设收益**）

把 `tl.dot(p,V)` 提前到 `p_sum`/`alpha` 之前发射（纯重排、数值等价）。
**三个同类算子三个结果：0 / +1.9% / +6.4%。** 幅度不可预期。

---
---

> 类别约束见 [flash_attention_constraints.md](../constraints/flash_attention_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §9 与 `attention.md` 的差异声明（FA 路径上的实测推翻）

保留原条目以供对照，但**在本类算子上以本文件为准**：

| `attention.md` 条目 | 本文件 | 证据 |
|---|---|---|
| L1.1「FP32 路径**禁用** `tl.dot`，会产生 NaN 或错误结果」 | **L1.1 条件式**：禁止的是"操作数升 fp32 再 dot"；原生 dtype + fp32 累加器**必须用** | fp32 case 精度 50/50 全过，且是最大的一笔（不可测量 → 2.3554） |
| L1.3「`BLOCK_D = next_power_of_2(head_dim)`」 | **F3 用 `ceil16(D)`** | D=96/160/192 上 `next_pow2` 浪费 30~60% tile |
| L1.6「归约与累加**全程** fp32」 | **L1.4 契约驱动**：先钉死参考算术路径，两套契约并列 | 参考改原生 dtype 后，旧契约交付版只过 20/50 |
| A2 路径 (a)「把 `head_dim` 拆出去逐维外积」 | **禁止**（与 L1.1 冲突） | 逐维外积正是 Cube 空转的成因 |
| A1（padding 掩码）、A2（UB 上界）、§2.0（host 语义）、L1.2 / L1.5 / L1.7 | **已承接**为 F1 / F4 / L1.11 / L1.2 / L1.3 / L1.10 | 无冲突，逐条保留 |
