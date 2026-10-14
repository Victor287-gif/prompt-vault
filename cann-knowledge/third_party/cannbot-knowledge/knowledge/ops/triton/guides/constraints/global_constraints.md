---
type: Guide
title: Triton Ascend 全局设计约束
description: 跨全部算子类别复用的 15 条设计期硬约束，覆盖并行配置、编译期常量化、精度提升、访存与平台限制；各类别约束卡只写本类别特有条款并链接引用本卡。
tags: [task.design, topic.tiling, topic.compilation, topic.precision, topic.implementation.masking, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# Triton Ascend 全局设计约束

本卡收敛原先散落在 13 个算子类别文件 `§1 通用经验` 中的跨类别重复条款。
每条约束在原始语料中至少被 2 个类别独立提出（括号内为命中的类别文件数）。

**引用方式**：类别约束卡与算子 playbook 卡引用本卡条款时写
`见 GC-N`（相对链接指向本卡对应锚点），不复制原文。

**编号纪律**：原语料中 `G1-G8` 被 `tensor-transform` / `index-computation` /
`matrix-compute-merged` / `transformer-training` 四个文件各自占用且含义不同，
跨文件引用因此存在歧义。本卡统一为 `GC-*` 命名空间，全局唯一。

---

## 适用范围

全部 Triton Ascend 算子，设计草图阶段即须满足。违反其中任何一条通常表现为
精度失败、编译失败或性能跌至 PyTorch 基线以下，而非可选优化。

---

## GC-1 动态读取核数，禁止硬编码（8 类）

- **必须** 运行时读取实际核数，elementwise / reduction 用 `num_vectorcore`，
  cube / matmul 用 `num_aicore`。
- **禁止** 硬编码 `num_cores = 8` / `48` / `40`。
- **必须** 在 `__init__` 中查询一次并缓存，**禁止** 在 `forward()` 中重复查询。

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]
```

- **Why**：硬编码在实测中仅利用约 20% Vector Core，加速比从 ~1.3x 跌至 0.67x，
  慢于 PyTorch。每次 `forward` 重复查询则引入无谓 host 开销。
- **已弃用**：旧式 `npu_config` 取值（仅 vector、硬编码 40 不准、设备未 init 会抛
  `RuntimeError`）。

---

## GC-2 BLOCK 向上取 2 的幂并声明为 constexpr（11 类）

- **必须** 将 BLOCK / 向量化长度取 `triton.next_power_of_2(N)`，且以
  `tl.constexpr` 传入 kernel。
- **Why**：Ascend 上 fixed-shape vector load 要求编译期常量长度；`tl.arange(0, N)`
  中 N 非 constexpr 会触发 dynamic-shape load，退化为标量循环。

典型档位：

```
width <= 64 -> 64      <= 512  -> 512     <= 4096 -> 4096
<= 128 -> 128          <= 1024 -> 1024    else    -> 8192
<= 256 -> 256          <= 2048 -> 2048
```

- **例外**：Flash Attention 的 `BLOCK_D` 用 `ceil16(head_dim)` 而非
  `next_pow2`——triton-ascend 支持非 2 的幂 `tl.arange`，`next_pow2` 在
  D=96/160/192 时白白浪费 30~60% tile 面积。见 attention 类别卡。

---

## GC-3 一维 grid + 核数钳制 + 核内循环（12 类）

- **必须** `grid = (min(total_blocks, num_cores),)`，kernel 内用
  `for i in range(pid, total_blocks, num_cores)` 覆盖剩余任务。
- **禁止** 3D grid；**禁止** 直接 `grid = (total_blocks,)`。
- **必须** 按**输出元素总数**分配，而非输入元素数。
- **必须** 当 kernel 内依赖 `num_cores` 入参计算循环 stride 时，
  `num_cores` 入参与实际 `grid_size` **严格相等**。
- **Why**：Ascend 仅支持一维 grid，且 grid 各维上限 65535；输出规模大时
  `total_blocks` 可能远超核数，多余 program 空跑。`num_cores ≠ grid_size` 会让
  `rows_per_core = cdiv(batch, num_cores)` 与实际 grid 不一致，导致部分 core
  处理范围错误（MoeGatingTopKSoftmax 实测引发结果错误）。

负载均衡分配（每核 block 数差不超过 1）：

```python
blocks_per_core = total_blocks // num_cores
remainder = total_blocks - blocks_per_core * num_cores
if pid < remainder:
    my_blocks = blocks_per_core + 1
    start_block = pid * (blocks_per_core + 1)
else:
    my_blocks = blocks_per_core
    start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core
```

- **例外**：`total_blocks <= num_cores` 时可用 2D grid 提高并行度。
- **例外**：kernel 单 program 的 UB 占用随「每 program 处理的行数」线性增长时，
  强行钳到 `num_cores` 会让每 program 多担行数而触发 UB overflow。典型为
  `ext.sort` / `tl.sort` 路径（multi-buffer，fp32 `L=4096` 实测 208KB > 192KB 上限），
  应直启 `grid = (n_rows,)` 或 `cdiv(n_rows, BLOCK_ROWS)`；同一算子的软件 bitonic
  路径 UB 占用低，仍按本条钳制。见
  [`operators/math_compute__sort.md`](../operator_playbooks/math_compute__sort.md) §5.3。
- **注意**：小 shape 下交织划分 `range(pid, total, num_cores)` 延迟偏高，
  可改连续块划分，见 convolution 类别卡。

---

## GC-4 `forward()` 中禁止 torch / torch_npu 计算（4 类）

- **必须** `ModelNew.forward()` 只做 shape 计算、路径分派、host 预计算；
  所有数值计算在 `@triton.jit` kernel 内完成。
- **禁止** `F.pad`、`F.conv*`、`torch.unfold`、`torch.softmax`、`torch.topk`、
  `torch.bincount`、`torch.cumsum`、切片赋值、用于计算的 `permute`、
  `min()` / `max()`。
- **允许**（纯布局/元信息）：`view` / `reshape` / `contiguous` / `squeeze` /
  `.to` / 纯布局的 `permute`。
- **Why**：`validate_triton_impl.py` 的 Type-3 退化检测会 flag 任何 forward 中的
  torch 计算；host 侧 torch 计算还引入大拷贝与额外 launch。

---

## GC-5 int32 索引，避免 int64 降级（6 类）

- **必须** kernel 内把 `tl.program_id` 与 `tl.arange` 结果 `.to(tl.int32)`。

```python
pid = tl.program_id(0).to(tl.int32)
offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
```

- **Why**：int64 标量触发地址计算降级；NPU 上 int32 索引更高效。
- **通用 32 位化**：除非数值精度需要，kernel 内计算/索引**禁止 int64**，统一 32 位。
- **比较与除法走 fp32/int32**：在不影响精度时，`>=`/`>`/`<=`/`<` 比较转 fp32、
  `==`/`!=` 转 fp32 或 int32、除法用 fp32/int32——int64/int32 CMP、int 除法
  会触发 scalar fallback（见 hardware `constraints_cheatsheet.md` P1/P2），
  转 fp32/int32 才启用向量化。

---

## GC-6 输入 contiguous 化（10 类）

- **必须** host 侧进入 kernel 前对参与计算的张量调用 `.contiguous()`。
- **Why**：避免 kernel 内 stride 计算复杂化与非连续访存退化。

> **已知冲突**：`transformer-training` 类别提出「多 layout 输入在 host 侧统一，
> **禁止**强制 `.contiguous()`」——当输入本身已是目标 layout 时，无条件
> `.contiguous()` 会引入一次多余全量拷贝。**处置**：默认遵循本条；当算子需要
> 支持多种输入 layout 时，改为在 host 侧判定 layout 后按需转换，并在类别卡中
> 显式声明例外。

---

## GC-7 归约维 padding 必须显式排除，`other=0.0` 不是掩码（9 类）

- **必须** 归约维长度非 BLOCK 整数倍时，`max` 前 `tl.where(valid, x, NEG)`，
  `sum` 前 `tl.where(valid, p, 0.0)`。
- **禁止** 只靠 `tl.load(..., other=0.0)` 就认为越界元素已被排除。
- **Why**：`other=0.0` 只让越界元素的值变成 0，而 `exp(0 - m) > 0`——这些
  「幽灵列」会混进 softmax 分母，把 `acc / l` 整体缩小。实测某 MHA 的 50 个
  case **全部** `S_K % 32 != 0`，单这一条的 MERE = 8.16e-01，超 fp32 阈值 6700 倍。
- **验证集要求**：不能指望随机 shape 覆盖到归约边界，必须主动构造
  `S % BLOCK == 0` 与 `!= 0` 两类用例；否则该缺陷会逃逸成「48/50 通过」
  这种更难诊断的形态。

---

## GC-8 UB 预算：中间张量元素数必须有与输入无关的上界（3 类）

- **绝对禁止** `BLOCK_* = triton.next_power_of_2(<运行期变量>)` 且不加 `min()` 封顶。
- **必须** 每个 UB 中间张量的元素数满足：3D broadcast ≤ 12288，
  2D tile ≤ 16384（kernel 内有循环时）。
- **Why**：UB 仅 192 KB（`ub overflow, requires ... while 1572864 bits available`），
  含 multi-buffer(×2) 与归约临时后有约 3x 放大。实测 2D 16384 元素 OK，
  32768 元素触发 `ub overflow`（实际占用 391~396 KB）。真正的病因是**某个 block
  维直接由运行期 shape 决定且无上界**——`BLOCK_N = next_pow2(S)` 在 `S=577`
  时需要 792 KB。
- ⚠️ **UB overflow 会污染设备，导致级联假失败**：一次 `ub overflow` 可触发
  vector core exception，使后续用例连带失败。实测一次全量从 49/50 掉到 36/50，
  其中真正的硬失败只有 3 个，其余 11 个是设备被污染后的陪葬。
  → **诊断原则**：看到大批连续失败时，先定位**第一个**失败用例并单独复跑；
  不要按失败总数估计缺陷数量。

---

## GC-9 编译期常量化：`tl.arange` 长度与循环边界必须 constexpr（12 类）

- **必须** 把 `tl.arange` 参数、`range()` 循环边界、repeat 次数等作为
  `tl.constexpr` 传入，包括非 2 的幂的语义维度（如 `num_expert=64`、`k=8`）。
- **禁止** 在 kernel 内用普通入参做 `tl.arange(0, n)`。
- **Why**：否则报 `ValueError: arange's arguments must be of type tl.constexpr`；
  constexpr 还让编译器展开循环、生成 vector load/store，触发启动级特化。
- **局部变量例外**：函数体内**禁止**写 `m: tl.constexpr = 1 << s` 这类局部
  constexpr 声明（会报 `_builder argument must be provided`）；去掉注解即可，
  编译器会自动常量传播。
- 与 GC-2 的分工：GC-2 管 BLOCK 取 pow2，本条管**任何**进入 `tl.arange` /
  `range()` 的变量都必须 constexpr。
- **循环索引从循环变量独立计算，禁止可变累加器**：展开循环内禁止
  `off_block = off_block + C`（跨迭代真实数据依赖 → 编译器无法并行化展开体，
  性能大幅劣化）；必须每轮 `off = i * C + base` 从循环变量 `i` 独立派生：

```python
# 禁止（可变累加器，迭代间数据依赖，展开体被迫串行）
off = tl.arange(0, BLOCK)
for i in range(NUM_BLOCK):
    ... tl.load(p + off) ...
    off = off + BLOCK
# 正确（每轮从 i 独立计算，无迭代间依赖，可并行调度）
base = tl.arange(0, BLOCK)
for i in range(NUM_BLOCK):
    off = i * BLOCK + base
    ... tl.load(p + off) ...
```

---

## GC-10 fp16/bf16 必须在 kernel 内升 fp32 计算（9 类）

- **必须** fp16/bf16 输入在 kernel 内 `.to(tl.float32)` 后再做乘加减与归约，
  最后转回原精度存储。
- **必须** 统计归约（sum / sq_sum / var）使用 fp32 累加器。
- **Why**：直接以 fp16/bf16 做归约会让相对误差超出 verify 阈值（>1e-3），
  且 PyTorch 参考实现普遍在算子内部隐式提升到 fp32。归一化的
  `(sq - sum*sum/N)/N`、`inv_std^3`、`1/S` 等小量乘法尤其容易下溢。

---

## GC-11 `tl.cdiv` 与 `triton.cdiv` 禁止混用（5 类）

- **必须** kernel 内用 `tl.cdiv`，host 侧用 `triton.cdiv`。
- **禁止** 在 kernel 内调用 `triton.cdiv`。

---

## GC-12 kernel 内禁止 `%` 模运算（6 类）

- **必须** 用 `a - (a // b) * b` 代替 `a % b`；比较用 int32。
- **典型场景**：tile / oh / ow / kk / group 的坐标解码。

---

## GC-13 掩码填充用有限极小值，不用 `-inf`（3 类）

- **适用范围**：填充值会**进入 `exp` / `tl.maximum` 归约链**的场景 —— softmax、
  online softmax、attention 掩码、logsumexp 等。
- **不适用**：填充值只参与**比较-选择**且排序后被截断丢弃的场景，典型为 Sort / TopK
  的 `pad 到 next_pow2`（见 [`operators/math_compute__sort.md`](../operator_playbooks/math_compute__sort.md) L1.3，
  按方向取 `+inf`(asc) / `-inf`(desc)）。该场景无 `exp`，`-inf` 是正确且必要的
  哨兵值，用 `-3.0e38` 反而可能与真实数据碰撞。
- **必须** 用 `-3.0e38`（fp32）或 `-1e30`，并声明为 `tl.constexpr`
  （否则报 `NameError: Cannot access global variable`）。
- **禁止** 用 `-inf`。
- **Why**：`-inf` 会在整行被掩时产生 `exp(-inf - (-inf)) = NaN`，并逼迫
  `tl.maximum` 保留 NaN 处理路径，与性能优化直接冲突。
- **安全性论证**（应写进代码注释）：causal + online softmax 下 KV 循环从 0 开始，
  第一块必含合法列 ⇒ 每行首次更新时 `m` 即为有限值，后续整块被掩的行
  `p = exp(-3e38 - m) = 0`，不污染结果。

---

## GC-14 多策略分派，禁止单 kernel 统所有路径（6 类）

- **必须** 在 host 侧按 `(维度, 模式, dtype, 采样比例, stride, groups)` 等特征
  分派到特化 kernel。
- **禁止** 用单一通用 kernel 统一处理所有路径。
- **Why**：通用 kernel 的坐标解码 overhead 极大，且丢失全部特化机会。
- **兜底**：通用 kernel 仅在不存在特化匹配时使用。

---

## GC-15 Triton Ascend 平台硬限制（7 类）

以下为平台级限制，违反直接编译失败或静默算错：

- **禁止** kernel 内嵌函数定义（全部内联）。
- **禁止** 3D grid（见 GC-3）。
- **禁止** kernel 内使用 `continue` / `break`。
- **禁止** 对已加载的局部张量做切片 `tensor[a:b, c:d]`
  → `unsupported tensor index: slice`，编译期失败。
- **禁止** 把 `reshape` / `trans` 的输出直接喂给 `tl.dot`——布局元数据被忽略，
  `store` 取值对但 `dot` 全错。`dot` 的两个操作数必须来自 genuine 2D
  `block_ptr` 或 `tl.load`。
- **禁止** 3D batched `tl.dot`（编译或执行挂死 >10min）。
- **禁止** 用运行时谓词包绕 `tl.dot`（`if runtime_cond: acc = tl.dot(...)`）
  → Ascend `linalg_to_bin` / bishengir-compile pass 崩溃，全 case
  `MLIRCompilationError`。改用编译期 `tl.constexpr` 分支，或双命名累加器 +
  无条件 `dot`；运行时边界只能靠 `tl.load` 的 `mask` / `boundary_check` 处理。

---

## 与类别约束卡的分工

本卡只收跨类别复用条款。以下内容留在各自类别约束卡，不上提：

| 类别 | 保留的本类别特有条款（示例） |
|---|---|
| convolution | 权重手工 CPU 初始化与 fan_in 口径、block_ptr 负 offset host 预补零、block_ptr 连续轴单位 stride、reduction 维对齐 Cube C0=16、groups 分组处理 |
| attention / flash_attention | `BLOCK_D = ceil16(D)`、结构性改动后重扫 tile、扫参子集含大中小三档 |
| normalization | 跨 tile 归约两阶段 reduce-then-apply、weight/bias 标量 load 路径 |
| index-computation | 并行轴与连续访问对齐 |
| math-compute | 语义维度归一化到末维、非 2 幂维 pad 到 next_pow2 + 极值填充 + 截断 |
| recurrent-neural-network | 时间维串行、每 program 持有完整隐维 H、双缓冲工作区规避递归 RAW |
| object-detection | NPU 原生算子与标准算法不兼容时替换 Golden |
| quantization | 多维输入 host 侧扁平化为 2D、int8 矩阵乘用 int32 accumulator |
