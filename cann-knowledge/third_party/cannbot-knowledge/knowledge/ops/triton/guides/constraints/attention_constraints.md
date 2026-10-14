---
type: Guide
title: attention 类算子设计约束
description: attention 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.attention, topic.implementation.memory_management, topic.compute.normalization, topic.memory, topic.compute.reduction]
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

# attention 类算子设计约束

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## Attention 类算子优化经验

> ⛔ **FA 类算子请改读 [`flash_attention.md`](flash_attention_constraints.md)，不要用本文件。**
> 判别特征（满足任意一条即属 FA 类）：① kernel 内对 KV 维度分块循环且跨迭代滚动 online softmax 状态量 `m`/`l`/`acc`；
> ② attention 主链带 `causal` / `window_left` / `window_right` / `softcap` 等 mask 属性；
> ③ 计算图含 `scaled_dot_product_attention` 或 `softmax(Q@Kᵀ/√d)@V` 且 `S` 大到不宜物化 `[S_Q,S_K]`。
>
> **原因**：本文件的 L1.1（FP32 路径禁用 `tl.dot`）、L1.3（`BLOCK_D = next_power_of_2`）、L1.6（归约与累加全程 fp32）
> 三条在 FA 路径上**已被实测推翻**（fp32 `tl.dot` 精度 50/50 全过且是最大的一笔收益；`ceil16` 比 `next_pow2` 省 30~60% tile；
> 精度策略须由参考实现的算术路径决定）。逐条差异见 `flash_attention.md` §9。
> 把它们当作硬性边界会让 Phase 2 Step 3 门禁把正确草图判为 A 类错误。
>
> 本文件保留用于**不含 KV 分块的朴素 attention**，内容待重构。

本文档合并该类别算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复；与其他类别共用的通用约束见 GC-1）
- **§2 MultiHeadAttention**（mha）：§2.0 首次生成必读 + §2.1 Layer 1 设计约束
- **§4 常见陷阱**（按算子分小节）

> Layer 2 骨架与 Layer 3 kernel 暂缺 —— 当前 Layer 1 已用文字写全全部约束，待后续成功探索经 Phase 7 回填代码骨架。

> ⚠️ **关键区分**：本类别的核心优化哲学是 **online softmax 分块 + UB 中间张量元素数有与输入无关的上界 + 归约全程 fp32 对齐参考语义**。生成时**禁止混用**其他类别的经验——尤其不要套用 `transformer-inference.md` 中 softcapping-softmax 的"多 kernel 拆分强制中间舍入"策略：attention 主链的参考实现普遍显式 `.float()`，中间不物化低精度，拆分反而引入额外误差。

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/attention__multiheadattention.md`](../operator_playbooks/attention__multiheadattention.md) — MultiHeadAttention 算子（mha）

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| MultiHeadAttention | `mha` | 4 个 Linear 投影 + `softmax(Q@Kᵀ/√d)@V`，`S_Q ≠ S_K`，多 dtype，`head_dim` 非 2 的幂 | host 侧忠实复刻 nn.Linear 语义 + kernel 侧 online softmax，UB 中间张量受控 |
| self-attention / cross-attention | `sdpa` | 同上但权重由外部传入 | 同上，去掉投影段 |
| flash-attention | `flash-attn` | KV 分块 + online softmax，不物化完整 `[S_Q, S_K]` | 同上，重点在 `m`/`l`/`acc` 同步更新 |

> 与 `transformer-inference.md` 的分工：后者收录 RotaryMul / MoE / softcapping-softmax 等**围绕** attention 的算子；本文件针对 **Q@Kᵀ → softmax → @V** 这条主链本身。

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 2 条约束是本类别算子**共有**且**未在其他 template 文件覆盖**的工程约束。GC-1）/ G2（pow2 BLOCK）/ G4（grid 不超核数）/ G7（contiguous）与 `transformer-inference.md` 的 T2（fp16/bf16 升 fp32）/ T4（编译期常量化）此处不再重复。

### A1 归约维度的 padding 必须显式排除，`other=0.0` 不是掩码

- **必须** 当 `S_K` 不是 `BLOCK_SK` 的整数倍时，`max` 前 `tl.where(sk_valid, scores, NEG)`，`sum` 前 `tl.where(sk_valid, p, 0.0)`。
- **禁止** 只靠 `tl.load(..., other=0.0)` 就认为越界列已被排除。
- **Why:** `other=0.0` 只是让越界列的 score 变成 0，而 `exp(0 - m) > 0` —— 这些"幽灵列"会混进 softmax 分母，把 `acc / l` 整体缩小。实测某 MHA 的 50 个 case **全部** `S_K % 32 != 0`，单这一条的 MERE = 8.16e-01（超 fp32 阈值 6700 倍）。
- **典型应用**：所有 KV 分块的 attention；同理适用于任何 `S % BLOCK != 0` 的归约。
- **验证集要求**：不能指望随机 shape 覆盖到归约边界，必须主动构造 `S_K % BLOCK_SK == 0` 与 `!= 0` 两类用例，否则该缺陷会部分逃逸成"48/50 通过"这种更难诊断的形态。

### A2 UB 预算：中间张量元素数必须有与输入无关的上界

- **绝对禁止** `BLOCK_* = triton.next_power_of_2(<运行期变量>)` 且不加 `min()` 封顶。这是本条唯一的硬禁止项，其余都是可选路径。
- **必须** 让每个 UB 中间张量的元素数满足：**3D broadcast ≤ 12288**，**2D tile ≤ 16384**（kernel 内有循环时）。实测边界：2D 16384 元素 OK；32768 元素触发 `ub overflow`（实际占用 391~396 KB > 192 KB）。
- **达成路径二选一，等价**：
  - **(a) 把 `head_dim` 拆出去** —— 归约维逐 d 从 GM 取 1D 向量做外积，UB 占用变为 `BLOCK_SQ × BLOCK_SK × 4B`，与 `head_dim` 完全无关；
  - **(b) 保留 `head_dim` 作为一维** —— 用 2D tile `[BLOCK_N, BLOCK_D]`，`BLOCK_D = head_dim`，但给 `BLOCK_N` 设**与输入无关的死上限**（实测 `min(..., 64)` 可覆盖 `head_dim` 20~80）。
- **Why:** UB 仅 192 KB（`ub overflow, requires ... while 1572864 bits available`），含 multi-buffer(×2) 与归约临时后有约 3x 放大。真正的病因不是"绑定了 `head_dim`"，而是**某个 block 维直接由运行期 shape 决定且无上界** —— 实测 `BLOCK_N = next_pow2(S)` 在 `S=577` 时需要 792 KB，直接溢出。
- ⚠️ **UB overflow 会污染设备，导致级联假失败**：一次 `ub overflow` 可触发 vector core exception，使**后续用例连带失败**。实测一次全量从 49/50 掉到 36/50，其中**真正的硬失败只有 3 个，其余 11 个是设备被污染后的陪葬**。
  → **诊断原则**：看到大批连续失败时，先定位**第一个**失败用例并单独复跑；不要按失败总数估计缺陷数量。
- 与 `transformer-inference.md` L1.5 的差异：L1.5 给的是"引入循环后 UB budget 从 128KB 收紧到 48KB"这个**总预算**，A2 进一步给出**按张量维度的元素数上界**，并明确禁止无封顶的 `next_power_of_2(运行期变量)`。

---

## §4 常见陷阱与避免方法

### §4.1 MultiHeadAttention 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 权重在 NPU 上重采样 | CPU 与 NPU 是两条独立 RNG 流，同 seed 结果完全不相关 | 照抄参考实现的构造语句（coding skill C1） |
| 直接以 fp16/bf16 采样权重 | 参考实现先 fp32 采样再 `.to(dtype)` | 同上，对齐 dtype 时序 |
| 投影算成 `x @ W` | `nn.Linear.weight` 是 `[out, in]`，方阵下 shape 检查放行 | 交换 stride：`w.stride(1), w.stride(0)`（coding skill C2） |
| softmax 分母偏大、结果整体偏小 | KV padding 列未置 `NEG` 就进了 `sum` | A1 |
| 输出大面积 NaN | **因果未确认**：某 MHA 实测 240/300 NaN，换有限大负数 `-3.0e38` 后消失；但同一改动还同时调整了 Q@Kᵀ 写法与 `p` 的显式清零，未做隔离验证。其他算子上 `-float('inf')` 工作正常 | 先按 A1 检查掩码；仍有 NaN 再试有限大负数（须 `tl.constexpr`，否则 `NameError: Cannot access global variable`） |
| `head_dim` 大时 `ub overflow`，小时正常 | 某 block 维直接由运行期 shape 决定且无上界 | A2：3D ≤ 12288 / 2D ≤ 16384 元素，路径 (a) 拆 `head_dim` 或 (b) 给另一维设死上限 |
| 大批用例连续失败，但逐个复跑只错几个 | `ub overflow` 触发 vector core exception 污染设备，后续用例陪葬（实测 49/50 → 36/50，真实缺陷仅 3 个） | 定位**第一个**失败用例单独复跑，不按失败总数估计缺陷数 |
| `BLOCK_N = next_power_of_2(S)` | 无 `min()` 封顶，`S=577` 时需 792 KB | A2 绝对禁止项 |
| 「FP32 `BLOCK_SK` 必须 ≤ 8」 | **误诊**：真实原因是 KV padding 列未排除出归约，不是 block 太大 | 按 A1 补掩码后实测 `BLOCK_SK=16` 与 8 结果一致（maxdiff ~1e-7），不必牺牲性能 |
| 非 fp32 case 全部编译失败 | `tl.dot` 两操作数 dtype 不一致 | L1.2 |
| fp32 分支编译失败 `unsupported tensor index` | 用了 `q_tile[:, d]` 运行时索引 | L1.7 |
