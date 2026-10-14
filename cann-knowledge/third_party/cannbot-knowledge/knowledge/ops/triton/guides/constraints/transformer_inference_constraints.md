---
type: Guide
title: transformer_inference 类算子设计约束
description: transformer_inference 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.attention, topic.tiling, topic.compute.ordering, topic.compute.normalization, topic.mixture_of_experts]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---

# transformer_inference 类算子设计约束

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## Transformer 推理类算子优化经验

本文档合并了六类 Transformer 推理算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复；与张量变换类共用的通用约束见 GC-1）
- **§2 RotaryMul**（rotarymul，RoPE 旋转位置编码）
- **§3 MoeComputeExpertTokens**（indexing-gather / counting，MoE 专家 token 计数 + 前缀和）
- **§4 MoeGatingTopKSoftmax**（sort-topk，门控 softmax + 迭代 top-k）
- **§5 AttentionSoftmaxWithSoftcappingAndDropout**（reduce，softcapping + 行级 softmax 融合）
- **§6 LightningIndexer**（topk-index，decode 场景 top-k 索引）
- **§7 SparseFlashAttention**（sparse-attention，MLA-absorb 稀疏 attention）
- **§8 各算子常见陷阱**
- **§9 相关记忆**

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/transformer_inference__attentionsoftmaxwithsoftcappinganddropout.md`](../operator_playbooks/transformer_inference__attentionsoftmaxwithsoftcappinganddropout.md) — AttentionSoftmaxWithSoftcappingAndDropout 算子（reduce）
- [`operators/transformer_inference__lightningindexer.md`](../operator_playbooks/transformer_inference__lightningindexer.md) — LightningIndexer 算子（topk-index）
- [`operators/transformer_inference__moecomputeexperttokens.md`](../operator_playbooks/transformer_inference__moecomputeexperttokens.md) — MoeComputeExpertTokens 算子（indexing-gather / counting）
- [`operators/transformer_inference__moegatingtopksoftmax.md`](../operator_playbooks/transformer_inference__moegatingtopksoftmax.md) — MoeGatingTopKSoftmax 算子（sort-topk）
- [`operators/transformer_inference__rotarymul.md`](../operator_playbooks/transformer_inference__rotarymul.md) — RotaryMul 算子（rotarymul）
- [`operators/transformer_inference__sparseflashattention.md`](../operator_playbooks/transformer_inference__sparseflashattention.md) — SparseFlashAttention 算子（sparse-attention）

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| RotaryMul | `rotarymul` | 4D `[B,H,S,D]` 张量 half/interleave 模式旋转乘法，fp16/bf16/fp32 | per-position 向量化 + uniform grid splitting + 2D tiling + broadcast stride |
| MoeComputeExpertTokens | `indexing-gather / counting` | expert 维度 token 计数（histogram）+ 小数组前缀和 | 两阶段分离：expert-parallel 无竞争计数 + 单 block 串行 prefix sum |
| MoeGatingTopKSoftmax | `sort-topk` | softmax over last dim + 迭代 top-k 选择，per-row 独立 | 纯寄存器 top-k 路径（无 GM temp buffer）+ grid 钳制到 num_cores |
| AttentionSoftmaxWithSoftcappingAndDropout | `reduce` | Gemma3 风格 softcapping `tanh(x/30)*30` + 行级 softmax，多 dtype 混合 | 4-kernel 分离强制中间舍入 + 分核优化（grid 钳制 + 循环处理多块） |
| LightningIndexer | `topk-index` | per-(B,S1) 对 S2 个 score 取 top-K；score = Σ_n w_n · ReLU(q_n @ k)；causal mask；BSND/PA_BSND/TND layout | 在 Triton-Ascend 可表达范围内，针对 (K, S2, R) 选择最低开销的 top-k 路径并尽量隐藏 score workspace 的 GM 往返 |
| SparseFlashAttention | `sparse-attention` | MLA-absorb (attention_mode=2)、MQA/N2=1、按 `sparse_indices` gather KV、`rightDownCausal` mask | 最小化 gather 流量与 GM 往返；超出 Triton 表达能力的硬件 gather 优化不要强行模拟 |

> ⚠️ **关键区分**：六类算子计算模式差异极大，优化哲学不可混用：
> - RotaryMul 关心 **per-position 向量化** 避免 flat-1D 标量退化
> - MoeComputeExpertTokens 关心 **无竞争 expert-parallel 计数** 避免 atomic contention
> - MoeGatingTopKSoftmax 关心 **寄存器内 top-k 路径** 避免 GM 往返导致 `tl.argmax` 不可靠
> - AttentionSoftmaxWithSoftcappingAndDropout 关心 **多 kernel 分离强制 dtype 舍入** 匹配 PyTorch 中间物化行为
> - LightningIndexer 关心 **decode/small-batch 场景下 top-k 选择的固定开销与并行度**，0.8x/1.0x 在当前纯 Triton 约束下不可达
> - SparseFlashAttention 关心 **sparse gather 的 MTE2/MTE3 流量**，超出 Triton 表达能力的硬件 gather 优化不要强行模拟

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 6 条约束是四类 Transformer 推理算子**共有**且**未在 GC-1）/ G2（pow2 BLOCK）/ G3（多策略分派）/ G4（grid 不超核数）/ G5（int32 索引）/ G6（负载均衡）/ G7（contiguous）/ G8（坐标 float32 比较）此处不再重复，各算子章节引用时标注。

### T1 grid_size 必须与 num_cores 严格匹配（grid = num_cores 或 grid ≤ num_cores）
- **必须**令 `grid_size = min(batch_size/total_blocks, VEC_CORE_NUM)`，且 `num_cores` 入参 = `grid_size`。
- **禁止**设置 `grid_size > VEC_CORE_NUM` 或 `num_cores ≠ grid_size`。
- **Why:** 当 `num_cores ≠ grid_size` 时，任务划分公式 `rows_per_core = cdiv(batch, num_cores)` 与实际 grid 不一致，导致部分 core 处理范围错误（MoeGatingTopKSoftmax 实测引发结果错误）。
- **典型应用**：MoeGatingTopKSoftmax（grid = min(batch_size, VEC_CORE_NUM)，num_cores = grid_size）、AttentionSoftmax（grid_size = min(natural_blocks, num_cores)）。
- 与 G4 的差异：G4 仅要求 grid 不超过核数，T1 进一步要求当 kernel 内依赖 `num_cores` 入参计算循环 stride 时，二者必须**严格相等**。

### T2 fp16/bf16 输入必须在 kernel 内升精度到 fp32 计算
- **必须**对 fp16/bf16 输入在 kernel 内 `.to(tl.float32)` 计算，再转回原精度存储。
- **Why:** 直接以 fp16/bf16 做乘加减/归约会导致精度误差超出 verify 阈值（relative error > 1e-3），且 PyTorch 参考实现普遍在算子内部隐式提升到 fp32。
- **典型应用**：RotaryMul（fp16/bf16 → fp32 旋转乘法 → 转回）、AttentionSoftmax（所有 div/tanh/mul/softmax 子算子均 fp32 计算）、MoeGatingTopKSoftmax（softmax 在 fp32 下做 max/exp/sum/div）。

### T3 禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）
- **必须** `ModelNew.forward()` 中只负责 shape 计算、分派、host 预计算；所有数值计算必须在 `@triton.jit` kernel 内完成。
- **Why:** `validate_triton_impl.py` Type-3 检查会 flag 任何 forward 中的 torch 计算为退化；同时也是 Triton-Ascend 算子的基本要求。
- **典型应用**：MoeGatingTopKSoftmax 禁用 `torch.softmax` / `torch.topk`；MoeComputeExpertTokens 禁用 `torch.bincount` / `torch.cumsum`。

### T4 编译期常量化：循环边界、tl.arange 长度、repeat 次数等必须为 tl.constexpr
- **必须**将 `num_expert`、`k`、`MAX_HALF_D`、`BLOCK_SIZE`、`TILE_S`、`r` 等作为 `tl.constexpr` 传入 kernel。
- **禁止**在 kernel 内使用 `tl.arange(0, num_experts)` 其中 `num_experts` 为普通入参（非 constexpr）。
- **Why:** Triton Ascend 要求 `tl.arange` 参数必须是编译时常量，否则报 `ValueError: arange's arguments must be of type tl.constexpr`；同时 constexpr 让编译器展开循环、生成 vector load/store。
- 与 G2 的差异：G2 强调 BLOCK 长度取 pow2，T4 强调**任何**进入 `tl.arange` / 循环 `range()` 的变量都必须 constexpr，包括非 pow2 的语义维度（如 `num_expert=64`、`k=8`）。

### T5 编译器融合会消除中间 dtype 舍入，必要时拆分独立 kernel 强制 GM round-trip
- **必须**当参考实现（PyTorch C++）在每个子算子处物化中间结果为输入 dtype 时，Triton kernel 若被编译器融合为单一 fp32 计算会跳过舍入，导致逐 bit 不匹配。
- **解决**：将表达式拆分为多个独立 kernel，每个 kernel 通过 GM store+load 往返强制中间 dtype 舍入。
- **典型应用**：AttentionSoftmaxWithSoftcappingAndDropout 的 `tanh(x/30)*30` 在 bf16/fp16 下必须拆为 div/tanh/mul 三独立 kernel。
- **fp32 输入例外**：fp32 下融合后仍 fp32，无舍入损失，不必拆分。

### T6 multibuffer / unit_flag 等编译选项需实测验证，禁止默认开启
- **禁止**盲目添加 `multibuffer=True, unit_flag=True` 期望提升内存密集型算子性能。
- **Why:** 在含迭代标量循环（如 MoeGatingTopKSoftmax 的 top-k）的算子上，multibuffer 的流水线优化收益被迭代开销掩盖，实测可能劣化（MoeGatingTopKSoftmax v3: 0.8527x vs 基线 0.8836x，v10: 0.5720x vs 基线 0.8194x）。
- **正确做法**：在 Phase 4 中实测验证后再决定是否启用。

---

## §8 常见陷阱与避免方法

### §8.1 RotaryMul 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Flat-1D 索引分解导致标量退化 | kernel 内用 `d = offsets % half_d; s = (offsets // half_d) % S` 等分解坐标 | 改用 per-position 处理 + `tl.arange` 向量化 D 维（L1.1/L3.1）；验证方法：检查编译后 IR 是否含大量 `scf.for` step-1 循环和 `memref.load %ptr[%c0]` 标量模式 |
| Adaptive TILE_S 导致编译器标量化 | 根据 S/D 动态选择 TILE_S（如 16/32/64），大 tile 被编译器拆成标量循环 | 固定 `TILE_S = 16`，用 grid splitting 解决负载均衡（L1.5/L3.3） |
| 忽略 broadcast stride | r1/r2 的 shape 为 `[B,1,S,D]` 时直接传 `t.stride()`，导致 kernel 内地址跳变错误 | Host 侧显式计算 broadcast stride（broadcast 维 stride=0，L1.2） |
| fp16/bf16 精度不足 | kernel 内直接以 fp16 做乘加减，relative error 超标 | kernel 内升 fp32 计算，存回前转回原精度（T2/L1.4） |
| Naive grid splitting 导致 idle core | `grid = (num_cores,)` + `for block in range(pid, num_blocks, num_cores)` 在 `num_blocks < num_cores` 时大量 core 空闲 | Uniform grid splitting（L2.2/L3.2）确保每个 core 处理连续且均匀的 block 范围 |

### §8.2 MoeComputeExpertTokens 陷阱

| 陷阱 | 表现 | 避免方法 |
|------|------|---------|
| atomic_add 竞争 | 某些 shape 延迟飙升到 0.04x | 改为 expert-parallel 无竞争计数（L1.1/L3.1） |
| torch.zeros 引入额外算子 | benchmark 中出现 `aclnnInplaceZero` | 使用 `torch.empty` + kernel 全覆盖写入（L3.3） |
| tl.cumsum fallback | 精度或性能异常 | 小数组直接用串行 for 循环（L1.2/L3.2） |
| 动态 grid 计算 | 增加 host 侧开销、编译器优化受限 | grid 固定为 `(num_expert,)` 和 `(1,)`（L1.4） |
| 跨 expert 循环计数 | 每个 block 做 64 次比较，指令膨胀 | grid 映射到 expert，每个 block 只比较一次（L1.3） |

### §8.3 MoeGatingTopKSoftmax 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| bool GM 缓冲区导致精度失败 | 使用 `torch.empty(..., dtype=torch.bool)` 作为 mask temp buffer，`tl.store`/`tl.load` 后 `tl.where` 条件类型变为 int8，导致大 f32 shape 精度错误 | 完全避免 bool GM 缓冲区，直接在寄存器中用 `tl.where` 更新 `curr` 向量（L1.2/L3.1） |
| num_cores 与 grid_size 不匹配 | `grid = (320,)` 但 `num_cores=40`，导致任务划分与实际 core 数不一致 | 始终令 `grid_size = min(batch_size, VEC_CORE_NUM)` 且 `num_cores = grid_size`（T1/L1.4） |
| tl.argmax 在 GM 往返后不可靠 | 将 `curr` store 到 GM temp buffer 再 load 回来，`tl.argmax` 返回错误索引 | 保持 `curr` 始终在寄存器中更新，不经过 GM（L1.3/L3.1） |
| host-side int32 cast 开销 | `finished.view(-1).to(torch.int32)` 引入额外 host 端计算 | 直接传入 `torch.bool`，kernel 内 `tl.load` 后直接使用（L3.3） |
| multibuffer/unit_flag 编译选项无效优化 | 盲目添加 `multibuffer=True, unit_flag=True` 期望提升内存密集型算子性能，本算子添加后性能从 0.8836x 下降至 0.8527x | 对 sort-topk 类算子（含迭代标量循环），multibuffer 收益被 top-k 迭代开销掩盖，不建议默认开启；应在 Phase 4 中实测验证后再决定（T6） |
| kernel 内使用 Python if/else 判断 None 指针 | `if finished_ptr is not None:` 在 Triton kernel 内编译失败或行为不可预期 | host 侧传入 dummy tensor（全 False），kernel 内直接 load 不使用条件判断（L1.6） |
| kernel 内使用 Python break | `break` 在 Triton kernel 中报 `unsupported AST node type: Break` | 使用 `tl.where` 条件赋值，或确保循环范围正确无需 break（L1.7） |
| 小 shape profiler 测量异常导致加速比失真 | framework 延迟 < 0.1ms 时，profiler 测量误差导致 speedup 异常高（2.5x~1251x） | benchmark 后过滤 framework 延迟 < 0.1ms 的 case，或设置 speedup 上限阈值（如 2.5x），重新计算几何平均加速比（L3.5） |
| 整行 load 向量长度超过 Ascend 稳定上限 | `tl.arange(0, NUM_EXPERTS)` 在 NUM_EXPERTS > 1536 时可能编译/运行正常但输出全 0（尤其非 pow2 如 1792） | 单 chunk 路径上限设为 1536；更大 E 使用带 mask 的 chunked load，CHUNK_E 按 2048/4096 自适应选择（L3.6） |
| LLVM_ROOT 环境变量未设置导致编译失败 | `clang++: symbol lookup error: undefined symbol: _ZN4llvm24createAutotuningDumpPassEv` | 设置 `LLVM_ROOT` 指向包含完整 libLLVM-17.so 的路径 |
| CANN 9.1.0 与 Triton 常量名不兼容 | `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` 在 CANN 9.1.0 中已重命名 | 修改 Triton 的 `npu_utils.cpp` 中的常量名为 `RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE`（一次性修复） |

### §8.4 AttentionSoftmaxWithSoftcappingAndDropout 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 编译器融合消除中间舍入 | 单 kernel `tanh(x/30)*30` 在 bf16 下 MERE 高达 22% | 拆为 div/tanh/mul 三个独立 kernel，GM round-trip 强制舍入（T5/L1.1） |
| tl.max axis=1 broadcast 错误 | `row_max = tl.max(t_fp32, axis=1); shifted = t_fp32 - row_max` 报 `Cannot make_shape_compatible` | 必须加 `[:, None]`：`row_max = tl.max(t_fp32, axis=1)[:, None]`（L3.3） |
| softcapping 子算子 dtype 不匹配 | `sc = 30.0`（Python float）或 `sc = tl.full((), 30.0, dtype=in_dtype)` 会导致 bf16/fp16 下除法精度不足 | `sc_fp32 = tl.full((), SOFTCAP, dtype=tl.float32)`，在 fp32 下做 div/tanh/mul，再 cast 回输入 dtype（T2/L1.2） |
| 循环引入后 UB overflow | 分核优化引入 `for block_id in range(pid, n_blocks, num_pids)` 循环后，大 K (BLOCK_N=1024) + ROW_TILE=16 触发 BiShengIR `ub over` 编译错误 | 收紧 UB budget 从 128KB 到 48KB，确保 `ROW_TILE * BLOCK_N * 4 <= 48KB`（L1.5） |
| grid 远超核数导致串行调度 | 朴素模式 `grid = ceil(num_rows/ROW_TILE)`，num_rows=32768 时 grid=2048，远超 48 核，NPU 串行执行 | `grid_size = min(natural_blocks, num_cores)`，每个 program 循环处理多块（T1/L3.1） |
| mask 元素污染归约 | padding 元素（load 时 other=0）参与 max/sum 导致结果错误 | max 前 `tl.where(mask, x, -inf)`，sum 前 `tl.where(mask, exp_val, 0.0)`（L1.4） |

### §8.5 LightningIndexer 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 用 torch.topk/sort/argsort 兜底 | 调用 aclnn kernel，判定作弊 | §6 L1.5 |
| 期望达到 0.8x/1.0x | 当前天花板 0.6152x | §6 L1.9 |
| 依赖 fused score+topk kernel | 当前后端编译失败 | §6 L1.7 |
| 重排/压缩 score 数据布局 | tie-break 会错 | §6 L1.6 |
| val 输出不做 bf16 RNE | 与 golden 差 0.05%–数千 | §6 L1.4 |
| 等值淘汰错方向 | 必须淘汰 index 最大者 | §6 L1.3 |
| PA_BSND device gather 默认启用 | 已验证无整体收益 | §6 L1.8 |

### §8.6 SparseFlashAttention 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 用 native SFA 作为 golden | ~1% bf16 系统误差导致正确实现也失败 | §7 L1.1 |
| host 侧拼接 q_rope/k_rope | 改变精度路径 | §7 L1.6 |
| 逐行 masked gather sparse row | 同地址 gather 被硬件串行化，11x 退化 | §7 L1.5 |
| softmax 用 2D bool where | 后端 vsel 开销大 | §7 L1.8 |
| QK K 维合并 | 改变 fp32 归约分组，0 容差失败 | §7 L1.9 |
| BLOCK_G=64 / BLOCK_T1=32 | UB / workspace 溢出 | §7 L1.7 |
| 忽略环境修复 sys.path | 可能加载错误 triton-ascend | §7 L1.10 |

---

## §9 相关记忆

- [[lightning-indexer-triton-gen]] — LightningIndexer 精度根因（tie-break + bf16 RNE）与历史优化路径
- [[lightningindexer-hybrid-native-topk-reverted]] — LightningIndexer 作弊回退记录
- [[lightning-indexer-hardware-opts]] — 原生 NPU 硬件优化机制（Triton 无法复现）
- [[lightning-indexer-spec]] — LightningIndexer 产品规格与 layout 语义
- [[sparse-flash-attention-hardware-opts]] — 原生 SFA 三层硬件优化、Triton 表达限制与残余差距分析
- [[sfa-golden-native-vs-purepytorch]] — SFA 精度基准选择
- [[sliding-fa-band-sparse-design]] — 区分：NSA 第三算子 Sliding FA（band 稀疏，无 gather 墙）
