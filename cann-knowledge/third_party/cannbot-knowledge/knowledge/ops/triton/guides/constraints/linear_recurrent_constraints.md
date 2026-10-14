---
type: Guide
title: linear_recurrent 类算子设计约束（状态递推）
description: 矩阵状态递推、无 softmax 主链算子的判别与硬约束（状态驻 UB 单融合 persistent kernel 等）；与 recurrent
  家族 host-loop 的分工。
tags: [task.design, topic.compute.composite, topic.precision, topic.implementation.memory_management, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-26T08:27:20Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# linear_recurrent 类算子设计约束（状态递推）

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`）；本卡只列 linear_recurrent 类别特有条款。

本卡负责**矩阵状态递推、无 softmax 主链**的线性复杂度注意力算子：锚定算子 `FusedRecurrentGatedDeltaRule`（fla 系 fused_recurrent gated delta rule），同类 Mamba/SSM/RetNet 状态递推变体可复用。核心优化哲学是**单融合 persistent kernel：`[K,V]` fp32 状态矩阵驻留 UB + flag constexpr 特化 + device 内串行 t 循环**。

- **证据基础**：锚定算子 50 case（fp32/fp16/bf16 × 全部 flag 组合），精度 **50/50 通过**（2026-08-24 独立复验确认）；加速比声明已撤回（测量口径事故，见 §5.1）。硬件 Ascend910_9372（910B 家族）/ CANN 9.1.0 / triton-ascend 3.2.2。

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 2 算法骨架、Layer 3 关键代码与性能实测，**命中对应算子时必须一并加载**。

- [`operators/linear_recurrent__fused_recurrent_gated_delta_rule.md`](../operator_playbooks/linear_recurrent__fused_recurrent_gated_delta_rule.md) — FusedRecurrentGatedDeltaRule 算子（Gated Delta Rule）实现手册：单融合 persistent 递推 kernel、状态 1D flat 写回与精度/性能实测。

---

## §0 适用范围与算子分类

| 算子 | 子类标签 | 计算特征 | 优化哲学 |
|------|---------|---------|---------|
| FusedRecurrentGatedDeltaRule | `linear-recurrent` | 门控衰减（g/gk/gv 三种 gate 组合）+ delta rule rank-1 状态修正 `S = decay∘S + k⊗((v−Sᵀk)·β)`，每 `(b, vh)` 一套 `[K,V]` fp32 状态，T 步串行递推 | 单融合 persistent kernel：状态驻留 UB + flag constexpr 特化 + device 内串行 t 循环 |
| Mamba/SSM、RetNet 等状态递推变体 | `linear-recurrent`（待验证） | 状态递推同构（`S = a·S + k⊗v` 族） | 可复用本卡；首验先小规模 |
| 结合律重排（无递推） | 待归档 | `(QKᵀ)V → Q(KᵀV)` 型重排 | 本卡递推约束不适用，勿混用 |

### §0.1 判别特征（决定用不用本文件）

1. **无 softmax 主链**：attention_index 行 12 判据——主链是**状态递推**或**结合律重排**，不是 `softmax(QKᵀ)@V`；
2. **状态递推有真数据依赖**：每 `(b, vh)` 维护一份状态、T 步串行 `S_t = f(S_{t-1})`；
3. 状态是**矩阵**（`[BK,BV]`，fp32 ≤32KB）而非向量——这是选 persistent kernel 而非 host-loop 的决定性判据。

### §0.2 与 `recurrent` 家族的分工（⚠️ 别拿错卡）

- **`recurrent` 家族**（LSTM/GRU/RNN，[recurrent_constraints.md](recurrent_constraints.md)）：状态是 `[B,H]` **向量**、无大 tile 可驻留 ⇒ 哲学是 **host-loop 单步 kernel**；
- **本卡**：状态是 `[BK,BV]` **矩阵**（fp32 ≤32KB）可整块驻留 UB ⇒ **单融合 persistent kernel + device 内 t 循环**更优。
- 两卡「时间维串行、program 持完整状态」结论一致，**分歧仅在发射粒度**：host-loop 仅当状态是向量、无 tile 可驻留时才是正解。

> ⚠️ **禁止套用 [flash_attention_constraints.md](flash_attention_constraints.md) 的 online softmax / KV 分块**——本类无 softmax，状态递推是唯一主链，FA 的滚动 `m/l` 约束整套不适用。

---

## Layer 1 设计约束（首次生成就要全部满足）

> **本节 9 条（L1.1~L1.9）全部是硬性边界。** 带 ★★★ 的 **L1.1**（状态驻留 UB 单融合 persistent kernel）是架构基石；★★ **L1.2**（flag 全 constexpr）与 ★ **L1.3**（masked 2D store 静默丢格）是精度命门。

### L1.1 ★★★ 状态矩阵驻留 UB，单融合 persistent kernel（强制）

- **必须**: 1D `grid = (B*VH,)`，每 program 负责一个 `(batch, value_head)`（GQA: `h = vh // GQA_R`，q/k 用 h、v 用 vh），状态 `[BK,BV]` fp32 驻留寄存器/UB，`for t in range(T)` 在 kernel 内串行递推，一次发射完成全部计算。
- **Why**: 状态 ≤ `128*64*4B = 32KB ≪ 192KB UB`，驻留免每步读写 GM；host-loop 逐步发射被 launch 开销淹没。
- **How to apply**: 见 playbook Layer 2 骨架；**禁止**把时间维展平进 grid 跨 program 并行（递推真数据依赖，RAW），**禁止**循环内 `break`（与 trip-count 相关的编译约束）。

### L1.2 ★★ 运行时 flag 全部 `tl.constexpr` 特化（强制）

- **必须**: 所有运行时开关（gate_mode 位 1/2/4、use_gate_in_kernel、use_bias、use_initial_state、output_final_state、use_qk_l2norm、use_beta_sigmoid、allow_neg_eigval、state_v_first 等）以 constexpr 参数传入，编译期消歧。
- **Why**: Triton 按 flag 组合各自编译缓存；单 case 内组合固定，无重复编译开销；不同组合各有最优代码路径。
- **How to apply**: 见 playbook Layer 3 constexpr 调用侧示例；**禁止** kernel 内用运行时标量 `if` 处理 flag——标量分支阻碍向量化。

### L1.3 ★ masked 2D state store 静默丢单元格（头号精度陷阱，强制）

- **症状**: `tl.store(ptr + 2D偏移, tile, mask=2D_mask)` 写 `[BK,BV]` 状态时，**被 mask 掉的单元格不写且不报错**，读回呈**非确定性脏值**；典型表现 output 全对（~9.5e-7）而 `final_state` 错——实测 **20/25 失败 case 与 `output_final_state=True` 相关**。
- **Why**: Triton-Ascend 的 2D masked store 对 unmasked 单元格静默丢弃（非确定性脏值）。
- **How to apply（两案均已验证）**:
  1. **1D 连续 flat 布局**（终版采用）：kernel 内按 `[BK*BV]` 写（flat index = `k*BV + v`），host 侧 `view(B,VH,BK,BV)` → 裁剪 `[:, :, :K, :V]` → 按需 `transpose`；
  2. **计算出的行指针逐行 store**（Variant B2）。
- **确诊手法**: 输出 buffer 预填 `torch.zeros`，错误格精确为 0 ⇒ 未写（本陷阱）；非 0 ⇒ 写错值（另有根因）。

### L1.4 边界 mask 用 fp32 向量比较（强制）

- `m_k = offs_k.to(tl.float32) < K.to(tl.float32)`——整数比较触发标量降级（scalar fallback），fp32 向量比较保向量核。

### L1.5 数值细节与 fp32 累加（强制）

- **必须**: 状态、门控衰减、归约全程 fp32；输入读入即 `.to(tl.float32)`，仅最终 store cast 回目标 dtype。
- softplus 对齐 `F.softplus(threshold=20)` 稳定形式：`tl.where(x > 20.0, x, tl.log(1 + tl.exp(x)))`。
- `F.normalize(p=2, eps=1e-6)` 对齐：`x / max(||x||₂, 1e-6)`（`tl.maximum` 防 0 除）。

### L1.6 kernel 返回前 host 侧同步（多 case 场景强制）

- host 侧 `torch.npu.synchronize()`：verify/benchmark 连续跑多 case 时，防止上一 case 的 kernel 异步写回与下一 case 的内存分配/复用竞争。单 case 无害，可常开。

### L1.7 可选输出指针不可传 None（强制）

- `output_final_state=False` 时 `ht_ptr` 仍需实参：传 1 元素占位张量（Triton 不接受 None 作指针），并用 constexpr 分支（L1.2）保证 kernel 不访问该指针。

### L1.8 ★ BK=64 触发 PlanMemory 编译失败（UB 红线）

- **症状**: 状态 + 多输入 tile 的 UB 需求超限时 bishengir 直接 `PlanMemory Failed`。
- **How to apply**: 需对 K 维分块（子 tile 递推）或降 BK。`[BK,BV]` 驻留与 BK 上限的边界本探索未完整扫描——遇编译失败先做 UB 记账再选 tile，不要盲目重试。

### L1.9 ★ verify/benchmark 必须设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（环境红线，强制）

- **症状**: 未设该变量时 torch 双 npu 后端注册冲突，报 `RuntimeError: Two accelerators cannot be used at the same time in PyTorch: npu and npu`。若发生在 impl/框架加载或运行期，会产生**大面积伪失败**——包括看似真实的数值错位（3/616 违例、1e3 量级脏值）、NaN、乃至整组 case 报错，极易被误判为代码 bug（2026-08-24 实例：同一份 50/50 正确的代码被误判为 41/50 失败）。
- **必须**: 所有 verify/benchmark 运行环境显式 `export TORCH_DEVICE_BACKEND_AUTOLOAD=0`（本卡 L1.6 的 `torch.npu.synchronize()` 路径同样依赖它）；配套 `LD_LIBRARY_PATH` 指向实际使用的 torch/torch_npu 构建。
- **教训**: 排查"数值失败"前先确认环境变量，再做代码归因；跨会话采纳/复核代码时双方必须对齐环境配方。

---

## §4 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| masked 2D store 写状态后 final_state 错、output 却对 | mask 掉的单元格静默不写，读回为脏值 | L1.3 flat 1D store + host reshape；zeros 预填确诊 |
| BK=64 编译失败 | UB 需求超限 → PlanMemory Failed | L1.8 先 UB 记账，K 维分块或降 BK |
| 整数边界比较 | 标量降级 | L1.4 fp32 向量比较 |
| softplus 大输入溢出 / 与参考不齐 | 未对齐 F.softplus threshold | L1.5 稳定形式 |
| 多 case 连续 verify 结果随机错 | kernel 异步写回与下 case 分配竞争 | L1.6 synchronize |
| 可选输出传 None | Triton 不接受 None 指针 | L1.7 占位张量 + constexpr 分支 |
| 时间维进 grid 并行 | 递推 RAW | L1.1 串行 t 循环 |
| forward 调 torch_npu 融合接口 | 本类任务用户约束 + AST 门禁 | 全部计算入 kernel |
| state_v_first 初态/末态错位 | 布局 `[V,K]` vs `[K,V]` 索引交换遗漏 | 加载与写回两侧都按 flag 交换（playbook Layer 2 骨注） |
| "Two accelerators: npu and npu" 大面积伪失败（含数值脏值/NaN） | 未设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，双 npu 后端注册冲突 | L1.9 环境红线；归因代码前先查环境 |
| speedup 声明虚高（如 5.75×） | framework 基线测量被拉慢（频率锁定异常/卡忙/计时段污染），impl 实际更慢 | §5.1：framework 绝对延迟与历史口径交叉校验；同口径复测后再声明 |

---

## §5 已验证无效/排除的尝试（别重复）

### §5.1 ⚠️ 5.7527× 加速比声明撤回（2026-08-24，测量口径事故）

- **现象**: 本类曾声明 geomean 5.7527×（复验 5.8651×）达标。独立复测（同代码 impl md5 `5454f179`/同参考 sha `321a37eb`/同环境配方：卡 1 + `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + 系统 torch 栈，warmup=5/repeats=50）仅 **0.143×**，且 impl 耗时与原测量一致——分歧全在 framework 基线（原测量慢 ~3-12×；case 5 报 0.4897ms，其余口径均为 0.038~0.040ms）。
- **疑似根因**: 复验所用 benchmark.py 的 NPU 频率锁定（lock_npu_frequency）在 framework 计时段失效或降频；或计时段卡忙/环境残留（`ASCEND_LAUNCH_BLOCKING=1` 泄漏会把逐算子 launch 变同步，对多算子 torch 参考惩罚最大）。
- **规避**: ① 任何 speedup 声明前，将 per-case framework 绝对延迟与历史/跨会话口径对照（本算子 case 1/5 应为 ~0.012/~0.040ms 量级）；② benchmark 环境与调试环境严格分离，`unset ASCEND_LAUNCH_BLOCKING`；③ 频率锁定失败按 error 处理而非 warn。

- **标量头基准指针的状态写回**（Variant A）：无效。
- **指针行索引写法的 row-wise store 首版**（Variant B 原始版）：编译错误；修正为"计算出的行指针"后可用（即 L1.3 方案 2）。
- **host 侧清理迭代**（输入布局预处理精简，迭代 #17）：无 speedup 提升，保留基线。
- **调试提示**: kernel launch 卡顿时用 `ASCEND_LAUNCH_BLOCKING=1` 同步复现定位；torch inductor compile_worker 的 traceback 属探测噪音，可忽略。
