---
type: Guide
title: FusionAttention 算子（attention） 实现手册
description: '算子类别: attention（标准稠密 MHA / FlashAttentionScore）。'
tags: [task.implementation, topic.attention, topic.compute.normalization, topic.compute.composite, topic.implementation.masking, topic.precision]
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
# FusionAttention 算子（attention） 实现手册

> 类别约束见 [transformer_training_constraints.md](../constraints/transformer_training_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 FusionAttention 算子（attention）

**算子类别**: `attention`（标准稠密 MHA / FlashAttentionScore）
**典型特征**: Q/K/V 同构，支持 BSH/BSND/BNSD/SBH，可选 pse/sink/mask/scale/keep_prob/inner_precise
**性能基准**: 61/61 pass；官方归档最佳 **0.8553x**（`opt_iter_46`）；实验最高 **0.9627x**（Exp32，Ascend910B，NPU7）

> ⚠️ **关键区分**：FusionAttention 的核心难点是**多 layout、多参数下的精度对齐**，其次才是性能。必须先保证 61/61 精度通过，再谈优化。

### §2.0 首次生成必读：为什么必须把主要框架写对

FusionAttention 是**精度敏感型**算子：layout 转换、score dot 精度、softmax 单 pass、mask 填充值、sink 处理、keep_prob clamp、inner_precise=2 全 masked 行等任意一条遗漏，都会导致大面积精度失败。首次生成若框架写偏（如三 pass softmax、先归一化 PV、错误 mask 值），后续局部修 bug 很难把性能救回。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 精度基准使用 `torch_npu.npu_fusion_attention` + keep_prob clamp

- golden 必须调用 `torch_npu.npu_fusion_attention`（其内部已对齐 CANN 实现）。
- `Model.forward` 中若 `keep_prob < 1.0` 必须先 `clamp(min=0.0, max=1.0)`，否则 case 42-45 等 keep_prob=0.85/0.9 场景会因 NPU dropout RNG offset 自增导致失败。
- 参考：[[fusion-attention-golden-precision-paths]]

#### L1.2 score dot 必须走原生精度

- fp16 输入：`tl.dot` 直接以 `tl.float16` 输入（`allow_tf32=False`），累加器 `tl.float32`。
- bf16 输入：同样以 `tl.bfloat16` 输入 `tl.dot`。
- **禁止**先将 Q/K cast 到 fp32 再 `tl.dot`：fp16-cast-dot 在极端 score 量级下会产生 ~1e-2 累加偏差，被 exp 放大后 softmax_sum 超 fp32 阈值。

#### L1.3 PV 必须延迟归一化

- 计算顺序：`E = exp(score - row_max)`（按输入精度落盘），然后 `output = (E @ V) / softmax_sum`。
- **禁止**先归一化 `probs = E / softmax_sum` 再 `probs @ V`：该路径 MERE 必然超标。
- 延迟归一化天然兼容 sink（分母含 exp(sink)）与全 masked 行（E=0, sum=Skv → out=0）。

#### L1.4 softmax kernel 必须单 pass 并正确处理全 masked 行

- `softmax_max`、`softmax_sum`、`exp` 必须合并为对同一行 score 向量的单次加载，**禁止三 pass 实现**。
- `BLOCK_KV` 必须根据实际 `Skv` 自适应：host 侧 `BLOCK_KV = triton.next_power_of_2(skv)` 作为 `tl.constexpr` 传入，禁止硬编码固定大值。
- 当 `Skv <= 512` 时，优先用 `BLOCK_KV = next_power_of_2(Skv)` 整行加载原地完成 max→exp→sum→div→store。
- 当 `inner_precise == 2` 时，全 masked 行（`masked_count == total_count`）必须满足：
  - `probs` 全置 `0`
  - `softmax_max` 输出 `0`
  - `softmax_sum` 输出 `Skv`
- 该约束是 61/61 精度的关键，漏掉会导致带 `atten_mask` / `inner_precise=2` 的 case 大面积失败。

#### L1.5 mask 填充值与 sink 处理

- mask 填充值必须是 `-FLT_MAX`（约 `-3.4028234663852886e+38`），**禁止**使用 `float('-inf')`。
- `sink` 必须作为 per-head bias 读入并参与 softmax：影响 `row_max`、`row_sum`，不能忽略。
- `HAS_SINK=False` 仅在 `sink is None` 时使用；只要 `sink` 张量存在，kernel 内必须加载并加到 score 上。

#### L1.6 pse 必须支持 broadcast

- `pse` 常见 shape 包括 `[1,8,1,128]`、`[B,N,Sq,Skv]` 等，kernel 内加载时必须按实际维度 broadcast。
- **禁止**假设 pse 与 score tile 同形；对 `[1,8,1,128]` 等维度必须显式处理 broadcast，否则触发 `MemRefLayoutAttrInterface` verify assert。

#### L1.7 多 layout 输入在 host 侧统一转 BNSD，禁止强制 `.contiguous()`

- BSH/BSND/SBH 统一在 host 侧转为 BNSD 后再传入 kernel；kernel 内只处理 BNSD，输出前再转回原始 layout。
- layout 转换使用 `transpose`/`view`/`permute` 即可，**不要**调用 `.contiguous()`。
- 无意义的 host 侧拷贝会显著增加 framework 延迟（实测去 `.contiguous()` 可提升 ~8% geomean）。

#### L1.8 输出 buffer 直接按目标 dtype 分配

- `attention_out` 直接分配为 `query.dtype`，避免最后 `.to(query.dtype)` 引入额外 cast kernel。
- `softmax_max` / `softmax_sum` 固定为 `torch.float32`。

#### L1.9 grid 并行度必须使用实际物理核数

- cube-heavy kernel 用 `cube_core_num`，vector-heavy kernel 用 `vector_core_num`，禁止硬编码核数。

#### L1.10 fused path 的 BLOCK_S 上限

- `BLOCK_S_FUSED` 可设为 32 的条件：
  - `dq <= 64 and skv <= 256`，或
  - `dq <= 128 and skv <= 128`
- 超出此范围必须回退到 `BLOCK_S=16`，否则 UB 溢出（编译器报 `hivm.hir.vcast` 或类似错误）。
- `BLOCK_S=32` 时不要同时放大 BLOCK_KV，UB 预算是硬约束。

#### L1.11 fallback / flash 单 kernel 路径的 BLOCK_N 限制

- `BLOCK_N` 可提升到 128 的条件：
  - flash_attention 单 kernel：`dv <= 64 and skv >= 1024`
  - score_kernel_flash / output_kernel_flash：`dv <= 64 and skv >= 1024`
- `BLOCK_M` 在 fallback/flash 路径中不要超过 128（BLOCK_M=256 实测 UB/accumulator 压力过大，性能回退）。

#### L1.12 性能天花板需管理预期

- 当前 Triton-Ascend 实验最高约 **0.9627x**（Exp32，61/61 pass，Ascend910B，NPU7）。
- 官方归档最佳为 **0.8553x**（`opt_iter_46`）。
- **1.0x 尚未达成**；0.90x 以后收益急剧收窄，瓶颈从单一 kernel 算力转向 mix-mode 同步、UB 预算和 MTE 往返。
- 若 `target_speedup >= 1.0`，必须在 report 中说明天花板，并记录已尝试方向。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 四 kernel pipeline（默认路径）

```python
# 1. score_kernel (cube): Q@K^T * scale (+pse)(+sink)(+mask) -> Score f32 [B,N,Sq,Skv]
# 2. softmax_kernel (vector): 行级 online softmax -> E + softmax_max + softmax_sum
# 3. dropout_kernel (vector): 仅 keep_prob < 1 时触发
# 4. output_kernel (cube): (E @ V) / softmax_sum -> Out
```

- score 输出 f32，E 按输入精度（fp16/bf16）落盘。
- softmax 必须单 pass 同时产出 max/sum/E。

#### L2.2 单 kernel flash 路径（大 D 回退）

```python
# 当 dv > 128 或 dq > 128 时，四 kernel pipeline 的中间 E-buffer [B,N,Sq,Skv] 过大
# 使用 flash_attention_kernel: 单 kernel 内做 online-softmax，不物化完整 Score/E
flash_attention_kernel[(grid,)](q, k, v, out, softmax_max, softmax_sum, ...)
```

- 该路径受 UB 限制，D=128 时通常只能 BLOCK_M=64。
- 小 D（<=64）大 Skv 可尝试 BLOCK_N=128 提升性能。

#### L2.3 Host 侧分派决策树

```python
if dv > 128 or dq > 128:
    # 单 kernel flash，避免 [B,N,S,Skv] 中间 buffer
    flash_attention_kernel(...)
elif use_fused:
    # fused score+softmax，省掉 Score HBM round-trip
    score_softmax_kernel(...)
    output_kernel(...)
else:
    # 原四 kernel pipeline
    score_kernel(...) + softmax_kernel(...) + output_kernel(...)
```

### §2.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 Host 侧无拷贝 layout 转换

```python
def _to_bnsd(tensor, layout, head_num):
    if layout == "BNSD":
        return tensor
    if layout == "BSND":
        return tensor.transpose(1, 2)
    if layout == "BSH":
        b, s, h = tensor.shape
        d = h // head_num
        return tensor.view(b, s, head_num, d).transpose(1, 2)
    if layout == "SBH":
        s, b, h = tensor.shape
        d = h // head_num
        return tensor.view(s, b, head_num, d).permute(1, 2, 0, 3)
```

#### L3.2 延迟归一化 output kernel

```python
# kernel 内：
acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
for n_start in range(0, Skv, BLOCK_N):
    v_tile = tl.load(...)
    e_tile = tl.load(exp_buf + ...)
    acc += tl.dot(e_tile, v_tile, out_dtype=tl.float32)
out = (acc / softmax_sum[:, None]).to(out_dtype)
tl.store(out_ptr, out, mask=...)
```

#### L3.3 fused score_softmax_kernel 的 BLOCK_S 选择

```python
BLOCK_S_FUSED = 32 if ((dq <= 64 and skv <= 256) or (dq <= 128 and skv <= 128)) else 16
```

#### L3.4 softmax 行级单 pass 模式

```python
val = tl.load(score_row, mask=kv_mask, other=-float('inf'))
row_max = tl.max(val, axis=0)
exp_val = tl.exp(val - row_max)
row_sum = tl.sum(exp_val, axis=0)
# 存储 E, max, sum
```

#### L3.5 突破 0.90x 的关键组合（Exp32）

- **output kernel 放大 BLOCK_N_OUT=256**：当 `dv <= 128 and skv >= 512` 时启用，减少 KV 循环次数。
- **fused path 放宽 `num_groups=1`**：在 D=128 / S=256 等场景禁用 flash 路径改走 fused，配合宽松 grid 分组降低 AIC-AIV 同步密度。
- 两者组合在 NPU7 上达到 **0.9627x**（实验最高）。
- 注意：该组合对 UB 预算敏感，D=64 大 Skv 下仍可能 overflow，必须做边界校验。

### §2.4 性能基准

| 版本 | 日期 | geomean | 精度 | 关键改动 |
|------|------|---------|------|---------|
| opt_iter_46 | 2026-07-21 | **0.8553x** | 61/61 | fused 路径 BLOCK_S=32 扩展到 D=128 小 Skv；官方归档最佳 |
| **Exp32** | 2026-07-27 | **0.9627x** | 61/61 | output BLOCK_N_OUT=256 + fused path 宽松 num_groups=1；**实验最高** |
| **fusion-search R8** | 2026-08-30 | **1.5933x** | 61/61 | 连续 kernel 结构重构（5dd2b06a→64319fd0）；G017 多 head 摊销 + G023 反向倾斜；C2 停止 |

### §2.5 融合搜索基因编码存档（3_FusionAttention, 910b2, 2026-08-30）

> 来源：`triton-optimizer-multi-agent` fusion-search 模式，R1 构建初始基因池（OKF 加载 8 基因 + B1/B2 融合方案），R2+ 遗传进化（选择/交叉/变异/profiling 驱动新增），10 轮 C2 停止。收敛 1.2573 → **1.5933**。torch 基线 = `npu_fusion_attention` msprof Task Duration geomean **33.5252us**（61 case）。full 29 基因定义见任务 `opt-round-10/gene_pool.json`。

**收敛路径**：R1 1.2573（5dd2b06a，OKF 初始基因）→ R2 1.2598 → R3/R4 no-op 1.2955/1.2953（MCTS virtual-root 复测同一 kernel）→ **R5 1.4494（e1de0c0c）** → R6 1.5708（88ce0e4e）→ R7 1.5856（cc0dfd02）→ **R8 1.5933（64319fd0，final）** → R9/R10 no-op 1.5765。关键跃迁均在**连续 kernel 结构重构**（kernel 面积 14962→22719→37732→38609B）。

#### active 基因（11，收敛后保留）

| 基因 | 类别 | fitness | 编码动作 | 实测结论 |
|------|------|---------|----------|----------|
| G001 | compute | 1.5765 | tl.dot 承载 QKᵀ 与 PV，原生 dtype + fp32 累加；禁逐维外积 | 走 Cube 低精度高吞吐；fp32 输入同样 tl.dot |
| G002 | tiling | 1.5765 | `BLOCK_D = ceil16(head_dim)`，不用 next_pow2 | D=96/160/192 免 30~60% tile 浪费 |
| G003 | tiling | 1.2598 | scores tile 面积 `BQ×BKV ≤ 8192`（fp32 32KB） | 防 UB 192KB 溢出级联假失败 |
| G004 | mask | 1.5765 | 掩码用 `-3e38` constexpr（非 -inf）+ 显式 tl.where 排除归约维 padding | 防整行掩码 NaN 与幽灵列混入 softmax 分母 |
| G005 | compute | 1.5765 | online softmax m/l/acc 滚动 + KV 分块循环，不物化完整 scores | 目标函数 = 循环内向量算子数 × KV 迭代数 |
| G006 | structural | 1.2598 | grid 固定核心数 + grid-stride | 减调度开销；小 case 配合 block 收缩 |
| G007 | memory | 0.0868 | block 收缩到实际 shape `min(BLOCK, ceil(shape))` | 小 case 免 padding 空转（实测最小 case 空转 85 倍） |
| G008 | pipeline | 0.0868 | @triton.autotune 搜 BLOCK_Q/BLOCK_KV（key=Sq,Skv,head_dim） | 支持 block/multibuffer，不支持 num_warps/num_stages |
| G011 | structural | 1.2142 | pse/mask 合入 dot 前预处理 | 减 KV 循环内 UB 依赖向量算子 |
| G017 | structural | 1.5765 | 多 head 并行 / GQA kv-group 摊销：program=(b,kv_head,sq_block) 内循环 query head | K/V 加载与索引摊销 4×（GQA group=4） |
| G023 | tiling | 1.5654 | d64_s512 反向倾斜：BQ=128/BKV=128（dot M 64→128） | Cube 半利用 0.31× 缓解；R8 胜者核心基因 |

#### 证伪基因（18，同窗 A/B 判否）

| 基因 | 类别 | 方向 | 证伪结论 |
|------|------|------|----------|
| G009 | compute | dual-dot/split-K（head_dim≤64 拆双 dot） | Cube 利用不升，-6~9% 回归 |
| G010 | compute | tl.exp2 快速近似 | 精度 61/61 过但 -2.7% 慢；tl.exp 已降 exp2 |
| G012 | compute | scale 预乘进 Q（无 pse） | 额外 scaled-Q buffer 溢出 192KB UB（30/30 case） |
| G013 | structural | split-P / head 并行 | 零提升（<1% 噪声）；瓶颈是标量指令链非并行 |
| G014 | memory | K-tile format-conversion 摊销 | +3.91% 慢；format-conv 非瓶颈 |
| G015 | compute | 原位 scale fold（G012 修正版） | 3 变体全 UB overflow 44~50/61 |
| G016 | compute | defer acc rescale / batch rowmax | 两 pass +32% 慢；rescale 非主导 |
| G018 | pipeline | KV 循环 num_stages=3/4 | +2.4% 慢；prefetch 不解决 per-program 标量链 |
| G019 | memory | block_ptr 增量地址推进 | +0.26% 中性；make_block_ptr 布局/编译缺陷 |
| G020 | structural | QBLOCK_MERGE 深度 3/4 | depth-3 回归 34-46%，depth-4 UB 溢出；深度已耗尽 |
| G021 | tiling | KV 迭代削减（BLOCK_Q↓/BLOCK_KV↑） | +0.01% 中性；kvIt 减少被 G017 摊销抵消 |
| G022 | structural | split-KV 跨 program（partial+combine kernel） | +26~40% 回归；24 核 grid 上限使 program 翻倍无益 |
| G024 | pipeline | FA_NW 1/2/4/8 × FA_NS 2/3 扫描 | 确认 cc0dfd02 无回归；未突破 |
| G025 | compute | QBLOCK_MERGE==2 Cube/Vector 相位批处理 | 1.4957（低于 1.5483 阈值）判否 |
| G026 | structural | grid 超订 total_tasks | cube-bound 上净中性 |
| G027 | pipeline | G023-B kernel NS=2/3/4 重测 | cube-bound 4-iter 循环中性 |
| G028 | pipeline | G023-B kernel NW=4/8 | bracketed R3 A/B + 交织 diag 中性 |
| G029 | compute | 2-KV-block batched rescale（unroll-2） | +47.3% 回归；cube-bound，rescale 链非墙 |

#### 结构性上限（勿再搜索方向）

- **d64_s512 Cube 半利用 0.31×**（dot M 收缩维 K=64），d512 M=32 微块；simulator 实测 MMAD 12% + Cube↔Vector 串行。
- 参数/基因级调优在 R7 后收敛；R8 单点突破（G022/G023）后 R9/R10 平坦。残余 headroom 需结构性算法变更（round-10 G027 证明不可行）。

---
