---
type: Guide
title: FA 精度闸门与测量口径
description: Flash Attention 的精度验收闸门与性能测量口径；不过闸门谈性能无意义。
tags: [task.implementation, topic.attention, topic.compute.normalization, topic.precision, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# FA 精度闸门与测量口径

> 类别约束见 [flash_attention_constraints.md](../constraints/flash_attention_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §6 精度闸门（先过闸门，再谈性能）

- 完整四条逐位契约、1-ulp 匹配率诊断法、误差放大链算法：
  `@../../../ops/triton-precision-debug/references/attention-lowprec-contract.md`
- **判据推论**：若失败集合**严格与 dtype 相关**（如"全部且仅仅是 fp16/bf16 case"），基本可直接判为**算术路径不匹配**，不用去搜 tiling。
- **判据推论（参数版）**：若失败集合**严格按某个运行时参数划分**（如"全部且仅仅 `sparse_mode=4` 的 case"），先探该参数在 golden 里的真实语义（§6.1），**不要去改 kernel 数学**——多半是 golden 行为与文档/直觉相反。
- **正式结论一律走 `verify.py`，不改阈值、不用 `--verify_not_required`。** 扫参阶段可用单 case 脚本秒级淘汰明显错误的变体。

### §6.1 ⭐ CANN 语义探针先行（golden = `npu_*` op 时**必做**）

`torch_npu.npu_xxx` 的 golden 是**真实 op 行为**，可能与文件内 torch fallback（docstring）**相反**。
每个参数 5 分钟独立探针：固定其他输入，只改该参数，比对三输出 diff。
**判别信号**：失败集合严格按某参数划分 → 探该参数，不要改 kernel。

已实测（FusionAttention `npu_fusion_attention`）语义表：

| 参数 | 文档/直觉语义 | **实测真实语义** |
|---|---|---|
| `sink` | per-head bias，加到 scores 上 | **attention-sink 列**：`m_init = sink[n]`、`l = 1`、`exp(sink-m)` 进 softmax 分母（无对应 V 列） |
| `pse` | — | **先加再乘 scale**：`(QK + pse) * scale`，不是 `QK*scale + pse` |
| `atten_mask` | — | **True = 掩蔽**；掩码值 = `-FLT_MAX`、`m_init=-FLT_MAX`；全掩行 `smax=-FLT_MAX`、`ssum=Skv`、`out=mean(V)` |
| `sparse_mode` + `pre/next_tockens` | 按 band 窗口裁剪 | **0/2/3/4 全被忽略 == 全可见**（此前提下禁做 §3.2 区间收缩） |
| `inner_precise` 0/1/2 | 精度档位 | **对输出无影响** |
| `keep_prob < 1` | dropout | **golden 不可复现**（NPU 内部 RNG，同输入两次输出 diff~4.13）；任务层强制无 dropout |
| GQA `nkv` | — | `nkv = nq * H_kv // H_q`（按 head dim 比值） |

- ⚠️ 探针要在**直接调 `npu_xxx` 的环境**里做，不要信任静态代码阅读——实测 sink 忽略、mode4 忽略窗口、atten_mask 掩蔽方向都是探针实锤的，与文件内 fallback 完全不同。

### §6.2 CANN 输出契约（`npu_*` op 的返回值形状必须先钉死）

`npu_fusion_attention` 返回 **7 元组**：
`(attention_out, softmax_max, softmax_sum, reserved, seed, offset, mask_length)`，其中：

| 输出 | 形状 / 值 | 从零生成最容易错 |
|---|---|---|
| `attention_out` | 按输入 layout 的 `[B,S,H]` / `[B,N,S,D]` 等 | 布局错位 → 全错 |
| `softmax_max` | **`[B,N,Sq,8]`（尾维 pad 到 8，8 列同值）** | 按 `[B,N,Sq]` 输出 → shape 不匹配直接 fail |
| `softmax_sum` | **`[B,N,Sq,8]`（同上）** | 同上 |
| `reserved` | `torch.empty(0)` | 非空 → shape 不匹配 |
| `seed` / `offset` / `mask_length` | 恒 `0` | — |

只读 `Model.forward` 签名写 kernel 会按 `[B,N,S]` 输出 softmax_max/sum，verify shape 不匹配 fail 且完全查不到原因。**先按本表钉死输出契约，再写 kernel。**

### §6.3 精度契约会倒逼结构

当参考要求"先用最终的 `l` 归一化再舍入"时，online softmax 只能拿到滚动的 m/l ⇒ 必须多趟（见 §3.3）。
各步实测收益：三趟 → 单块单趟 **+21.9%**；多块三趟 → 两趟 **+10.1%**。

> ⚠️ **首选是延迟归一化而非"在线两趟"**（0811/3 61/61 vs 0818/3 57/61，见 §3.3）：
> 延迟归一化（softmax 写未归一化 E，`Out=E@V/sum`）从结构上绕开滚动 alpha 舍入，是 FP 精度全过的根本方案。"在线 softmax + two_pass"只多算一次 max，滚动 alpha 舍入仍在，fp16 大 S case 仍会 MERE 超阈。**失败后若 MERE 卡在 1e-3 附近而非 ~1.8e-4，结构没对齐，改延迟归一化，不要用"golden 噪声"当台阶。**

---
---

> 类别约束见 [flash_attention_constraints.md](../constraints/flash_attention_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §7 测量口径（不做这一步，上面所有数字都是噪声）

### §7.0 ★★★ 官方 `speedup_vs_torch` 是**几何平均**，优化要盯最慢的用例

`benchmark.py` 的 `_geomean_speedup()` 取的是**逐 case speedup 的几何平均**，
不是总时间比、也不是算术平均。这一条决定了优化的**投入方向**：

几何平均在对数空间 ⇒ **把一个 0.5x 的用例修到 1.0x，收益等于把一个 2x 的用例提到 4x。**
盯着已经很快的大 shape 继续调，对总分几乎没有贡献。

实测杠杆（op54，50 个用例，其中 16 个 <1.0x、均值 0.753）：

| 情形 | 几何平均 |
|---|---|
| 现状 | 1.7413 |
| 算术平均（**不是**官方口径，会严重高估） | 2.3550 |
| **把 16 个 <1.0x 全提到 1.2x** | **2.0422** ✅ |
| 提到 1.4x | 2.1830 |
| 提到 1.6x | 2.3382 |

⇒ **达标路径不是"整体再快 15%"，而是"把垫底的那批修到 1.2x 以上"。**

操作步骤：从 `perf_result.json` 的 `per_shape_results` 按 `speedup_vs_torch` 升序排列 → 只看 `<1.0x` 的那批找共同特征（典型是中小 shape、`B*N` 大、`S` 小）→ 为这批特征单独开一条分派路径，而不是改全局参数（全局改会同时动到已很快的用例，往往净负收益）。

⚠️ 慢用例常常**不是**大 shape。op54 里最慢的是 `B=16,S=256`（0.535x）和 `B=1,S=1024`（0.641x，仅 0.5M 元素），瓶颈是分块与同步开销占比高，而 torch 在这些形状上走的是高度优化的批量 matmul。

### §7.1 环境闸门与噪声

- torch 参考代码恒定，其时延是天然环境探针。**必须用双探针**（全量几何平均 + 大 shape 子集）：实测某轮全量探针在带内、大 shape 探针却比基线高 31%，该轮大 shape 数据整体失真。
- **噪声水平**：20 case 单次 ±2%；同一份代码复测两次可差 2.2%；10 case 基线包夹（base→A→base）可到 ±0.3%。
  ⇒ **≤3% 的改动必须用包夹协议复测**，否则会把噪声当收益采纳（实测有 4 个候选栽在这里）。
- ⚠️ **闸门必须双侧——异常变快和异常变慢一样会伪造结论**。实测两次拦截：机器高负载时 torch 时延 +85%（若不拦会把污染数据当成 +3% 提升）；机器临时空闲时 torch 时延 **−40%**（若不拦会把一个**回归**当成 **+52.4% 的最大突破**合入）。只设上界差点收下假突破。
- **A/B 必须对当前 best 做**，不要拿历史数字当参照——实测因此把一个回归误判成大幅提升。
- ⚠️ **profiler 的 `operators` 聚合 ≠ 几何平均**：前者是跨 case 的总耗时（大 case 占大头），后者每 case 一票。实测出现过"attention kernel 聚合耗时降 7%、几何平均反而下降"——改动让大 case 变快、小 case 变慢。**几何平均下"别让小 case 变慢"与"让大 case 变快"同等重要。**
- **"无效"的改动未必中性**：某次手工外提常量张量测下来是噪声（判定无效），但它把该张量提到 kernel 顶层、**全程占 32KB UB**；等 UB 成为瓶颈时这笔账才浮现。
- 比值型指标不能自证有效：分子分母同步变慢时 speedup 反而"看起来变好"。
- ⚠️ **`torch_npu.profiler` 会在部分 device 上提取不到数据**：表现是 `msprof 旧格式回退解析为空` / `SQLite Error: no such table: TASK`，**大面积假失败**（实测某次 57/61 的 benchmark 全因此，代码本身无问题）。**必须 `torch.npu.set_device(<目标 NPU>)` 后再跑 benchmark**（benchmark.py 用默认 `torch.device("npu")`=device 0，不改就会中招）。判别：framework 与 impl 两侧 profiler 都解析为空、且 case 全部失败 → 先查 device/profiler，**不要改代码**。

### §7.2 ⚠️ benchmark 时延口径（写报告时必须注明）

`benchmark.py` 按 **kernel Name 分组**，同名 kernel 一次 forward 内发射 L 次**只被计成 1 次**（低估 L 倍）。

- ✗ **禁止**为迎合口径把多次发射合并成同一个 Name（用 `MODE: tl.constexpr` 分支合并 kernel）——**不改变任何真实性能，纯粹利用口径缺陷**。
- ✗ 也**不要**因为口径而放弃真实优化（如三个投影融合成一次 GEMM 是真实收益，但会让口径分数变差）。
- **做天花板估算前先确认分数口径**：实测某算子 benchmark 口径 6.24 而真实设备时间 **9.24x**，不确认会把接近目标的实现误判为差 40%。

### §7.3 ⚠️ 历史结论会随编译器 / CANN 版本失效——成功与失败都会

| 改动 | 旧编译器 | 新编译器 |
|---|---|---|
| KV 循环整块/尾块拆分 | **+4.5%** | **−11.7%**（撤销它反而更快） |
| `BLOCK_SQ=256` | **0/50 全线编译失败** | **编过且 +3.2%** |
| 原始基线代码（同一份） | 0.1052 | **0.8390**（快 8 倍） |

> **规则：换编译器 / CANN 版本后，全部结论作废，必须重新标定基线并重扫参数。不要查历史失败清单。**
> 这条与 L1.4（参考实现算术路径一变、精度结论全废）是同一类风险的两个来源。

### §7.4 编译器已经做了的事，不要手工做

手工循环不变量外提、手工 buffer 复用、手工删除常量张量——**三次实证全部 no-op**（其中一次还有 UB 副作用）。
**UB 手算模型（把张量尺寸相加）预测不了真实边界，边界只能实测撞出来。**

⚠️ **UB 超限在 MLIR 层的表现要与精度失败区分**：

```
'hivm.hir.vcast' op Unsupported op for finding the root alloc
'hivm.hir.vexp'  op Unsupported op
[ERROR] Failed to run BiShengIR pipeline
```

**这是编译失败，不是精度失败**，处置路径完全不同。

---
