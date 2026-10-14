---
type: Guide
title: recurrent 类算子设计约束
description: recurrent 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.compute.composite, topic.tiling, topic.implementation.memory_management, topic.compute.elementwise.arithmetic, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# recurrent 类算子设计约束

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## 循环神经网络类算子优化经验

本文档合并该类别算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子（LSTM/GRU/RNN）共有的递归工程约束，首次生成必须遵守
- **§2 LSTM**（首个归档算子）/ **§3 GRU**（待归档）/ **§4 RNN**（待归档）
- **§5 常见陷阱**（按算子分小节）

> ⚠️ **关键区分**：本类别的核心优化哲学是 **「host-loop 单步 kernel + 禁止 host 转置 + kernel 内 coalesced 加载」**。时间维递归必须在 host 侧串行；权重以原生布局传入，kernel 内通过 `tl.trans` 获得转置视图，禁止调用 `.t()`/`.transpose()`。
> 首次归档算子：`1_LSTM`（单层 LSTM 前向，60-case 多 shape，ascend910b1）；归档时最佳几何平均 **13.3673x**。
> 历史经验仅供启发，**禁止直接复制代码结构**；变量名/结构必须重新设计。
>
> **边界与 `linear_recurrent_constraints.md` 的分工**：本卡适用于**向量状态**（`[B,H]`，无大 tile 可驻留）的递归算子，走 host-loop 单步 kernel；**矩阵状态**（`[K,V]` fp32 ≤32KB 可整块驻留 UB）的线性递推算子（gated-delta-rule 等）路由到 [linear_recurrent_constraints.md](linear_recurrent_constraints.md)，其正解是单融合 persistent kernel + device 内 t 循环。两卡 R1/R2/R3「时间维串行、program 持完整状态」的结论一致，仅 launch 粒度不同。

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/recurrent__lstm.md`](../operator_playbooks/recurrent__lstm.md) — LSTM 算子（lstm）

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| LSTM | `recurrent-neural-network` | 4-gate(i/f/g/o) 递归，cell c + hidden h 双状态 | **host-loop 单步 kernel 保证冷启动稳定**，matmul + elementwise 解耦 |
| GRU | `recurrent-neural-network` | 3-gate(r/z) 递归，单 h 状态，无 cell | 待归档；可套用 §1 R1-R5 |
| RNN（vanilla） | `recurrent-neural-network` | 单 gate `h=tanh(x@W+h@U+b)` 递归 | 待归档；重点是 §1 R1-R4 + UB/tile |

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 5 条约束是本类别算子**共有**的递归工程约束。其他文件已提取的通用约束（如 GC-1）此处不再重复，各算子章节引用时标注。

### R1 时间维必须串行（递归数据依赖）
- **必须** 在单个 program 内 `for t in range(S)` 串行推进时间步；**禁止**把时间维展平进 grid 跨 program 并行。
- **Why:** 递归依赖 `h_t = f(h_{t-1}, ...)` 使时间步间存在真数据依赖；Triton 缺跨 program 全局屏障，跨 program 并行时间维会 RAW 冒险。
- **典型应用**：所有 recurrent 算子（LSTM/GRU/RNN）的前向。
- 与 tensor-transform G4（grid 不超核数）的差异：R1 进一步约束 grid 维度只能落在 batch，**时间维绝对不可分**。

### R2 每个 program 必须持有完整隐维 H（递推归约全 H）
- **必须** 让每个 program 持有完整隐维 H；隐维 tile 只能在**同一 program 内**循环（`for ht in range(num_hidden_tiles)`），且每个 tile 的 W_hh 归约仍读全 H。**禁止**按隐维 tile 把不同 program 分到不同 H 段。
- **Why:** 递推项 `h_{t-1} @ W_hh` 的归约维是全 H（W_hh 为 [H, G·H]，G=gate 数：LSTM=4/GRU=3/RNN=1），每个输出 gate 维依赖全部输入隐维。跨 program 切隐维 → program A 只算了 h_{t-1} 半段，下一步 W_hh 却要读全段 → RAW。
- **How to apply:** 外层 `for ht`（输出隐维 tile），内层 `for k in range(0,H,BLOCK_K)` 读全 H 的 h_tile，同一 h_tile 复用于全部 G 个 gate 的 `tl.dot`（避免 G× 重复 load h）。
- **典型应用**：LSTM(G=4)/GRU(G=3)/RNN(G=1) 递推 kernel。

### R3 仅按 batch 块分核
- **必须** `grid=(num_cores,)`，每个 program 处理一个 `BLOCK_BATCH` 块并串行全部时间步 + 全部隐维 tile。**禁止**用隐维分核"提升并行度"（违反 R2）。
- **Why:** 时间维（R1）与隐维（R2）都不可跨 program 分，唯一可并行的维度是 batch。**小 batch case 并行度低是固有特性**（B=1 时 19/20 核空闲），非优化能解。
- **典型应用**：所有 recurrent；B=1 大 S case 是性能洼地，受限于递归时间屏障（host-loop 逐时间步全 GEMM grid 亦被时间屏障挡）。

### R4 (2,B,H) 双缓冲工作区规避递归 RAW
- **必须** 相邻时间步用乒乓读写 h/c（及 cell state）：`read_off=(t%2)*stride` / `write_off=((t+1)%2)*stride`。**禁止**单缓冲。
- **Why:** 单缓冲下 t 步写 h_t 与 t+1 步读 h_t 落在同一地址，UB/GM 间 RAW 冒险；双缓冲让读写落在不同物理页。
- **How to apply:**
  ```python
  read_off  = (t % 2)       * stride_ws_s
  write_off = ((t + 1) % 2) * stride_ws_s
  ```
- **典型应用**：LSTM（h+c 双状态需两组双缓冲）、GRU（h 单状态一组）。

### R5 recurrent kernel 瓶颈是 scalar/address overhead，不是 cube 吞吐
- **优化方向必须**是削减 K-loop 内的 load 条数与地址算术，**而非**放大 BLOCK_M/BLOCK_K 堆 cube。
- **Why:** profiling 实测传统 fused recur kernel PipeUtilization = scalar ~45% / mte(load) ~29% / **cube(macc) 仅 ~7%**。根因是每个隐维 tile 在 K-loop 内对每个 gate 各做一次 strided load + 小-N dot，地址算术与 load 指令数爆炸，cube 空转。堆 BLOCK 实测普遍退化（§5.1：BLOCK_K=256→0.594x、BLOCK_BATCH=32→UB 溢出）。
- **How to apply in host-loop design:** 把 recurrence 拆成独立 matmul + elementwise，让 matmul 走标准二维 coalesced load，elementwise 无 dot；避免在单个 kernel 内做 4-gate strided load 的复杂 K-loop。
- 与 math-compute M4（UB 决定 tile）的差异：R5 强调瓶颈在地址/load 而非 cube 或纯 UB，优化杠杆是"简化 load"而非"调 tile 形状"。

---

## §5 常见陷阱与避免方法

### §5.1 LSTM 陷阱

| 陷阱 | 现象 | 正确做法 |
| --- | --- | --- |
| 跨 program 隐维分核 | verify 58/60，RAW 冒险 | 遵守 §1 R2，隐维 tile 留在同一 program 内 |
| `(B,4H)` 缓冲切片提 4 gate | `unsupported tensor index: slice(constexpr[0,64])` | 在 elementwise kernel 中通过 `g*H + h_offs` 索引读取，不切片 |
| BLOCK_K=256（输入投影 matmul） | 0.594x，UB 压力退化 | 输入投影 `BLOCK_K=128` |
| BLOCK_BATCH=32 + BLOCK_H_TILE=128 | AICore 异常 507015（UB 溢出） | 保持 16/64 |
| 循环不变量 bias 外提到时间环外 | ~14 case 精度错（impl 该 0 处非 0） | bias 并入输入投影一次性加完 |
| **verify 偶发报 fp32 Path-B case impl 全零、framework 正常**（典型 case12，forward 内实际 `S=1,B=2,H=128`，`batch_first` 翻转后看 S/B） | **kernel 内有 `for t in range(S)` 跨 timestep 循环 + double-buffer workspace 时，冷编译首次 device 执行出现 gate 累加异常**：`gate_g/gate_o` 的 dot 贡献缺失，`sigmoid(gate_o)=0` → `h_new` 全零；**同 shape 第 2 次执行即正确**。这是**真 bug**（偶发精度 fail 必须修，硬标准：单次执行也要对）。已排除 workspace 未初始化（output 预填 sentinel 仍被写成 0，证明 store 执行了）、host 同步、grid 大小、4 连续 dot 模式等假根因 | **真正修复 = 按功能重实现为 host-loop 单步 kernel**：把 `for t` 移到 host 侧，每 timestep 启动独立的 `lstm_hh_gemm_kernel`（纯 matmul）和 `lstm_gate_update_kernel`（纯 elementwise），没有任何 kernel 包含跨 timestep 循环或 double-buffer workspace，从而避开首启不稳定结构。`self-warmup` 只是用第二次执行掩盖问题，不被接受。实测 case11(H=128 fp32) 冷启动 6/6 PASS、case13(H=256 fp32) 3/3 PASS、热缓存 verify 60/60 PASS |
| 原生 `[4H,I]` 权重直接交换 stride 读 `[BK,BN]` tile | benchmark 几何平均从 7.2x 暴跌到 **0.12x**（实现平均延迟 0.056ms → 7.18ms） | 遵守 L1.1：以原生朝向 `[BN,BK]` coalesced 加载 + 寄存器 `tl.trans`，禁止让 tile 快速轴落在非连续维 |
