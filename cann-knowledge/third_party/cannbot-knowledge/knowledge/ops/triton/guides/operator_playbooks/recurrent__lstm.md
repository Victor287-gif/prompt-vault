---
type: Guide
title: LSTM 算子（lstm） 实现手册
description: '算子类别: recurrentneuralnetwork。'
tags: [task.implementation, topic.compute.composite, topic.data_movement.layout_transform, topic.implementation.memory_management, topic.compute.elementwise.arithmetic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# LSTM 算子（lstm） 实现手册

> 类别约束见 [recurrent_constraints.md](../constraints/recurrent_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 LSTM 算子（lstm）

**算子类别**: `recurrent-neural-network`
**典型特征**: 4-gate(i/f/g/o) 递归，cell state c + hidden h 双状态，单层前向；输入 x[S,B,I]、W_ih[4H,I]、W_hh[4H,H]；多 dtype（fp32/fp16/bf16）、多 shape（含 batch_first 翻转）。
**性能基准**: 60/60 pass，几何平均加速比 **13.3673x** vs torch（ascend910b1）

### §2.1 Layer 1: 设计约束（LSTM 专属，Agent 必须遵守）

#### L1.1 禁止 host 侧 `.t()` / `.transpose()`；权重按原生 `[4H,I]` / `[4H,H]` 布局传入 kernel，kernel 内 coalesced 加载 + `tl.trans`
- **必须** 以 nn.LSTM 原生布局 `[4H,I]` / `[4H,H]` 直接传入 kernel，**禁止** host 侧 `weight_ih_l0.t().contiguous()`、`weight_hh_l0.t().contiguous()` 或 `output.transpose(0,1).contiguous()`。
- **Why:** host 转置方法在需求上被禁止；同时在 triton-ascend 上，若直接按 `[BK,BN]` tile 交换 stride 读取（`offs_k*1 + offs_n*I`），tile 的快速轴 `BN` 会落在内存中非连续的 `4H` 维度上，产生非 coalesced gather load，实测几何平均从 7.2x 暴跌到 **0.12x**。
- **How to apply:** 在 kernel 内先以**原生朝向 `[BN,BK]`** 加载权重（此时 `BK=I/H` 轴连续，coalesced），再在寄存器中用 `tl.trans` 转成 `[BK,BN]` 供 `tl.dot` 使用：
  ```python
  w_native = tl.load(w_ptr + offs_n[:,None]*stride_w_n + offs_k[None,:]*stride_w_k, ...)
  acc += tl.dot(x_tile, tl.trans(w_native))   # w_native[n,k] = weight[n,k] == W^T[k,n]
  ```
  其中 `stride_w_k = weight.stride(1)`（I/H 轴，=1），`stride_w_n = weight.stride(0)`（4H 轴，=I/H）。`tl.trans` 是 tile/寄存器级操作，不是 tensor 方法，不在禁止之列。实测修复后几何平均回升并提升到 **13.37x**。
- **batch_first I/O:** 输出按目标布局直接 `torch.empty(B,S,H)` 或 `(S,B,H)`，将逻辑 S/B/H stride 传入 gate-update kernel，避免末尾 `transpose()`；batch_first 输入用每步仿射切片 `x[:, t, :]` 读取（融合 GEMM 无法以仿射 stride 读 `[B,S,I]`）。

#### L1.2 冷编译首次执行稳定性：禁止 kernel 内跨 timestep 循环，必须使用 host-loop 单步 kernel
- **必须** 将跨时间步的 recurrence 放在 **host 侧循环**（`for t in range(S):`），每个 timestep 启动独立的单步 kernel（输入投影 matmul / h@W_hh matmul / gate elementwise-update）。**禁止**在单个 `@triton.jit` kernel 内写 `for t in range(S)` 并配合 double-buffer workspace 完成多步递归。
- **Why:** triton-ascend 上带有内部 timestep 循环 + double-buffer workspace 的 recurrence kernel，在冷编译后首次 device 执行时存在不稳定的首启路径；表现为 gate 累加异常（`gate_g/o` dot 贡献缺失），`sigmoid(gate_o)=0` → `h_new` 全零，同 shape 第二次执行才正确。该现象与 kernel 内部 recurrence 结构强相关，单纯的 dot 模式修改、grid 调整、kernel 间/内同步均无法可靠消除。**偶发精度 fail 一律按真 bug 处理，硬标准：单次执行也要对。**
- **How to apply:** 采用三层稳定 kernel 结构：
  1. `lstm_input_proj_kernel`：一次性 `x_proj[S*B,4H] = x @ W_ih + bias`（稳定 matmul）。
  2. Host 侧 `for t in range(S)`：每步调用 `lstm_hh_gemm_kernel(h_cur, W_hh, gbuf)` 计算 `h_prev @ W_hh`（纯 matmul，无 recurrence）。
  3. 同一步调用 `lstm_gate_update_kernel(gbuf, x_proj[t], c_cur, h_nxt, c_nxt, output)` 完成激活与状态更新（纯 elementwise，无 dot）。
  这样没有任何 kernel 包含跨 timestep 的循环或 double-buffer workspace，从根本上避开首启不稳定结构。实测 case11(H=128 fp32) 冷启动 6/6 PASS，case13(H=256 fp32) 3/3 PASS，热缓存 verify 60/60 PASS，benchmark geomean 13.3673x。
- **与 R1/R4 的关系**：R1/R4 描述的是"若必须在 kernel 内做时间串行"时的约束；L1.2 是更高优先级的**替代方案**——直接把时间串行移出 kernel，用 host 循环保证正确性。当稳定性和 peak 性能冲突时，优先遵守 L1.2。

### §2.2 Layer 2: 算法骨架

当前稳定实现统一走 **host-loop 单步 kernel**（L1.1 + L1.2），所有 shape 共用同一路径，不做 I/H dispatch，也不在 kernel 内做 gate 合并。整体结构如下：

```
# (1) 一次性输入投影：x_proj[S*B,4H] = x_flat @ W_ih^T + (b_ih+b_hh)
lstm_input_proj_kernel[grid_proj](x_flat, W_ih, b_ih, b_hh, x_proj)
x_proj = x_proj.view(S, B, 4H)

# (2) host 循环：每 timestep 一个 matmul + 一个 elementwise
gbuf = empty(B, 4H)
for t in range(S):
    h_cur = h_ws[t % 2];      h_nxt = h_ws[(t+1) % 2]
    c_cur = c_ws[t % 2];      c_nxt = c_ws[(t+1) % 2]
    lstm_hh_gemm_kernel[grid_gemm](h_cur, W_hh, gbuf)          # gbuf = h_cur @ W_hh^T
    lstm_gate_update_kernel[grid_upd](gbuf, x_proj[t], c_cur, h_nxt, c_nxt, output)
```

**4-gate 处理**：`lstm_input_proj_kernel` 与 `lstm_hh_gemm_kernel` 都输出 4H 宽度的 gate buffer；`lstm_gate_update_kernel` 在 elementwise 阶段拆成 i/f/g/o，分别做 sigmoid/tanh，然后 `c_new = f·c_old + i·g`、`h_new = o·tanh(c_new)`。

**实测 tile 配置**：
- `lstm_input_proj_kernel`: `BLOCK_M=BLOCK_N=BLOCK_K=128`（大 matmul，一次性处理 S*B 行）。
- `lstm_hh_gemm_kernel`: `BLOCK_M=BLOCK_N=BLOCK_K=64`（小 matmul [B,4H]=[B,H]@[H,4H]，避免 UB 溢出）。
- `lstm_gate_update_kernel`: `BLOCK_B=32, BLOCK_H=64`（同时驻留 8 个 gate tile 时的 UB 上限）。

### §2.3 Layer 3: 关键技巧（可参考，不可照抄）

- **bias 合并到输入投影**：`b_ih + b_hh` 在 host 侧相加后一次性加到 `x_proj`，避免在每个 timestep 重复加 bias；bias 为 None 时创建 `torch.zeros(4H)` 占位，避免 kernel 分支。
- **工作区强制 fp32**：`h_ws/c_ws` 与 `x_proj/gbuf` 都用 fp32，只在最终写 output / 返回 h_n/c_n 时 `.to(x.dtype)`，减少递归中 fp16 累放误差。
- **gate buffer 的宽连续布局**：`gbuf[B,4H]` 与 `x_proj[t][B,4H]` 都按 `[batch, 4*H]` 连续存储，`lstm_gate_update_kernel` 通过 `g*H + h_offs` 取对应 gate 段，无需切片或 split。
- **UB/tile 上限**：`lstm_gate_update_kernel` 同时加载 8 个 tile（4 个来自 gbuf、4 个来自 x_proj），容易触发 UB 溢出。需根据 H 调整 `BLOCK_B × BLOCK_H`，实测 32×64 在 192KB UB 下安全。
- **测量必须用 benchmark.py（torch_npu.profiler 解析 op 时长）**，自写 harness（npu.Event/host loop）对 <0.3ms 的快 case 有 host-launch 开销下限，不可信。

### §2.4 LSTM 性能基准（ascend910b1，60-case 几何平均）

| 方案 | 几何平均 | 说明 |
| --- | --- | --- |
| **host-loop 单步 kernel 重实现 + 原生权重 `tl.trans`（L1.1 + L1.2）** | **13.3673** | **2026-08-07 最终稳定版本，移除 host `.t()`/`.transpose()`，kernel 内 coalesced load + `tl.trans`，60/60 verify + 冷启动稳定性验证** |

**host-loop + 原生权重布局**：将 recurrence 从 kernel 内部移到 host 循环，每 timestep 拆分为独立 matmul + elementwise kernel；同时以原生 `[4H,I]`/`[4H,H]` 权重布局传入，kernel 内做 `[BN,BK]` coalesced 加载 + `tl.trans` 得到 `W^T`。该结构既消除了冷编译首次执行的不稳定首启路径（case11 6/6 冷启动 PASS），又避免了 host 转置方法的禁止调用，几何平均达到 **13.37x**。

### §2.5 融合搜索基因编码存档（1_LSTM, 910b2, 2026-09-01）

> 来源：`triton-optimizer-multi-agent` fusion-search 模式，R1 构建初始基因池即达收敛，**C1 停止（1 round）**。best_speedup = **14.7686**（route-1，60/60 case 精度 PASS）。torch 基线 = `nn.LSTM` msprof geomean **0.3024ms** → impl **0.0205ms**。final kernel `1_LSTM_final.py`（md5 96160035）。full 9 基因定义见任务 `opt-round-1/gene_pool.json`。

#### active 基因（收敛后保留）

| 基因 | 类别 | fitness | 编码动作 | 实测结论 |
|------|------|---------|----------|----------|
| G001 | structural | 14.7686 | 权重原生布局 + kernel 内 tl.trans：W_ih[4H,I]/W_hh[4H,H] 按 [BN,BK] coalesced 加载，寄存器转置 | 结构性最大杠杆；禁 host .t()/.transpose() |
| G005 | compute | 14.7686 | bias 合并到输入投影：b_ih+b_hh host 侧一次相加；None 时 zeros(4H) 占位 | 循环不变量；避免每 timestep 重复加 bias |
| G007 | memory | 14.7686 | gate buffer [B,4H] 宽连续布局，不做切片/split | 循环不变量；g*H+h_offs 直接索引 |
| G008 | structural | 14.7686 | host-loop 单步 kernel（L1.2）：for t 移 host 侧，每步独立 hh GEMM + gate-update | 消除冷编译首启不稳定路径 |
| G002 | tiling | 1.42 | 输入投影 BLOCK_M=N=K=128（一次性处理 S*B 行） | BLOCK_K 禁 256（0.594x UB 退化） |
| G003 | tiling | 1.42 | hh GEMM BLOCK_M=N=K=64 | 小 matmul [B,H]@[H,4H] 避免 UB 溢出 |
| G004 | tiling | 1.42 | gate-update elementwise BLOCK_B=32, BLOCK_H=64 | 8 个 gate tile 的 UB 上限 |
| G006 | stability | — | workspace 强制 fp32，仅最终写 output/h_n/c_n 时 .to(x.dtype) | 减少 fp16 递归累放误差 |
| G009 | compute | — | tl.dot fp32 累加器 + 输入原生 dtype | 禁逐维外积 / 先 upcast |

**结构性结论**：host-loop 单步结构 + 原生权重 `tl.trans` 是结构性最大杠杆（旧版 runbook 13.37x → fusion-search 14.77x）；bias 合并（G005）与 gate buffer 宽连续布局（G007）为循环不变量优化。R1 即 C1 停止——首次融合就大幅超越目标 5x，无需继续遗传进化。

---

## 并入：recurrent__gru

## §3 GRU 算子（gru）

**待归档**。3-gate(r/z) 递归、单 h 状态（无 cell），递推 `h_t = (1-z)·h_{t-1} + z·r̃`。可复用：§1 R1-R5 全部通用约束；无 cell state → R4 只需一组双缓冲。优化重点在于 host-loop 拆分后的单步 matmul + elementwise 的 tile 与 UB 适配。

---

## 并入：recurrent__rnn

## §4 RNN 算子（vanilla rnn）

**待归档**。单 gate `h_t = tanh(x_t@W_ih + h_{t-1}@W_hh + b)`。可复用：§1 R1-R4；单 gate 无 multi-load 问题，优化重点回到 UB/tile 形状与 R3 batch 分核的并行度挖掘。

---
