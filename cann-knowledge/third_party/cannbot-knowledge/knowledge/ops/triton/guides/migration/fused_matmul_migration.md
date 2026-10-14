---
type: Guide
title: Fused Matmul GPU→NPU 迁移对照
description: Fused Matmul（fwd/bwd_b/bwd_x/bwd_w 四 kernel）从 GPU Triton 迁移到 Ascend NPU
  的 10 个 Diff 点：bias 融合进 Cube 流水线、指针基地址+偏移、SPLIT_K 简化与编译参数选择。
tags: [task.migration, topic.compute.composite, topic.quality.compatibility.version_migration, topic.programming_model.execution_unit.cube, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# Fused Matmul GPU→NPU 迁移对照

基于 `fused_matmul.py`（GPU）与 `fused_matmul_npu_v3.py`（NPU）逐行对比。适用：GEMM+Bias/Activation 融合类算子（含 fwd/bwd_b/bwd_x/bwd_w 多 kernel）迁移。

## Diff 1：Autotune 配置（移除 num_stages/num_warps，收窄 BLOCK）

NPU 上 `num_stages`/`num_warps` 无意义应移除（源 NPU 版本保留属遗留代码）。BLOCK 候选值收窄：BM/BN/BK 优先 128（Cube 高效分块），BK 由 `[32, 64]` 收窄为 `[128]`。**避坑**：源 NPU 版 bwd_w 出现 `BM in [86, 128]`——86 非 2 的幂次，NPU 上有对齐风险，应只用 2 的幂次值。

## Diff 2：Bias 融合（提前加载 + 显式广播，性能收益关键）

NPU 为 CV 分离架构。bias 在 matmul 循环后加载并隐式广播时，编译器无法将其识别进 Cube 流水线，bias 累加落到 Vector 核，产生 Cube->UB 搬运和 CV 同步开销。

```python
# GPU
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    ...
    accumulator = tl.dot(x, w, accumulator)
if HAS_BIAS:
    b = tl.load(b_ptrs, mask=offset_wn < N, other=0.0)
    accumulator += b[None, :]                    # 隐式广播，Vector 核执行
# NPU
if HAS_BIAS:
    b = tl.load(b_ptrs, mask=offset_wn < N, other=0.0)   # 循环前加载
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    ...
    accumulator = tl.dot(x, w, accumulator)
if HAS_BIAS:
    accumulator += tl.broadcast_to(b[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))  # 显式广播
```

**收益**：bias 累加进入 Cube 流水线内完成，消除跨核搬运与 CV 同步。

## Diff 3：指针计算（基地址+偏移，避免 +=）

GPU 循环内 `x_ptrs += BLOCK_SIZE_K` 推进指针，NPU 编译器上会形成长指针依赖链、不利指令调度。改为循环外算基地址、循环内重算：

```python
# GPU
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(x_ptrs, ...); accumulator = tl.dot(x, w, accumulator)
    x_ptrs += BLOCK_SIZE_K; w_ptrs += BLOCK_SIZE_K * N
# NPU
a_ptrs_base = x_ptr + (offs_am[:, None] * K + offs_k[None, :])
b_ptrs_base = w_ptr + (offs_k[:, None] * N + offs_bn[None, :])
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
    w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N
```

迁移模板：`ptrs_base = base_ptr + (row_offs[:, None] * stride + col_offs[None, :])`，循环内 `cur_ptrs = ptrs_base + k * BLOCK_K`。

## Diff 4：Mask 计算（移除 % 取余，独立 mask 变量）

```python
# GPU
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
x = tl.load(x_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
# NPU
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)   # 无取余
msk_m = offs_am < M; msk_n = offs_bn < N
x = tl.load(x_ptrs, mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
```

前置条件：循环前 `if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N): return`，保证不越界才允许去掉 `% M`。取余在 NPU 上产生额外标量开销；独立 mask 变量利于编译器复用。

## Diff 5：编译参数配置

| Kernel | 类型 | NPU 参数 |
|--------|------|---------|
| `fwd_kernel` | CV 融合 | `enable_auto_bind_sub_block=True, set_workspace_multibuffer=2, sync_solver=True, limit_auto_multi_buffer_of_local_buffer="no-limit", multibuffer=True, enable_flatten=True` |
| `bwd_b_kernel` | 纯 Vector | `enable_auto_bind_sub_block=False` |
| `bwd_x/bwd_w_kernel` | 纯 Cube | `enable_auto_bind_sub_block=False` |

选择逻辑：无 `tl.dot` → 纯 Vector；有 `tl.dot` 且 Cube 后有 Vector 处理 → CV 融合（全参数）；有 `tl.dot` 但无 CV 协同需求 → 纯 Cube（`enable_auto_bind_sub_block=False`）。fwd 用 `enable_flatten=True` 的前提是 bias 已通过 Diff 2 融入 Cube 流水线，剩余 Vector 后处理（类型转换+store）可展平。

## Diff 6：bwd_x / bwd_w 指针计算

同一模式扩展：`offset_k` 移入循环内定义（`offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)`），指针每次循环重算，mask 拆为 `(offs_m < M) & (offs_k < K)` 独立变量，移除 `%` 取余。

## Diff 7：SPLIT_K 简化（重要负面结论）

```python
# GPU：pid_k = tl.program_id(axis=1)，逆序循环 + SPLIT_K 步进 + atomic_store
for k in range(K, 0, -BLOCK_SIZE_K * SPLIT_K): ...
if SPLIT_K == 1: tl.store(dw_ptrs, dw, mask=dw_mask)
else: atomic_store(dw_ptrs, dw, dw_mask, LOCK_W, SPLIT_K)
# NPU：无 pid_k，正序单 program 内循环，直接 store
for pid_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)): ...
tl.store(dw_ptrs, dw, mask=dw_mask)
# grid 也从 2D (num_blocks, SPLIT_K) 变为 1D (num_blocks,)
```

**注意（避坑）**：NPU 版将 K 维归约完全放在单 program 内循环完成，autotune config 中虽保留 SPLIT_K 但 kernel 逻辑不再使用。**若 K 很大导致性能问题，需重新引入 SPLIT_K 并行归约**——这是当前实现的已知局限，不是最佳实践。

## Diff 8：设备 API（torch.cuda → torch.npu）

`import torch_npu`；`device="cuda"` → `"npu"`；`torch.cuda.synchronize()` → `torch_npu.npu.synchronize()`；profiler 用 `torch_npu.profiler.profile(activities=[ProfilerActivity.NPU], ...)`。

## Diff 9：装饰器 maybe_triton_jit → triton.autotune

NPU 直接用标准 `@triton.autotune(configs=..., key=["N", "K"])`，移除 `maybe_triton_jit` 兼容层导入。

## Diff 10：精度评估工具

NPU 版新增 `benchmark_compare_close` + `eval_standard`（fp32 rtol 1e-6 / fp16 rtol 1e-3 / bf16 rtol 4e-3），从相对误差最大值、均值、小值域 error 占比、RMSE 四个维度评估，比 `allclose` 更严格。

## 910_95 特别注意

| 特性 | 说明 |
|------|------|
| multibuffer | 910_95 默认 `False`，必须显式 `=True` |
| UB 空间 | 910_95 为 256KB（910B 192KB），允许更大 BLOCK_SIZE |
| FixPipe 直通 | 支持 L0C -> UB 直通（910B 需 GM 中转） |
| BLOCK_SIZE 对齐 | 须满足 32B 对齐，用 2 的幂次值天然满足 |

## 迁移检查清单

移除 `num_stages`/`num_warps` → BLOCK 收窄为 2 的幂次（优先 128）→ bias 提前加载 + `tl.broadcast_to` → 指针改基地址+偏移（禁 `+=`）→ 移除 `%` 取余（配 early return）→ mask 拆独立变量 → fwd 配 CV 融合参数集、bwd 配 `enable_auto_bind_sub_block=False` → bwd_w SPLIT_K 简化为单 program 循环 → 设备 API 换 `torch_npu` → 装饰器换 `triton.autotune` → 910_95 显式 `multibuffer=True`。