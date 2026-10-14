---
type: Optimization
title: Tiling 与 Grid 分核策略（910_95 平台）
description: 910_95/910B 平台 Tiling 大小与 Grid 分核数确定方法：UB 预算公式、Grid 收缩、单行改多行、NPU Autotune
  规则。
tags: [task.optimization, topic.tiling, topic.parallelism, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# Tiling 与 Grid 分核策略（910_95 平台）

## 触发条件

- 编写新 kernel 需设定 BLOCK_SIZE / BLOCK_M / BLOCK_N
- 固定 Tiling 改 Autotune
- Grid 分核数超物理核数需收缩
- 每 program 只处理一行，SIMD 利用率低
- UB overflow 编译错误需重算 Tiling

## UB 容量（分平台，禁止混写）

| 芯片 | UB 硬件 | 编译器可用 | 开 multiBuffer 后 | 源码值 (bits) |
|------|--------|-----------|------------------|--------------|
| 910B (A2) | 192 KB | 192 KB | 96 KB | 1572864 |
| **910_95 (dav-c310)** | **256 KB** | **248 KB（预留 8KB）** | **124 KB** | **2031616** |
| 950 | 256 KB | 248 KB | 124 KB | 2031616 |

> 910_95 关键：硬件 256KB，编译器预留 8KB，实际可用 **248KB**。源码注释 `UbSize = 2031616; // bits = 248KB, reserve 8KB for compiler`。

```
UB 预算：所需 = (输入+输出+中间张量数) × BLOCK_SIZE × dtype_bytes × multiBuffer副本数
约束：所需 <= UB 可用
```

对齐（两平台一致）：Vector 算子尾轴 32B（fp16/bf16: 16 倍数；fp32: 8；int8: 32）；CV 融合算子 512B（fp16: 256；fp32: 128）。单值缓冲区（mean/var/max/sum）逻辑 4B，硬件按 32B 对齐分配。

## 核数与算子类型

910_95：vectorcore_num = aicore_num × 2（9581: 32/64；9599: 36/72）。

| 算子类型 | 判断 | 使用核数 |
|---------|------|---------|
| 纯 Vector | 无 `tl.dot`/`tl.matmul` | num_vectorcore |
| CV 融合 | 含 `tl.dot`/`tl.matmul` | num_aicore |

## Grid 收缩到物理核数（GPU→NPU 对照）

判断：一维 `grid > num_cores`；多维 `grid1 × grid2 [× grid3] > num_cores`。

```python
# 优化前（GPU 风格）：grid 随数据量增长
grid = triton.cdiv(X.shape[0], BLOCK_SIZE)
kernel[(grid,)](X, Y, BLOCK_SIZE)

# 优化后：grid 固定物理核数，核内跨步循环
@triton.jit
def kernel(X, Y, BLOCK_SIZE: tl.constexpr, NUM_TASKS: tl.constexpr, NUM_CORES: tl.constexpr):
    pid = tl.program_id(0)
    for task_id in range(pid, NUM_TASKS, NUM_CORES):
        off = task_id * BLOCK_SIZE
        x = tl.load(X + off)
        ...
```

二维展平（kernel 内反解）：`row_id = task_id // NUM_COL_TASKS; col_id = task_id % NUM_COL_TASKS`；三维再套一层除余。`NUM_TASKS`/`NUM_CORES` 必须传 `tl.constexpr` 启用编译器优化。

**不适用**：任务间有依赖须按 program_id 顺序执行；使用 `tl.barrier()`/`tl.NamedBarrier` 同步的分核策略。`TRITON_ALL_BLOCKS_PARALLEL=1` 可让编译器自动收缩（仅核间可并行时）。

## 单行 → 多行（BLOCK_ROWS）

每 program 处理 BLOCK_ROWS 行，提高 SIMD 利用率：

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK_ROWS": BM}) for BM in [1, 2, 4, 8, 16, 32]],
    key=["X_stride", "total_rows"]
)
@triton.jit
def kernel(X_ptr, Y_ptr, X_stride, Y_stride, total_rows,
           BLOCK_ROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row_offs < total_rows
    col_offs = tl.arange(0, BLOCK_SIZE)
    X = tl.load(X_ptr + (row_offs[:, None] * X_stride + col_offs[None, :]),
                mask=row_mask[:, None] & (col_offs[None, :] < X_stride))
    ...
# grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROWS"]),)
```

关键点：2D mask `row_mask[:, None] & col_mask[None, :]`；offset 广播；BLOCK_ROWS 候选取 2 的幂次。

## NPU Autotune 规则

- **Config 只含 Tiling 参数，禁止 `num_stages`/`num_warps`**（NPU 无 warp/软件流水线概念）
- 装饰器置于 `@triton.jit` 上方；key 用代表数据规模的变量（n_numel/M/N）
- 候选值 2 的幂次，小值用满物理核、大值用满 UB；总选到边界值需扩展范围
- `TRITON_PRINT_AUTOTUNING=1` 打印最优参数

## UB 预算实例（910B vs 910_95，batch/iteration）

| 算子 (D=4096, fp16→fp32) | 合计 | 910B | 910_95 |
|---|---|---|---|
| RMSNorm | ≈64 KB | 3 | 3 |
| LayerNorm | ≈64 KB | 3 | 3 |
| Softmax | ≈48 KB | 4 | 5 |
| MatMul BM=BN=128, BK=64 fp16 | ≈96 KB | 2（开 multiBuffer 不足） | 2（开 multiBuffer 124KB 可行） |

## 常见错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| ub overflow | 缓冲区总量超 UB | 减 BLOCK_SIZE 或二级分块 |
| coreDim 超限 | `ceil(N/BLOCK) > 65535` | 增大 BLOCK 或 Grid 收缩 |
| SIMD 利用率低 | 每 program 1 行 | BLOCK_ROWS 多行 |
| Autotune 无效 | Config 含 num_stages/num_warps | 移除 GPU 专属参数 |


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [utils.py#L41-L50](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/utils.py#L41-L50)
- [utils.py#L48-L49](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/utils.py#L48-L49)
- [utils.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/utils.py)
- [programming_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/programming_guide.md)
