---
type: Optimization
title: 数据搬运与内存访问模式优化
description: NPU 数据通路（GM/L1/UB/L0C 与 MTE1/2/3/FIX 流水线）特征、存算并行与 MultiBuffer 双缓冲、for
  循环 Tiling，以及 取余转 mask、load 重排、指针 += 消除、入参 constexpr 化四个访存优化模式。
tags: [task.optimization, topic.data_movement, topic.data_movement.memory_access, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Memory_Pointer_Ops/tl.load.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# 数据搬运与内存访问模式优化

## NPU 数据通路特征

```
延迟从低到高：L0C(~1ns) < UB(~5ns) < L1(~10ns) < GM(~100ns, ~1.8TB/s HBM)
```

| 流水线 | 方向 | 说明 |
|--------|------|------|
| MTE2 | GM → UB | Vector 搬入，与 MTE3 共享 GM 带宽（同时搬运各约一半） |
| MTE3 | UB → GM | Vector 写出 |
| MTE1 | GM → L1 | Cube 搬入，独立带宽 |
| FIX (fixpipe) | L0C → UB（910_95）/ L0C → GM/L1（A2/A3） | Cube 结果传递 |

理论搬运耗时 = 数据量 / 带宽。例：fp16 4096×4096（32MB）约 17.8us。

## 存算并行与 MultiBuffer

存算并行 = 当前批次计算时提前搬入下一批数据。前提：kernel 内有 for 循环 Tiling（多迭代才能形成流水线）、迭代间无数据依赖、UB 空间充足。

MultiBuffer 为同一张量创建 2 个缓冲区交替使用（ping-pong）：

```python
from triton.language.extra.cann import extension as al

for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
    xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    al.multibuffer(tmp0, 2)     # size 当前仅支持 2
    tl.store(out_ptr0 + xindex, tmp0, xmask)
```

| 配置 | UB 可用空间 | 默认值 |
|------|------------|--------|
| multiBuffer=False | 192KB（A2/910B）/ 248KB（910_95） | 910_95 默认 |
| multiBuffer=True | 减半：96KB / 124KB | A2/A3 默认；开启后可能需减小 BLOCK_SIZE_SUB |

AutoTilingTuner 会为每个候选配置自动生成 multibuffer 开/关变体。

### for 循环 Tiling（存算并行的前提）

优化前（单次处理，无流水线）：

```python
offset_many_page = tl.arange(0, max_num_extend_tokens)
page_start = tl.load(free_page_ptr + new_page_start_loc + offset_many_page // page_size,
                     mask=offset_many_page < num_part2)
tl.store(out_indices + output_start_loc + offset_many_page,
         page_start * page_size + offset_many_page % page_size,
         mask=offset_many_page < num_part2)
```

优化后（分块循环，搬入 N+1 / 计算 N / 写出 N-1 重叠）：

```python
num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
blk_offset = tl.arange(0, BLOCK_SIZE)
for i in range(num_loop):
    offset_many_page = blk_offset + i * BLOCK_SIZE
    page_start = tl.load(free_page_ptr + new_page_start_loc + offset_many_page // page_size,
                         mask=offset_many_page < num_part2)
    tl.store(out_indices + output_start_loc + offset_many_page,
             page_start * page_size + offset_many_page % page_size,
             mask=offset_many_page < num_part2)
```

注意：BLOCK_SIZE_SUB 在 UB 允许内尽量大以减少循环次数；每个子块独立 mask；循环后性能反降通常是子块过小或存在数据依赖。

验证存算并行生效：msprof 看 `aiv_mte2_time` 与 `aiv_vec_time` 是否有显著重叠。

## 模式 1：变量取余 → Mask 替代（收益 2x-10x）

触发：用 `%` 对地址索引取余处理边界。`%` 在 NPU 上走标量 ALU，地址离散化，破坏向量化访存，可能慢 5-10 倍。

优化前：

```python
offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
```

优化后（循环外一次计算边界 mask，`&` 组合）：

```python
offset_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offset_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
msk_m = offset_xm < M
msk_n = offset_wn < N
# load 时：mask=msk_m[:, None] & (offset_k[None, :] < K - k * BLOCK_SIZE_K)
```

要点：用 `&`（位与）而非 `and`（逻辑与，返回单布尔值不适用于 tensor mask）；所有需边界的维度都要有 mask。

## 模式 2：Load 指令重排序（收益 1.05x-1.20x）

触发：循环内多条 `tl.load`，其中一条依赖上一次迭代的 store。编译器不重排 load 顺序，排前的有依赖 load 会阻塞排后的无依赖 load。

优化前（load B 等上次 store B，load A 被连带阻塞）：

```python
for i in range(HEAD_NUM):
    idx_B = tl.load(p_B_index)
    b_B = tl.load(B + idx_B)      # 依赖上次迭代的 store B
    b_A = tl.load(p_A)            # 无依赖，但被阻塞
    ...
    tl.store(p_B, b_B)
```

优化后（无依赖的 load A 提前，与上次 store B 并行）：

```python
for i in range(HEAD_NUM):
    b_A = tl.load(p_A)            # 提前发射
    idx_B = tl.load(p_B_index)
    b_B = tl.load(B + idx_B)
    ...
    tl.store(p_B, b_B)
```

## 模式 3：避免循环内 `+=` 更新指针（收益 1.05x-1.20x）

触发：循环内 `ptr += offset`。这产生迭代间 RAW 依赖，阻塞流水线调度。改为基地址 + 偏移独立计算：

优化前：

```python
x_ptrs = a_ptrs_base
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(x_ptrs, ...)
    x_ptrs += BLOCK_SIZE_K          # RAW 依赖
```

优化后：

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K   # 各次迭代独立
    x = tl.load(x_ptrs, ...)
```

## 模式 4：入参静态化 `tl.constexpr`

运行时不变的参数（BLOCK_SIZE、stride 等）声明为 `tl.constexpr`，启用编译期常量折叠/传播：

```python
def kernel(A, B, M, N,
           stride_am: tl.constexpr,           # 优化后
           BLOCK_SIZE_M: tl.constexpr):
```

只适用于编译期已知值；不确定时询问用户确认。

## 其他搬运技巧

- **先搬后选**：离散场景先整体搬到 UB 再 `tl.gather`，优于多次小批量 L2 搬运：
  ```python
  x_shared = tl.load(x_ptr + tl.arange(0, M) * stride_x)   # 整体搬入 UB
  val = tl.gather(x_shared, idx, 0)
  ```
- **care_padding=False**：对 padding 数据无依赖时减少同步开销
- **L1 缓存**：矩阵乘 K 维循环中 A/B 分块可缓存 L1，减少重复搬运
- **对齐**：32B/512B 对齐是高效搬运前提

## 910_95 特别注意

- Vector Core 单次处理 256bit，标量化退化影响更严重，模式 1 收益可达 5-10x
- UB 248KB（910B 192KB）；L0C 256KB（910B 128KB）；L1/L0C/UB 对齐分别 32B/512B/32B
- multibuffer 默认 False，需显式开启；开启后 UB 可用减半为 124KB


## 相关源

- 主源：docs_triton_ascend/05-Performance-Optimization/06-data-movement-optimization.md
- 合并来源：docs_for_triton_agent/04-memory-access-patterns.md（四个访存优化模式）

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [multibuffer.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Extension_Ops/multibuffer.md)
- [programming_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/programming_guide.md)
- [performance_guidelines.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/performance_guidelines.md)
