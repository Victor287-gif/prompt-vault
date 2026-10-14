---
type: Guide
title: Sum 算子（sum） 实现手册
description: '算子类别: sum（多维 reducesum，支持 dim / keepdim / dim=None 全归约）。'
tags: [task.implementation, topic.compute.reduction, topic.tiling, topic.implementation.atomic, topic.compute.normalization, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Reduction_Ops/sum.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Sum 算子（sum） 实现手册

> 类别约束见 [math_compute_constraints.md](../constraints/math_compute_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 Sum 算子（sum）

**算子类别**: `sum`（多维 reduce-sum，支持 dim / keepdim / dim=None 全归约）
**典型特征**: 按 (ndim, 归一化 dim) 分派到专用归约 kernel；累加在 fp32 完成，输出 cast 回原 dtype
**性能基准**: 几何平均 **0.9459x**（44/44 通过，repeats=20）；1D 与 reduce_first(dim=0) 方向普遍 1.4-4x，reduce 末维大 shape 为主要短板

### §2.1 Layer 1: 设计约束（Agent 必须遵守）

#### L1.1 归约累加必须在 float32 进行
（见 GC-10）
- **必须**在 kernel 内将 load 的数据 `.to(tl.float32)` 后再 `tl.sum` / 累加。
- **Why:** fp16/bf16 直接累加大 tensor 会精度溢出 / 丢精度，verify 比对会失败。
- **How to apply:** `acc = tl.zeros(..., dtype=tl.float32)`；最终 `tl.store(..., acc.to(output_ptr.dtype.element_ty))`。

#### L1.2 归约计算禁止落到 PyTorch
（见 GC-4）
- **禁止**在 `forward()` 中用 `torch.sum` / `x.sum()` / 任何 `torch.*` reduction。
- **Why:** 触发 A-PyTorchFallback，AST 预检查直接失败。
- **How to apply:** 所有归约在 `@triton.jit` kernel 内用 `tl.sum` + 循环累加完成。

#### L1.3 dim=None（全归约）必须用 atomic 累加到标量输出
- **必须**预分配 `output = torch.zeros((), dtype=torch.float32, ...)`，kernel 内 `tl.atomic_add(output_ptr, block_sum)`，返回前 `.to(x.dtype)`。
- **Why:** 全归约无独立输出元素，多 block 部分和需归约到同一地址。
- **How to apply:** grid 按元素分块，每 block 求部分和后 atomic_add。

#### L1.4 keepdim 仅影响 host 侧输出形状
- **禁止**把 keepdim 语义传入 kernel。
- **Why:** keepdim 只改变输出张量形状（是否保留长度为 1 的归约维），与归约计算无关。
- **How to apply:** host 侧 `out_shape[dim]=1`，`keepdim=False` 时再 squeeze 掉该维；kernel 只看归约后的扁平输出索引。

#### L1.5 负 dim 必须在 host 侧归一化
- **必须**在 host 侧 `if dim < 0: dim += ndim`。
- **Why:** kernel 内不处理负 dim 语义；与 M1 的末维归一化配合使用。

### §2.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树（伪代码）

```python
def launch(x, dim, keepdim):
    ndim = x.ndim
    if dim is None:                      # 全归约
        -> sum_kernel_1d + atomic_add (标量输出)
    if dim < 0: dim += ndim              # 负 dim 归一化（L1.5）

    if ndim == 2:
        if dim == last:  -> reduce_last  (grid = cdiv(M, BM), 沿 N 循环)
        else:            -> reduce_first (grid = cdiv(N, BN), 沿 M 循环)
    elif ndim in (3, 4):
        -> kernel_reduce_dim{dim}        (grid = 输出元素数, 沿 reduce 维 BLOCK_R 步进)
    else:
        -> sum_kernel_1d + atomic_add    (高维兜底，展平全归约)
```

#### L2.2 多核并行骨架模式

**模式 A — 按输出元素分配（3D/4D 归约）**:
```
total_out = prod(非归约维)        # 输出元素总数（M5）
grid = (total_out,)               # 每个 program 负责一个输出点
pid -> 该输出点在非归约维的坐标 (a, b, [c])
沿归约维 R 以 BLOCK_SIZE_R 步进循环累加到 acc
store acc 到输出对应位置
```

**模式 B — 按行/列分块（2D 归约）**:
```
reduce_first(dim=0): grid=(cdiv(N,BN))  每个 program 负责 BN 列、沿 M 全累加
reduce_last (dim=1): grid=(cdiv(M,BM))  每个 program 负责 BM 行、沿 N 步进累加
```

**模式 C — 全归约分块（dim=None）**:
```
grid=(cdiv(numel, BLOCK))  每 block 求部分和 -> atomic_add 到标量
```

#### L2.3 通用 N-D strided 索引构造

所有 kernel 用 broadcasting offset 构造多维索引，支持任意非连续 stride：
```python
idx = d0_off[:,None,None]*stride0 + d1_off[None,:,None]*stride1 + d2_off[None,None,:]*stride2
mask = m0[:,None,None] & m1[None,:,None] & m2[None,None,:]
```

### §2.3 Layer 3: 关键技巧

#### L3.1 2D reduce_first（沿 dim0 归约）用列分块 — 实测高效

按列分块，每个 program 负责 `BLOCK_SIZE_N` 列、沿 M 维全累加。访存沿 M 连续、并行度高。

```python
# 示意（变量名/结构须重新设计，勿直接复制）
pid_n = program_id(0)
acc = zeros(BN, fp32)
for m_start in range(0, M, BM):
    load [BM, BN] tile -> sum(axis=0) -> acc += ...
store acc  # 输出形状 (N,) 或 (1, N)
```

**实测**: case5 [128,128]fp16 4.08x、case9 [64,64]bf16 3.65x、case8 [256,256]fp16 1.52x。BLOCK 取 BM=128, BN=64。

**可替代方向**: 对超大 M 可引入 split-K 两阶段归约，进一步降低单 program 串行长循环。

#### L3.2 通用 N-D strided load — 支持任意 stride 与维度

用 broadcasting offset 一次性 load 一个多维 tile，归约指定 axis，是 3D/4D 归约的核心通用模式。

```python
# 示意：3D reduce dim1
for r in range(0, D_reduce, BR):
    d0_idx = a_off[:,None,None]; d1_idx = r_off[None,:,None]; d2_idx = b_off[None,None,:]
    idx = d0_idx*s0 + d1_idx*s1 + d2_idx*s2
    tile = load(ptr+idx, mask=...)
    acc += sum(tile, axis=1)   # 归约掉 r 所在轴
```

**可替代方向**: stride 已知连续时可省略 stride 参数改用线性 offset，减少指令。

#### L3.3 末维归约（reduce_last）当前为短板，勿照搬

当前实现 `BLOCK_SIZE_M=16, BLOCK_SIZE_N=1024` 沿 N 步进，大 N 时串行迭代多、性能差。

**实测**: case31 [8192,16384]fp16 dim=-1 仅 0.22x、case30 [4096,18432]fp32 dim=1 仅 0.285x。

**可替代方向**（待验证，非本版实现）:
- 沿 N 做 split-K：多 program 分段求部分和，再 atomic 或第二阶段归约
- 增大 BLOCK_SIZE_N 减少循环次数 / 提高 M 向并行
- 对末维连续布局改用向量化的连续 load + `tl.sum`

#### L3.4 3D/4D reduce 末维的碎 grid 问题

当前 `BLOCK_SIZE_A=B=C=1`（每 program 仅 1 个输出点）+ `BLOCK_SIZE_R=64`，小 tensor 下 launch 开销主导。

**实测**: case13/14 [64,64,64] dim=2 仅 0.158x（im=0.031ms vs fw=0.0049ms，6x 慢，纯 launch 浪费）。

**可替代方向**: 把输出维也分块（A/B > 1），一个 program 处理多个输出点摊薄 launch；或对小 tensor 直接回落 atomic 全归约。

### §2.4 Sum 性能基准

| Shape 类型 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 1D 全归约 (dim=None/0) | 1.4-1.6x | atomic_add 分块，稳定优于 torch |
| 2D reduce_first (dim=0) | 1.5-4.1x | 列分块，最强方向 |
| 2D reduce_last (dim=1, 大 N) | 0.22-0.7x | 末维大 shape 短板 |
| 3D/4D reduce 非末维 | 1.4-3.1x | 输出点并行，性能良好 |
| 3D/4D reduce 末维 | 0.15-0.6x | 碎 grid + launch 开销，短板 |

**关键结论**: 几何平均 0.9459x（>0.8 归档线但 <1.0）。优势在 reduce_first 与 1D；**主要提升空间在末维归约的大 shape（split-K / 增大 BLOCK_N）和小 tensor 末维归约的碎 grid（输出维分块）**。

---
