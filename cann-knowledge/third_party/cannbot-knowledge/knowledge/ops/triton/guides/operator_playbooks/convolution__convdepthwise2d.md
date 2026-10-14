---
type: Guide
title: ConvDepthwise2d 算子（2D depthwise） 实现手册
description: 算子类别：convolutiondepthwise2d　典型特征：(N, C, H, W) × weight C, 1, kH, kW（groups==C），无跨通道规约。
tags: [task.implementation, topic.compute.neighborhood.convolution, topic.data_movement.memory_access, topic.implementation.atomic, topic.implementation.memory_management, topic.data_movement.layout_transform]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# ConvDepthwise2d 算子（2D depthwise） 实现手册

> 类别约束见 [convolution_constraints.md](../constraints/convolution_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 ConvDepthwise2d 算子（2D depthwise）

**算子类别**：`convolution-depthwise-2d`　**典型特征**：`(N, C, H, W)` × weight `[C, 1, kH, kW]`（groups==C），无跨通道规约。
**性能基准**：50/50 pass，geomean **~0.90x** vs torch（target 0.8x，达标）。

### §3.1 Layer 1：特化约束

- **fan_in**（G1）：depthwise `groups==C_in`，故 `fan_in = kH * kW`（**禁止** 写成 `out_channels * kH * kW`）。
- **UB 限制**：`BLOCK_M = 16`；`BLOCK_N ≤ 128`（如 OH_TILE=8, OW_TILE=16）。5D block_ptr 单次 load 会把整个 receptive field `[BLOCK_M, OH_TILE, OW_TILE, kH, kW]` 驻留 RF，总占用约 `BM·OH·OW·KHW·4B`，上限约 50KB；tile 须按 KHW 自适应。
- depthwise **禁止** `tl.dot`（A4）。

### §3.2 Layer 2：三路径 hybrid

| 条件 | 路径 | tile |
|------|------|------|
| stride==1 | 预 pad + **3D block_ptr KHW-loop** | 大空间 16×16，否则 8×16 |
| stride≥2 | **NCHWc pack + 5D block_ptr conv**（A2） | 按 KHW 自适应（见下） |
| k=5/k=7 大空间且 UB 受限 | **split-KHW + atomic_add**（G11） | 放大 spatial tile |

- **3-way pad 分派**（避免统一大 alloc）：stride=1&pad=0 → 透传 x（零开销）；stride=1&pad>0 → halo 分配；stride≥2 → 全 alloc，`H_alloc = max(H+2p, stride*H_out_aligned + kH - 1)`。
- **5D block_ptr tile（按 KHW 自适应）**：KHW≤9(k=3) → OH=4,OW=16（W_out≥32）或 4×8；KHW≤25(k=5) → 4×8；KHW≤49(k=7) → 2×4。
- **NCHWc pack+conv 流水线**：conv 前用 pack kernel 把 `x[N,C,H,W]` 转置为 `x_packed[N, C_BLK, H, W, 16]`（NCHWc），conv 用 5D block_ptr 读 `[OH_TILE, OW_TILE, kH, kW, 16]`，每个 x 元素只读 1 次；conv 内部 `tl.trans` 后**直接写 NCHW 输出**，**禁止**加独立 unpack kernel；**禁止** fuse pack+conv 进单 kernel（实测 geomean 0.90x → 0.59x）。
- **split-KHW + atomic_add 启用阈值**（经验值）：
  - k=5 stride≥2 且 H_out·W_out ≥ 256
  - k=5 stride=1 且 C ≤ 16 且 H_out·W_out ≥ 256
  - k=7 stride=1 且 H_out·W_out ≥ 400
  - k=7 stride=2 不受益；小 case 不受益。

### §3.3 性能基准

50/50 pass，geomean **~0.90x**。收益链：blockptr 向量化 0.0455→0.32x → 5D 单 load 0.50→0.58x → NCHWc pack+conv 0.58→0.83x → split-KHW+atomic 0.83→0.90x。

---
