---
type: Guide
title: Pad 算子（transformation-memory） 实现手册
description: '算子类别: transformationmemory。'
tags: [task.implementation, topic.data_movement.layout_transform, topic.tiling, topic.compute.elementwise.arithmetic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Shape_Manipulation_Ops/reshape.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Pad 算子（transformation-memory） 实现手册

> 类别约束见 [tensor_transform_constraints.md](../constraints/tensor_transform_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §3 Pad 算子（transformation-memory）

**算子类别**: `transformation-memory`
**典型特征**: 数据搬运为主，计算极简（仅边界坐标映射），输出 shape != 输入 shape
**性能基准**: 51 cases 全过，几何平均加速比 **1.68x**（大矩阵可达 18x+）

### §3.1 Layer 1: 设计约束

#### L1.1 必须做多 kernel 分支
（特化分派原则见 §1 G3）
- 通用 4D kernel 的逐元素坐标解码 overhead 极大，仅作为兜底方案
- **必须**为高频场景（2D/3D constant）写特化 kernel

#### L1.2 constant 模式必须拆分为 fill + copy
- **禁止**在 kernel 内逐元素判断 `if in_bounds else fill_value`
- **必须**先 `output.fill_(value)`，再用 copy kernel 搬运有效数据

#### L1.3 Host 侧必须做维度压缩
- **必须**在调用 kernel 前 squeeze 前导 size-1 维度
- 压缩后需同步调整 pad_list 的维度对应关系

#### L1.4 坐标比较必须用 float32
（见 §1 G8）

#### L1.5 禁止硬编码 num_cores
（见 §1 G1）

### §3.2 Layer 2: 算法骨架

#### L2.1 Host 侧分支决策树（伪代码）
```
ndim = squeeze(x) 后的维度
mode = constant/reflect/replicate/circular
if ndim == 2:
    if mode == constant:
        output.fill_(value)
        launch copy_kernel_2d
    else:
        launch pad_kernel_2d  # 逐行边界映射
elif ndim == 3:
    if mode == constant:
        output.fill_(value)
        launch copy_kernel_3d
    else:
        if D0 * D1_out > THRESHOLD:  # THRESHOLD ~ 3000
            launch pad_kernel_3d_nonconstant_v2  # 1D grid
        else:
            launch pad_kernel_3d_nonconstant_2d   # 2D grid
else:
    pad_to_4d()
    launch pad_kernel_4d  # 通用兜底
```

#### L2.2 多核并行骨架模式

**模式 A - 按元素分配（适合通用/1D 场景）**:
```
elements_per_core = cdiv(total_elements, num_cores)
core_start = pid * elements_per_core
core_end = min(core_start + elements_per_core, total_elements)
for block_idx in range(cdiv(core_end - core_start, BLOCK_SIZE)):
    # 处理一个 block
```

**模式 B - 按行分配（适合 2D/3D 场景）**:
```
rows_per_core = cdiv(total_rows, num_cores)
row_start = pid * rows_per_core
row_end = min(row_start + rows_per_core, total_rows)
for row_idx in range(row_end - row_start):
    # 处理一行，内部按 block 遍历列
```

### §3.3 Layer 3: 关键技巧

#### L3.1 坐标压缩公式（constant 模式）
```python
# 将多维坐标压缩为 1D 偏移，避免 kernel 内多维索引计算
coords = tl.arange(0, BLOCK_SIZE)
# 逐维度解码（仅对需要处理的维度）
d0 = coords // stride_d0
d1 = (coords % stride_d0) // stride_d1
# ...
```

#### L3.2 边界映射模板（reflect/replicate/circular）
```python
# reflect 模式：先转 float 比较，再映射
in_coord = out_coord - pad_left
in_coord_f = in_coord.to(tl.float32)
# 处理负边界
in_coord = tl.where(in_coord_f < 0, -in_coord, in_coord)
# 处理超界
in_coord = tl.where(in_coord_f >= in_dim, 2 * (in_dim - 1) - in_coord, in_coord)
```

#### L3.3 2D constant 特化 kernel 结构
```python
# Host 侧：先 fill，再启动 copy kernel
output = torch.empty(...)
output.fill_(fill_value)
# copy kernel：仅搬运有效数据，无边界判断
copy_kernel[grid](input, output, ...)
```

#### L3.4 Block Size Scaling（关键，决定能否过目标加速比）
- Pad 为纯内存搬运算子，**BLOCK cap 从默认 512 提升到 16384 可显著降低 kernel launch 次数、提高访存效率**。
- 实测 51 cases：cap=512 时 geomean ~1.08x；cap=1024 → 1.14x；cap=2048 → 1.20x；cap=4096/8192 → 1.20x；**cap=16384 → 1.25x**；cap=32768 时大 W 形状出现 MLIR 编译错误。
- **推荐策略**：在 verify 不失败的范围内选择最大 cap；典型有效区间为 **2048~16384**，需以实际 case 集合验证上限。
- 注意：cap 作为 `_select_block` 的可调参数，只需在 forward() 中修改变量值即可，无需改动 kernel 内部逻辑。

### §3.4 Pad 性能基准
- 几何平均 **1.68x**，51/51 cases 全过（大矩阵可达 18x+）

---
