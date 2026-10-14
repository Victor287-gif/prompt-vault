---
type: Optimization
title: 标量降级规避（910_95 / 910B）
description: HIVM 向量操作标量降级（Scalar Lowering）的触发条件、910_95 Reg-based 与 910B Mem-based
  差异速查表及 int64/整数比较/归约的规避代码模式。
tags: [task.optimization, topic.programming_model.execution_unit.scalar, topic.programming_model.execution_unit.vector, topic.compute.reduction]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/cv_autotune/hardware_consts.py
  role: constraint
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# 标量降级规避（910_95 / 910B）

## 触发条件

kernel 性能远低于预期（10x-100x 损失）时首先排查标量降级：

- 执行时间与 CPU 标量循环相当
- HIVM IR 中大量 `scf.for` 包裹 `arith` 标量操作
- 涉及 i64、整数比较、特定归约模式

机制：编译器经 `ImplByScalarOpInterface.shouldLowerToScalarLoops()` 判定，由 `HIVMLowerToLoopsPass` 将向量操作退化为逐元素标量循环。是功能等价变换，只影响性能；不建议禁用降级（硬件不支持时后续编译会失败）。

## 架构分类（判定函数 `isRegBasedArch()` / `isMemBasedArch()`）

| 分类 | 代表芯片 | 核间同步 |
|------|---------|---------|
| Reg-based | 910_95 系列、Ascend310B、950 | 寄存器级指令（SetFlag/WaitFlag） |
| Mem-based | 910B、910_93 | FFTS 内存机制 |

## 降级条件速查表

通用算术（VAdd/VSub/VMul/VMin/VMax/VAbs/VShL/VShR）：f16/f32/i8/i16/i32 SIMD 模式不降级；**i64 始终降级**；SIMT VF 模式全部降级。

VCmp 比较：

| 类型 | EQ | NE | LT/GT/LE/GE |
|------|----|----|-------------|
| f16/f32 | 不降级 | 不降级 | 不降级 |
| i32 | 不降级 | 不降级 | **降级** |
| i8/i16/i64 | **降级** | **降级** | **降级** |

VMulExt 扩展乘法：i32/i64 始终降级（实际总走标量路径，应避免）。

VCumsum/VCumprod 累积：累积维度为最后维度时降级；i64 任何维度降级；累积维度非最后维度（且非 i64）不降级。

归约（核心架构差异）：

| reduceOp | 910_95（Reg-based） | 910B（Mem-based） |
|----------|-------------------|------------------|
| sum/prod/max/min/xori | **不降级（含 i64）** | 仅 i64 降级 |
| max/min_with_index（argmax/argmin） | 仅内存对齐不合法时降级 | i16/i32/i64 降级；f16/f32/bf16 且 flatten rank>2 降级 |
| any/all/ori/andi | 不降级 | 不降级 |

910_95 上 argmax/argmin 的对齐合法性由 `isLegalAccessAlignment` 判定（32B 对齐）；**缺少 `StridedLayoutAttr` 时默认判不合法而降级**（临时 workaround）。

## 规避代码模式

**场景 1：i64 向量算术 → int32**

```python
# 退化：c = a + b（i64 VADD 不支持）
# 规避：load 后转 int32，算完按需转回
a = tl.load(ptr_a + offsets).to(tl.int32)
c = a + b                       # VADD 向量路径
c = c.to(tl.int64)
```

风险：数据须在 `[-2^31, 2^31-1]` 内，否则溢出。

**场景 2：整数比较 → fp32 比较**

```python
# 退化：cols = tl.arange(0, BLOCK_N) 为 i64，cols < N 降级
cols_cmp = tl.arange(0, BLOCK_N).to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)  # vector 路径
```

注意：fp32 精确整数范围 `[-2^24, 2^24]`；i32 的 EQ/NE 无需转换；load/store 的 mask 比较编译器通常自动优化。

**场景 3：tl.where 单点条件 → get_element + insert_slice**

condition 仅一个位置为 false 时：

```python
import triton.language.extra.cann.extension as extension
if base <= y < base + BLOCK_SIZE:
    _offs = y - base
    _val = extension.get_element(B, (_offs,))
    _tensor = tl.full((1,), _val, dtype=A.dtype)
    A = extension.insert_slice(A, _tensor, offsets=(_offs,), sizes=(1,), strides=(1,))
```

**场景 4：归约降级（910B）**

```python
# 910B：i64 归约降级 → 转 fp32
result = tl.sum(tl.load(ptr + offsets).to(tl.float32))
# 910B：整数 argmax 始终降级 → 转 fp32 后 argmax（低维不降级）
idx = tl.argmax(x_int.to(tl.float32), axis=0)
```

910_95 上基本归约（含 i64）不降级、无需规避；argmax/argmin 性能异常时检查输入 tensor 的 stride 信息是否完整。

**场景 5：cumsum 累积维度在最后维度 → 转置**

```python
result_t = tl.cumsum(tl.trans(x), axis=-1)   # 累积维度不再最后
result = tl.trans(result_t)
```

i64 累积任何维度都降级，应避免。

## 910_95 特别注意

- 基本归约 sum/prod/max/min/xori 所有 dtype（含 i64）均走向量路径——与 910B 的核心差异
- argmax/argmin 仅对齐不合法时降级；缺 `StridedLayoutAttr` 默认不合法
- 910_95 系列均属 Reg-based：910_950z/9579/957b/957d/9581/9589/958a/958b/9599 等


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [ShouldLowerToScalarLoops.cpp#L221-L272](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L221-L272)
- [ShouldLowerToScalarLoops.cpp#L139-L219](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp#L139-L219)
- [Utils.cpp#L232-L260](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HACC/Utils/Utils.cpp#L232-L260)
- [ShouldLowerToScalarLoops.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp)
- [LowerToLoops.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/LowerToLoops.cpp)
- [ImplByScalarOpInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.td)
- [Utils.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HACC/Utils/Utils.cpp)
