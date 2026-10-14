---
type: Optimization
title: 性能优化实践
description: 优先使用一维 grid (n,)，便于与 Ascend 核心映射一致。
tags: [task.optimization, topic.implementation.masking, topic.compilation, topic.tiling, topic.compute.reduction, topic.compute.elementwise.arithmetic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 性能优化实践

## 5. 性能优化

### 5.1 1D Grid 优先

优先使用一维 grid `(n,)`，便于与 Ascend 核心映射一致。

### 5.2 核内循环优化

编译器会对核内 `for` 循环做自动流水线优化，实现存算并行：

```python
@triton.jit
def optimized_elementwise(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    for i in range(4):
        offsets = (pid * 4 + i) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)
```

### 5.3 算子拆分策略

复杂算子可拆分为多个 kernel，减少单 kernel 复杂度，便于调优。

### 5.4 NPU 核心配置

- Element-wise / Reduce：使用 VEC 核心，`grid=(VEC_CORE_NUM,)`
- MatMul：使用 CUBE 核心，`grid=(CUBE_CORE_NUM,)`

### 5.5 数据传输单位

Ascend NPU 数据传输以 **256B** 为单位，设计 block 大小时需考虑对齐。

### 5.6 数值稳定性

- **Softmax 防溢出**：先减最大值再 `exp`
- **sqrt 非负检查**：`tl.maximum(variance, 0.0)` 再开方
- **累加用 float32**：reduce 累加建议用 `tl.float32` 避免精度损失

```python
# 错误：直接 exp 可能溢出
scores = tl.exp(x)

# 正确：减去最大值
max_val = tl.max(x, axis=0)
scores = tl.exp(x - max_val)

# 方差开方前确保非负
variance = tl.maximum(variance, 0.0)
rstd = 1.0 / tl.sqrt(variance + eps)
```

### 5.7 API 限制（Ascend 特有）

#### 禁止使用的语法
- 禁止 `return`, `break`, `continue` → 使用mask控制
- 禁止 lambda表达式 → 使用内联函数或tl.where
- 禁止 链式布尔运算 → 分步计算mask
- 禁止 张量直接索引 → 使用tl.load/tl.store

#### tl.constexpr 正确用法
- **仅在内核参数中使用**: `BLOCK_SIZE: tl.constexpr`
- **不可在host侧使用**: 启动函数中不可用tl.constexpr

| 限制 | 说明 |
|------|------|
| 无 return/break/continue | 使用变量和条件控制流程 |
| 无 lambda | 使用普通函数或内联逻辑 |
| 无 while 循环 | 用 `for` + `if` 替代 |
| 无 Python 切片 | 用 `tl.get_element` / `tl.extract_slice` / `tl.insert_slice` |
| constexpr 仅用于 kernel 参数 | 编译时常量通过参数传入 |
| 避免 tl.where 做内存偏移 | 易导致低效，改用 boundary_check 或 mask |
| 标量类型转换 | 用 `.to(tl.float32)` 等，不用 `tl.float16()` |

```python
# 不推荐：tl.where 用于偏移
# ptr = base + tl.where(mask, idx, 0)

# 推荐：mask 用于 load/store
data = tl.load(ptr + idx, mask=mask, other=0.0)
```

---

## 失效边界与不触发场景

- **不触发**：算子不含本卡所列代码特征时不适用。
- **失效**：切分策略或 kernel 结构变更后，本卡基于旧结构的结论需重新判定。
