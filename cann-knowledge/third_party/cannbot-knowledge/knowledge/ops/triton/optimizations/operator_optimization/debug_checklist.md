---
type: Optimization
title: 调试检查清单
description: 调试检查清单 的 Triton Ascend 编码要点。
tags: [task.optimization, topic.compilation, topic.tiling, topic.implementation.masking, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 调试检查清单

## 6. 调试清单

### 6.1 内存访问清单

- [ ] 所有 load/store 是否有正确的 mask 或 boundary_check
- [ ] 指针偏移是否可能越界
- [ ] 是否使用 `.contiguous()` 保证连续访问
- [ ] Block 大小是否满足 256B/512B 对齐

### 6.2 控制流清单

- [ ] 是否使用了 `return` / `break` / `continue`（Ascend 不支持）
- [ ] 是否使用了 `while`（需改为 `for` + `if`）
- [ ] 是否使用了 lambda
- [ ] 是否使用了 Python 切片（需改为 tl 切片 API）

### 6.3 Grid/Block 清单

- [ ] Grid 各维度是否 ≤ 65535
- [ ] BLOCK_SIZE 是否 < 65536
- [ ] 大 shape 是否采用交错循环或连续块法
- [ ] 核心数是否在 `__init__` 中获取

### 6.4 并发清单

- [ ] 多 kernel 调用时中间张量是否正确传递
- [ ] 原子操作是否正确使用（如有）

### 6.5 切片清单

- [ ] 多维索引是否正确计算
- [ ] 是否使用 `tl.extract_slice` / `tl.insert_slice` 替代 Python 切片

### 6.6 性能清单

- [ ] 是否优先 1D grid
- [ ] Element-wise 是否使用 VEC 核心
- [ ] MatMul 是否使用 CUBE 核心
- [ ] 是否利用核内循环实现存算并行

### 6.7 常见错误速查表

| 错误现象 | 可能原因 | 排查方向 |
|----------|----------|----------|
| 编译失败：unsupported op | 使用了 Ascend 不支持的 API | 检查 return/break/while/lambda/切片 |
| 数值不一致 | 精度、reduce 维度、边界处理 | 检查 mask、axis、float32 累加 |
| 越界 / 崩溃 | 指针或索引计算错误 | 检查 offset 公式、mask 覆盖 |
| Grid 超限 | grid 维度 > 65535 | 使用交错循环或拆分 grid |
| 性能差 | 访存不连续、block 过小 | 使用 contiguous、调整 BLOCK_SIZE |
| tl.where 相关错误 | Ascend 上 tl.where 用于偏移易出问题 | 改用 mask + other 或 boundary_check |

### 6.8 常见错误解决表

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| 内存越界访问 | 运行时错误、结果异常、随机崩溃 | load/store缺少mask或boundary_check | 添加正确的mask或boundary_check保护 |
| Grid超限 | 编译失败或运行时错误 | grid总大小超过65535 | 使用交错循环`for i in range(pid, total, core_num)`或连续分块处理 |
| 控制流错误 | 编译失败、语法错误 | 使用了return/break/continue | 移除禁用语句，使用mask控制流程 |
| while循环错误 | 编译失败（Ascend后端） | 使用了while循环 | 改用for + if替代：`for i in range(MAX): if i < n:` |
| 切片语法错误 | 编译失败 | 使用了`b[0]`或`b[i:j]`直接切片 | 使用`tl.get_element`或`tl.extract_slice` |
| tl.arange索引错误 | 编译失败 | 对`tl.arange`结果使用`get_element` | 直接计算索引值而非提取 |
| 类型转换错误 | 编译警告或错误 | 使用`tl.float16(scalar)`转换 | 改用`scalar.to(tl.float16)` |
| constexpr误用 | 编译失败 | 在host侧使用tl.constexpr | 仅在kernel参数中使用tl.constexpr |
| Stride设置错误 | 计算结果错误、数据错位 | stride参数计算或传递错误 | 验证stride设置，检查tensor.stride() |
| 数值不稳定 | 结果为NaN或Inf | softmax/sqrt等操作溢出 | 减去最大值、检查非负、使用float32 |
| 数据竞争 | 结果不确定、每次运行不同 | 多program并发写入同一位置 | 使用tl.atomic_add等原子操作 |
| BLOCK_SIZE过大 | 编译失败或运行时错误 | BLOCK_SIZE超过65536或硬件限制 | 减小BLOCK_SIZE，使用循环处理 |
| tl.where偏移计算 | 编译失败（Ascend后端） | 在内存偏移中使用tl.where | 改用if-else静态分支处理 |
| 性能低下 | 运行缓慢 | 内存访问不连续、切分不合理 | 优化内存布局、调整BLOCK_SIZE、使用block_ptr |

---

## 失效边界与不触发场景

- **不触发**：算子不含本卡所列代码特征时不适用。
- **失效**：切分策略或 kernel 结构变更后，本卡基于旧结构的结论需重新判定。
