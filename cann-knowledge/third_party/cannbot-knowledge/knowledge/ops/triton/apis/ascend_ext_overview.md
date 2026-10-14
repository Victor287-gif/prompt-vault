---
type: API
title: Ascend 扩展 API 总览与导入方式
description: Triton-Ascend 扩展 API 全景：导入方式、与标准 Triton API 的差异、适用场景与完整 API 分类索引。
tags: [task.lookup, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.synchronization, topic.implementation.memory_management, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/__init__.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---
# Ascend 扩展 API 总览与导入方式

Triton-Ascend 在标准 OpenAI Triton 之上提供一套 Ascend 扩展 API，暴露昇腾 NPU 硬件特性（Cube/Vector 双核、L0C/L1/UB 多级存储、NZ 数据排布）。需要极致性能或底层控制时使用。

## 导入方式

```python
# 全量导入
from triton.language.extra.cann.extension import *

# 按需导入（常用约定别名 al）
import triton.language.extra.cann.extension as al

# Buffer 编程模型（约定别名 bl）
import triton.extension.buffer.language as bl

# libdevice 数学函数库
from triton.language.extra.cann import libdevice
```

## 扩展 API 与标准 Triton API 的差异

| 维度 | 标准 Triton API | Ascend 扩展 API |
|------|----------------|----------------|
| 编程模型 | 基于 Tensor 的 SPMD 模型 | Tensor + Buffer 双模型 |
| 硬件抽象 | 通用 GPU（SM、Shared Memory） | 昇腾 NPU（Cube/Vector、UB/L1/L0C） |
| 同步机制 | `tl.debug_barrier` | `sync_block_set/wait/all`（Cube-Vector 间） |
| 数据搬运 | `tl.load/store` | `fixpipe`、`copy`、Buffer `alloc/to_buffer/to_tensor` |
| 类型转换 | `tl.cast` | 扩展 `cast`（支持 `overflow_mode`、`fp_downcast_rounding`） |
| 索引操作 | `tl.load` + 指针算术 | `index_put`、`gather_out_to_ub`、`scatter_ub_to_out`、`index_select_simd` |
| 自定义算子 | 不支持 | `@register_custom_op` + `al.custom()` |

## 何时使用扩展 API

1. **Cube-Vector 协同计算**：同时利用 Cube 与 Vector 单元时必须用 `scope`、`sync_block_set/wait`。
2. **矩阵乘法后处理**：`fixpipe` 将 Cube 结果从 L0C 搬到 UB（L0C→UB 仅 910_95），实现零拷贝量化/ReLU 融合。
3. **精细内存控制**：Buffer 模型直接操作 UB/L1 片上存储，避免多余 GM 访问。
4. **SIMD gather/scatter**：`gather_out_to_ub`/`scatter_ub_to_out` 实现高效索引读写。
5. **编译器优化提示**：`compile_hint`、`multibuffer` 指导流水线优化。
6. **自定义算子集成**：`register_custom_op` 注册 CANN 自定义算子（bitcode）。

## 完整 API 分类索引

### 核心枚举与常量

| API | 说明 |
|-----|------|
| `PIPE` | 流水线类型（PIPE_S/PIPE_V/PIPE_M/PIPE_MTE1/PIPE_MTE2/PIPE_MTE3/PIPE_FIX/PIPE_ALL） |
| `CORE` | 核心类型（VECTOR/CUBE/CUBE_OR_VECTOR/CUBE_AND_VECTOR） |
| `MODE` | 执行模式（SIMD/SIMT/MIX） |
| `IteratorType` | 迭代器类型（Parallel/Broadcast/Transpose/Reduction/Gather 等 12 种） |
| `SYNC_IN_VF` | Vector 内部同步模式（VV_ALL/VST_VLD 等 12 种） |
| `int64` | 自定义算子中指定 int64 参数的类型包装 |
| `ascend_address_space` | NPU 缓冲区地址空间（UB/L1/L0C/GM） |
| `builtin` / `is_builtin` | Ascend 内建函数装饰器 / 检查函数 |

### 同步操作

| API | 说明 |
|-----|------|
| `sync_block_set` | 生产者核心发送同步信号 |
| `sync_block_wait` | 消费者核心等待同步信号 |
| `sync_block_all` | 全局屏障同步 |
| `debug_barrier` | Vector 内部流水线同步（SYNC_IN_VF） |

### Cube-Vector 协同

| API | 说明 |
|-----|------|
| `scope` | 上下文管理器，指定代码块运行在 Cube 或 Vector 核心 |
| `sub_vec_id` | 获取子向量核 ID（0 或 1） |
| `sub_vec_num` | 获取子向量核数量（通常 2） |

### fixpipe 操作

| API | 说明 |
|-----|------|
| `fixpipe` | L0C 到 UB 的数据搬运（支持 NZ 格式转换，仅 910_95） |
| `FixpipeDMAMode` | DMA 传输模式（NZ2ND/NZ2DN/NZ2NZ） |
| `FixpipeDualDstMode` | 双目的模式（NO_DUAL/COLUMN_SPLIT/ROW_SPLIT） |
| `FixpipePreQuantMode` | 预量化模式（NO_QUANT/F322BF16/F322F16/S322I8） |
| `FixpipePreReluMode` | 预 ReLU 模式（NO_RELU/NORMAL_RELU/LEAKY_RELU/P_RELU） |

### 内存拷贝 / 自定义算子 / 辅助操作

| API | 说明 |
|-----|------|
| `copy` | UB 到 UB/L1 的数据拷贝（`copy_from_ub_to_l1` 已废弃，用 `copy` 替代） |
| `custom` / `register_custom_op` | 调用 / 注册自定义算子 |
| `parallel` | 并行范围声明（支持 `bind_sub_block`） |
| `compile_hint` | 编译器提示标注 |
| `multibuffer` | 多缓冲设置（ping-pong 流水线） |

### 向量操作

| API | 说明 |
|-----|------|
| `insert_slice` / `extract_slice` | 子张量插入 / 提取 |
| `get_element` | 按索引取单个元素 |
| `sort` / `flip` | 排序（仅最后一维）/ 翻转 |
| `cast` | 类型转换扩展版（支持 `overflow_mode`） |

### 内存操作（GM↔UB 索引读写）

| API | 说明 |
|-----|------|
| `index_put` | 索引写入（UB → GM） |
| `gather_out_to_ub` | 索引读取（GM → UB） |
| `scatter_ub_to_out` | 索引散列写入（UB → GM） |
| `index_select_simd` | SIMD 并行索引选择（GM → UB，零拷贝） |

### 数学操作与 libdevice

`atan2`、`isfinited`（fp64/fp32/fp16/bf16 有限值检测）、`finitef`（仅 fp32）。libdevice 提供 HMF 硬件加速高精度数学函数（`pow`/`tanh`/`log1p`/`expm1`/`erfinv`/`fast_dividef` 等），详见 [ascend_ext_libdevice.md](./ascend_ext_libdevice.md)。

### Buffer 编程模型

| API | 说明 |
|-----|------|
| `bl.address_space` / `bl.buffer_type` / `bl.buffer` | 地址空间基类 / 缓冲区类型 / 缓冲区对象 |
| `bl.alloc` | 分配缓冲区（不支持 int1） |
| `bl.to_buffer` / `bl.to_tensor` | Tensor ↔ Buffer 零拷贝视图转换 |
| `bl.subview` | 创建缓冲区子视图（32 字节对齐约束） |


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [__init__.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/__init__.py)
- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py)
- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/extension/buffer/language/core.py)
