---
type: Guide
title: Triton API 在 Ascend NPU 上的差异对照
description: Triton API 在 Ascend NPU（含 910_95）上的不支持项与替代方案、行为差异、Ascend 扩展 API（al.*）清单，以及归约精度、tl.where
  标量退化等关键陷阱。
tags: [task.migration, topic.quality.compatibility.version_migration, topic.precision, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# Triton API 在 Ascend NPU 上的差异对照

触发条件：迁移或编写 Triton kernel 时，确认 API 在 NPU 上的行为差异、替代方案或 NPU 特有扩展。

## 1. 不支持的 API 及替代方案

| API | 不支持内容 | 替代方案 | 说明 |
|-----|-----------|---------|------|
| `tl.load` / `tl.store` | uint8/uint16/uint32/uint64/fp64 | int8/int16/int32/int64/fp32 | 硬件限制，需 host 端转换 |
| `tl.load` / `tl.store` | `cache_modifier` / `eviction_policy` / `volatile` | 忽略即可 | NPU 无 GPU L1/L2 cache 层次，传入不报错但无效 |
| `tl.dot` | `input_precision="tf32"` | 使用 `"hf32"` | NPU 不支持 TF32，自动映射为 hf32 |
| `tl.dot` | `input_precision="tf32x3"` | 无替代 | NPU 不支持 tf32x3 |
| `tl.dot` | `out_dtype=bfloat16` | `out_dtype=float32` 后 `.to(tl.bfloat16)` | 需手动 cast |
| `tl.dot` | `max_num_imprecise_acc` | 忽略即可 | 所有累加使用完整精度 |
| `tl.dot` | int16/int32 输入 | 仅 int8/fp16/bf16/fp32 | 矩阵乘输入类型受限 |
| `tl.sort` | 全类型 | `al.sort`（Ascend 扩展） | 毕昇编译器限制 |
| `tl.round` | fp16/bf16 | 仅 fp32 | NPU 限制 |
| `tl.umulhi` | 负数输入 | 避免负数 | 仅支持 i32 |
| `tl.gather` | axis != n-1 | 仅支持最后一维 | NPU 限制 |
| `tl.permute` / `tl.trans` | 不相邻轴转置 | 仅支持相邻轴 | NPU 限制 |
| `tl.cat` | `can_reorder=False` | 必须 `can_reorder=True` | 元素顺序可能重排 |
| `tl.flip` | 非 2 的幂维度 | 翻转维度与 numel 须为 2 的幂 | 基于 bitonic merge 实现 |
| `tl.histogram` | 浮点输入 | 仅 1D 整数 | NPU 限制 |
| `tl.arange` | 非 int32 类型 | 仅 int32 | NPU 限制 |
| `// (floordiv)` | fp16/bf16/fp32 | 仅整数 | 浮点地板除不支持 |
| `tl.where` | int64 | 不支持 | API 支持矩阵中 i64 不支持 |
| `atomic_add` | int64/fp64 | int8/int16/int32/fp16/fp32/bf16 | 硬件限制 |
| `atomic_cas` | bf16 (910B) | 910_95 支持 bf16 | 910_95 类型更多 |
| 所有 atomic op | `sem` 非 `"acq_rel"` | 仅 `"acq_rel"` | NPU 唯一支持的内存序 |
| 所有 atomic op | `scope` 非 `"gpu"` | 仅 `"gpu"` | NPU 唯一支持的同步范围 |
| 部分 atomic op | loop 内使用 | 避免循环内 atomic_and/or/xchg/xor/cas | NPU 限制 |

## 2. 行为不同的 API

| API | GPU 行为 | NPU 行为 | 影响及处理 |
|-----|---------|---------|-----------|
| `tl.load` (mask=False, other=None) | masked 位置值未定义 | `care_padding=True` 填零；`False` 为随机值 | 默认填零；`care_padding=False` 可提升性能 |
| `tl.store` (离散 mask) | 正常写入 | 拆解为 atomic {load, select, store} | 可能有精度/正确性问题，用连续 mask 或 block pointer |
| `tl.dot` 精度 | `"tf32"` 用 19-bit TF32 | `"tf32"` 自动映射 `"hf32"` | hf32 是 NPU 特有半精度 fp32 格式 |
| `tl.dot` 输入精度 | hf32 不适用 | 非 fp32 输入时 hf32 被忽略回退 `"ieee"` | 仅 fp32 输入可用 hf32 |
| `tl.sum`/`tl.max`/`tl.min` | int8/int16 提升为 int32 归约 | int8/int16 **不提升**直接归约 | 可能溢出，手动 `.to(tl.int32)` |
| `tl.sum`/`tl.max`/`tl.min` | bf16 直接归约 | bf16 自动提升 fp32 归约 | NPU 不支持 bf16 FMAX/FMIN/FCMP |
| `tl.maximum`/`tl.minimum` | bf16 直接比较 | bf16 提升 fp32 比较 | 同上 |
| `/ (div)` | fp16/bf16 直接除 | 自动提升 fp32 执行 | Vector 单元不原生支持低精度除法 |
| `tl.max`/`tl.min` | `propagate_nan` 参数 | 支持 `True/False` | `True` 用 maximum（传播 NaN），`False` 用 maxnumf |
| `atomic_max`/`atomic_min` | 浮点需 bitcast 整数实现 | 硬件直接支持浮点 atomic_max/min | NPU 实现更简洁高效 |
| `tl.cast` | 标准 cast | 扩展 `overflow_mode`（`"saturate"` 等） | NPU 特有扩展 |
| `tl.make_block_ptr` | 可调整 stride 顺序实现转置 | 不允许 stride 交换转置 | 转置只能通过 `order` 参数表达 |

## 3. NPU 特有 API（Ascend 扩展）

导入：`from triton.language.extra.cann import extension as al`

| API | 类别 | 说明 | 典型用法 |
|-----|------|------|---------|
| `al.compile_hint` | 编译提示 | 指导编译器优化 | `al.compile_hint(tensor, "dot_pad_only_k")` |
| `al.get_element` | 向量操作 | 取张量单个元素 | `al.get_element(tensor, indices)` |
| `al.insert_slice` | 向量操作 | 插入子张量 | `al.insert_slice(tensor, sub, offsets)` |
| `al.extract_slice` | 向量操作 | 提取子张量 | `al.extract_slice(tensor, offsets, sizes)` |
| `al.sort` | 向量操作 | 排序（替代 tl.sort） | `al.sort(tensor, dim, descending)` |
| `al.flip` | 向量操作 | 沿维度翻转 | `al.flip(tensor, dim)` |
| `al.cast` | 类型转换 | 支持 overflow_mode | `al.cast(t, dtype, overflow_mode="saturate")` |
| `al.fixpipe` | Cube 后处理 | L0C→UB 搬运（仅 910_95） | `al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2ND)` |
| `al.sync_block_set` | 同步 | 生产者核发同步信号 | `al.sync_block_set("cube", "vector", 0)` |
| `al.sync_block_wait` | 同步 | 消费者核等同步信号 | `al.sync_block_wait("cube", "vector", 0)` |
| `al.sync_block_all` | 同步 | 全局屏障 | `al.sync_block_all("all", 0)` |
| `al.scope` | CV 协同 | 指定代码块运行在 Cube/Vector | `with al.scope(core_mode="cube"):` |
| `al.multibuffer` | 存算并行 | 多缓冲 ping-pong | `al.multibuffer(tensor, 2)` |
| `al.parallel` | 并行迭代 | 支持 bind_sub_block | `for i in al.parallel(0, N, bind_sub_block=True):` |
| `al.index_put` | 内存操作 | 索引写入（UB→GM） | `al.index_put(dst, indices, values)` |
| `al.gather_out_to_ub` | 内存操作 | 索引读取（GM→UB） | `al.gather_out_to_ub(src, indices)` |
| `al.scatter_ub_to_out` | 内存操作 | 索引散写（UB→GM） | `al.scatter_ub_to_out(dst, indices, values)` |
| `al.index_select_simd` | 内存操作 | SIMD 并行索引选择 | `al.index_select_simd(src, indices)` |
| `al.int64` | 类型包装 | 自定义算子指定 int64 参数 | `al.int64(value)` |
| `tl.PropagateNan` | 枚举 | NaN 传播控制 `NONE`/`ALL` | `tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)` |
| `al.Pipe`/`al.CORE`/`al.MODE` | 枚举 | 流水线/核心/执行模式 | `al.CORE.CUBE_AND_VECTOR` |
| `al.FixpipeDMAMode` | 枚举 | fixpipe DMA 模式 | `al.FixpipeDMAMode.NZ2ND` |
| `al.FixpipePreQuantMode` | 枚举 | fixpipe 预量化 | `al.FixpipePreQuantMode.F322BF16` |
| `al.FixpipePreReluMode` | 枚举 | fixpipe 预 ReLU | `al.FixpipePreReluMode.NORMAL_RELU` |

## 4. 关键细节

### 4.1 tl.dot 分块对齐（910_95）

VV 类尾轴 32B 对齐（fp16: 16 的倍数；fp32: 8 的倍数）；CV 类尾轴 512B 对齐（fp16/bf16: 256 的倍数；fp32: 128 的倍数）。推荐 `BLOCK_M=128, BLOCK_N=256, BLOCK_K=256`（fp16/bf16 时满足 512B）。mask 仅在 K 维越界时用 `al.compile_hint(a, "dot_pad_only_k")` 减少 padding。

### 4.2 归约精度要求

| 输入类型 | NPU 行为 | 建议 |
|---------|---------|------|
| fp32 / bf16 / int32 | 直接归约（bf16 自动升 fp32） | 正常使用 |
| fp16 | `tl.sum` 直接 fp16 归约；max/min 自动升 fp32 | sum 前手动 `.to(tl.float32)` |
| int8 / int16 | **不提升**，直接归约，可能溢出 | 手动 `.to(tl.int32)` 后归约 |

### 4.3 tl.where 标量退化

Vector CMP 不支持 int64/int32。`tl.where(cols < N, ...)` 中 `tl.arange` 返回 int32 会退化为 scalar：

```python
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)   # CMP 使用 Vector 单元
```

诊断：msProf 看 `aiv_scalar_ratio`，异常高即可能存在标量退化。

### 4.4 类型提升规则（NPU 特有）

bf16 与非 bf16 运算、bf16/fp16 除法、bf16 比较/maximum/minimum/归约均自动提升 fp32；int8/int16 归约**不提升**（与 GPU 不同）。性能选型：Vector ADD 用 int32 不用 int64；Vector CMP 用 fp32；矩阵乘输入优先 bf16、累加 fp32。

## 5. 910_95 与 910B 差异速查

| 维度 | 910B | 910_95 |
|------|------|--------|
| UB 容量 | 192KB / 96KB(double buffer) | 256KB / 128KB(double buffer) |
| L0C 容量 | 128KB | 256KB |
| fixpipe | 不支持 | 支持 L0C→UB 直通 |
| MultiBuffer | 默认开启 | 默认关闭 |
| FP8 | 不支持 | 支持 dot_scaled 与类型转换 |

## 6. 常见迁移陷阱速查

| 陷阱 | 症状 | 解决 |
|------|------|------|
| `tl.where` 标量退化 | aiv_scalar_ratio 异常高 | 比较操作数转 fp32 |
| int8/int16 归约溢出 | 归约结果错误 | 归约前 `.to(tl.int32)` |
| UB 溢出 | `ub overflow` 编译错误 | 减 BLOCK_SIZE 或关 MultiBuffer |
| coreDim 超限 | `coreDim > UINT16_MAX` | 增大 BLOCK_SIZE 或 `TRITON_ALL_BLOCKS_PARALLEL=1` |
| 尾轴不对齐 | 性能严重下降 | VV: 32B 对齐；CV: 512B 对齐 |
| 离散 mask store | 正确性问题 | 连续 mask 或 block pointer |
| `tl.sort` 不可用 | 编译错误 | 用 `al.sort` |
| `out_dtype=bfloat16` | ValueError | fp32 输出后 `.to(tl.bfloat16)` |
| atomic_add 多核累加 | 结果不正确 | NPU 不支持多核 add+保存中间结果 |

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L1589-L1726](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1589-L1726)
- [core.py#L1518-L1581](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1518-L1581)
- [core.py#L24](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L24)
- [core.py#L1954-L1975](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1954-L1975)
- [semantic.py#L1559-L1626](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1559-L1626)
- [semantic.py#L46-L109](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L46-L109)
- [aux_ops.py#L114-L151](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/aux_ops.py#L114-L151)
- [vec_ops.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/vec_ops.py)
- [compiler.py#L68-L69](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L68-L69)
