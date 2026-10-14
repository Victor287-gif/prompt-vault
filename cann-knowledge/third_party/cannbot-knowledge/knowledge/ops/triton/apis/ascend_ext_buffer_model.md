---
type: API
title: Buffer 编程模型（bl.alloc / to_buffer / to_tensor / subview）
description: Triton-Ascend Buffer 编程模型：显式控制 UB/L1/L0C/GM 地址空间的缓冲区分配、Tensor↔Buffer
  零拷贝转换与子视图，是 fixpipe/copy 与 CV 协同的数据共享基础。
tags: [task.lookup, topic.implementation.memory_management, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/builder.py
  role: implementation
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/scope.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# Buffer 编程模型（bl.alloc / to_buffer / to_tensor / subview）

Buffer 模型是标准 Tensor 模型之外的另一种数据抽象：Tensor 关注计算语义（值），Buffer 关注存储语义（物理内存区域），允许直接控制数据在 UB/L1/L0C 的分配与访问。它是 fixpipe、copy 等底层操作与 Cube-Vector 数据共享的基础。

导入：`import triton.extension.buffer.language as bl`

## Buffer 与 Tensor 的区别

| 维度 | Tensor | Buffer |
|------|--------|--------|
| 抽象层次 | 计算语义（值） | 存储语义（内存区域） |
| 地址空间 | 隐式（编译器管理） | 显式（UB/L1/L0C/GM） |
| 数据排布 | 逻辑形状 | 物理排布（含步长） |
| 可写性 | 默认可写 | `writable` 参数控制 |
| 使用场景 | 标准计算操作 | fixpipe/copy/子视图等底层操作 |
| 转换 | `bl.to_buffer(tensor)` | `bl.to_tensor(buffer)` |

## 地址空间（ascend_address_space）

| 地址空间 | 说明 | 容量级别 | 访问速度 |
|----------|------|----------|----------|
| `ascend_address_space.UB` | Unified Buffer，Vector 核心工作存储 | 片上（~1MB） | 最快 |
| `ascend_address_space.L1` | L1 缓存，Cube 核心缓存 | 片上（~1MB） | 快 |
| `ascend_address_space.L0C` | L0C 缓冲区，Cube 输出 | Cube 输出 | 快 |
| `ascend_address_space.GM` | Global Memory | 片外（GB 级） | 慢 |

`ascend_address_space` 是 `ascend_address_space_group` 对象，属性由 `ascend_ir.AddressSpace` 枚举动态生成，可用空间取决于平台。

## API 参考

### bl.alloc — 分配缓冲区

```python
@builtin
def alloc(
    etype: tl.dtype,                 # 元素类型（不支持 int1）
    shape: List[tl.constexpr],       # 缓冲区形状
    _address_space: address_space = None,  # UB/L1 等
    is_mem_unique: bool = False,     # 是否核心独占（多核各持独立副本）
    _builder=None,
) -> buffer
```

`is_mem_unique=True` 时编译器为每个核心分配独立物理内存（添加 `mem_unique` 标注），用于多核并行独立缓冲场景。

### bl.to_buffer — Tensor 转 Buffer

```python
@builtin
def to_buffer(
    tensor: tl.tensor,               # 不支持标量
    space: address_space = None,     # 目标地址空间
    bind_buffer: buffer = None,      # 绑定已有缓冲区
    _builder=None,
) -> buffer
```

`bind_buffer` 语义：tensor 数据绑定到指定 buffer，后续对 tensor 的写直接落入该 buffer，避免额外分配与拷贝。

### bl.to_tensor — Buffer 转 Tensor

```python
@builtin
def to_tensor(
    memref: buffer,
    writable: bool = True,           # 生成 tensor 是否可写
    target_shape=None,               # 与原始形状不同时生成 convert_layout 布局转换
    _builder=None,
) -> tl.tensor
```

### bl.subview — 创建子视图

```python
@builtin
def subview(
    src: buffer,
    offsets: List[tl.constexpr],     # 各维偏移（必须非负）
    sizes: List[tl.constexpr],       # 各维大小
    strides: List[tl.constexpr],     # 各维步长
    _builder=None,
) -> buffer
```

**32 字节对齐约束（硬件限制，违反抛 TypeError）**：
1. 偏移值必须 32 字节对齐；
2. 所有步长必须为 1（要求连续内存访问模式）；
3. 最后一维第二行的起始偏移必须 32 字节对齐。

## 代码示例

### 基本 Buffer 分配与转换

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl

@triton.jit
def buffer_basic_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    x = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))
    ub_buf = bl.alloc(tl.float32, [BLOCK_SIZE], al.ascend_address_space.UB)
    x_buf = bl.to_buffer(x, al.ascend_address_space.UB, bind_buffer=ub_buf)
    result = bl.to_tensor(x_buf) * 2.0
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

### fixpipe 中的 Buffer 使用

```python
ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
al.fixpipe(acc, ub_buf)                      # L0C tensor → UB buffer
acc_tensor = bl.to_tensor(ub_buf)            # 转回 tensor 做向量操作
tl.store(c_ptr + ..., acc_tensor)
```

### 子视图与 UB→L1 拷贝

```python
half_buf = bl.subview(data_buf,
                      offsets=[0, 0],
                      sizes=[BLOCK_M, BLOCK_N // 2],
                      strides=[1, 1])
half_tensor = bl.to_tensor(half_buf)

# UB → L1
ub_buf = bl.alloc(tl.float32, [BLOCK_SIZE], al.ascend_address_space.UB)
l1_buf = bl.alloc(tl.float32, [BLOCK_SIZE], al.ascend_address_space.L1)
data_ub = bl.to_buffer(data, al.ascend_address_space.UB, bind_buffer=ub_buf)
al.copy(data_ub, l1_buf)
```

## NPU 适配要点

1. **to_buffer/to_tensor 零拷贝**：两者仅是视图转换，不产生数据搬运；实际搬运发生在 `copy`、`fixpipe` 中。`to_tensor(target_shape=...)` 例外——形状不同会生成 `convert_layout`。
2. **何时用 Buffer**：需显式控制物理存储位置、使用 fixpipe/copy、Cube-Vector 间共享数据、精细控制布局与步长时。
3. **地址空间选择**：UB 是 Vector 工作存储（向量操作源/目的）；L1 是 Cube 缓存（matmul 输入）；L0C 是 Cube 输出（fixpipe 数据源）；GM 容量最大延迟最高。
4. **不支持 int1**：硬件限制，无法分配 bool 缓冲区。
5. **bind_buffer 生命周期**：buffer 由分配点决定，kernel 结束自动释放。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/extension/buffer/language/core.py)
- [semantic.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/extension/buffer/language/semantic.py)
- [core.py#L143-L163](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L143-L163)
