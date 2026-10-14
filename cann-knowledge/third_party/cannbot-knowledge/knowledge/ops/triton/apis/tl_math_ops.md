---
type: API
title: 数学运算 API 及 NPU 精度提升规则
description: Triton-Ascend 数学运算 API：算术、初等函数、融合运算（fma/sigmoid/softmax）签名与类型提升规则；bf16/fp16
  在 NPU 上的精度提升差异。
tags: [task.lookup, topic.compute.elementwise.arithmetic, topic.programming_model.execution_unit.vector, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/add.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# 数学运算 API 及 NPU 精度提升规则

NPU 上数学运算由 Vector 计算单元执行（对应 GPU CUDA Core）。关键差异：bf16 的 FMAX/FMIN/FCMP 不支持需提升 fp32；fp16/bf16 除法自动提升 fp32。

## 算术运算

```python
tl.add(x, y, sanitize_overflow=True)   # +；指针操作数时执行 addptr
tl.sub(x, y, sanitize_overflow=True)   # -
tl.mul(x, y, sanitize_overflow=True)   # *
```

`sanitize_overflow`（默认 True）控制整数溢出检测。

| 除法变体 | 签名 | 说明 |
|---------|------|------|
| `truediv`（`/`） | `x / y` | 真除法：整数自动提升 fp32；fp16/bf16 提升 fp32 |
| `fdiv` | `tl.fdiv(x, y, ieee_rounding=False)` | 快速除法，不保证 IEEE 精度 |
| `div_rn` | `tl.div_rn(x, y)` | 精确除法，IEEE 四舍五入，支持 bf16/fp16/fp32/fp8/fp64 |

其他：`%`（mod，浮点 frem / 有符号 srem / 无符号 urem，不同符号整数不能直接取模）、`tl.cdiv(x, div)`（向上取整除法）。

## 初等函数

均支持 bf16/fp16/fp32/fp8e4nv/fp8e5/fp64（NPU 上 fp64/uint 受硬件限制，见支持矩阵）：

| 函数 | 说明 | 函数 | 说明 |
|------|------|------|------|
| `tl.exp` / `tl.exp2` | e^x / 2^x（近似计算） | `tl.log` / `tl.log2` | 自然对数 / log2 |
| `tl.sin` / `tl.cos` | 正弦 / 余弦 | `tl.sqrt` / `tl.sqrt_rn` | 快速 / IEEE 精确平方根 |
| `tl.rsqrt` | 1/sqrt(x)，比 `1/tl.sqrt` 更快更准 | `tl.erf` | 误差函数 |
| `tl.abs` | 绝对值（所有数值类型） | `tl.floor` / `tl.ceil` | 向下 / 向上取整（浮点；整数 ceil 为 no-op） |

## 融合与其他运算

```python
tl.fma(x, y, z)            # x * y + z，单次舍入，精度高于分步
tl.sigmoid(x)              # 1 / (1 + exp(-x))，JIT 高层函数
tl.softmax(x, ieee_rounding=False)  # exp(x-max)/sum(exp(x-max))，JIT 高层函数
```

| 函数 | 说明 |
|------|------|
| `tl.umulhi(x, y)` | 2N 位乘积高 N 位；仅 int32/int64/uint32/uint64（NPU 上仅 i32 且不支持负数输入） |
| `tl.maximum(x, y, propagate_nan=NONE)` / `tl.minimum` | 逐元素最大/最小 |
| `tl.clamp(x, min, max, propagate_nan=NONE)` | 限制在 [min, max] |

## 类型提升规则（computation_type_impl）

1. 任一操作数为 fp64 → 结果 fp64
2. 任一操作数为 fp32 → 结果 fp32
3. 任一操作数为 fp16 → 结果 fp16；**除法提升 fp32**
4. 两操作数均 bf16 → 结果 bf16；bf16 与其他类型运算提升 fp32；**bf16 除法提升 fp32**
5. 不同 fp8 类型运算提升 fp16

## NPU 适配要点

1. **bf16 提升 fp32**：NPU Vector 单元不支持 bf16 的 FMAX/FMIN/FCMP，`maximum`/`minimum`/比较运算自动经 `_promote_bfloat16_to_float32` 提升 fp32。需 bf16 结果时显式 `.to(tl.bfloat16)`。
2. **fp16/bf16 除法提升**：`/`、`fdiv` 自动提升 fp32 执行（Vector 单元不原生支持低精度除法），需低精度结果时计算后显式转回。
3. **精度选择**：`sqrt`/`fdiv` 快速版不满足 IEEE 精度；需精确结果用 `sqrt_rn`/`div_rn`（更慢）。
4. **sigmoid/softmax** 为 JIT 组合实现而非硬件指令，中间计算受精度提升规则影响。

## 代码示例

```python
@triton.jit
def gelu_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    inner = x * 0.7978845608028654 * (1.0 + 0.044715 * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = 0.5 * x * (1.0 + tanh_inner)
    tl.store(out_ptr + offsets, result, mask=mask)
```

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [math.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/math.py)
- [semantic.py#L46-L109](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L46-L109)
- [semantic.py#L219-L352](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L219-L352)
- [core.py#L2100-L2106](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L2100-L2106)
- [standard.py#L43-L57](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/standard.py#L43-L57)
