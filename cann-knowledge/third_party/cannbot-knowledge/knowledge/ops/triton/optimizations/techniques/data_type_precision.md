---
type: Optimization
title: 数据类型与精度保护（910_95 平台）
description: 910_95 平台数据类型支持速查、归约升精度规则、matmul 混合精度模式、bf16 精度保护、类型转换开销与精度验证参数。
tags: [task.optimization, topic.precision, topic.compute.reduction, topic.compute.composite, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Creation_Ops/cast.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# 数据类型与精度保护（910_95 平台）

> 触发条件：处理数据类型转换或精度保护时查阅。

## 910_95 数据类型速查

完全支持（512B 对齐元素数）：int8(1B,512)、int16(2B,256)、int32(4B,128)、int64(8B,64，Vector ADD/CMP 退化 scalar)、fp16(2B,256)、bf16(2B,256)、fp32(4B,128)、bool(内部转 int8)。

部分支持：fp8e4nv/e4b15/e5/e4b8/e5b16 支持类型转换，**910_95 额外支持 `tl.dot_scaled` 的 FP8 输入**；uint8 在 Block Pointer 场景不支持。

不支持：fp64（用 fp32）、uint16/32/64（用 int16/32/64）。

## 归约升精度规则（核心：所有归约在 FP32 下进行）

| 操作 | 输入 | NPU (910_95) 行为 | 正确做法 |
|------|------|-------------------|---------|
| `tl.sum` | fp16 | 直接 fp16 归约（精度不足） | **手动 `.to(tl.float32)`** |
| `tl.sum` | bf16 | 编译器自动升 fp32 | 无需处理 |
| `tl.sum` | int8/int16 | 直接归约（可能溢出） | 手动 `.to(tl.int32)` |
| `tl.max`/`tl.min` | fp16/bf16 | 自动升 fp32（bitwidth<32 浮点） | 无需处理 |
| `tl.dot` 累加 | fp16/bf16 | 硬件默认 fp32 累加 | 无需处理 |

标准 LayerNorm 写法：

```python
x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)  # load 后即升精度
mean = tl.sum(x, axis=0) / N
xbar = tl.where(cols.to(tl.float32) < N, x - mean, 0.0)
var = tl.sum(xbar * xbar, axis=0) / N
rstd = 1 / tl.sqrt(var + eps)
tl.store(Out + cols, (x - mean) * rstd, mask=mask)          # 写回降精度
```

## 矩阵乘法混合精度

```
存储(fp16/bf16) ─load→ 计算(fp16/bf16) ─dot→ 累加器(fp32) ─store→ 存储(fp16/bf16)
```

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    acc += tl.dot(a, b)   # fp16/bf16 输入自动累加到 fp32
tl.store(c_ptr, acc.to(tl.float16), mask=...)
```

`tl.dot` 精度参数：`input_precision` 默认 `"ieee"`，可选 `"hf32"`（仅 fp32 输入生效）；`allow_tf32` 在 NPU 上映射为 hf32。acc 不支持 FP16；`max_num_imprecise_acc` 不支持。dot 输入组合：int8×int8→int32；fp16/bf16/fp32 同型 → fp32 acc。

## BF16 精度保护

1. bf16 归约/比较/除法取模自动升 fp32（NPU 无 bf16 FMAX/FMIN/FCMP），有额外转换开销
2. bf16 与 fp16 混合运算升 fp32——同一 kernel 统一一种 16-bit 类型
3. libdevice 的 `acos/asin/sinh/cosh/acosh/asinh/atanh/atan2/hypot` 不支持 bf16，先 `.to(tl.float32)`
4. bf16 尾数仅 7 位（~2.1 位十进制），关键中间结果在 fp32 计算
5. 转换开销：bf16↔fp32 低（仅截尾数）；fp16↔fp32 中；bf16↔fp16 中（经 fp32 中转）

舍入与溢出控制：

```python
y = tl.cast(x, tl.bfloat16)                                  # 默认 rtne
y = tl.cast(x, tl.bfloat16, fp_downcast_rounding="rtz")      # rtz
y = tl.cast(x, tl.int8)                                      # 默认 trunc
y = tl.cast(x, tl.int8, overflow_mode="saturate")            # 饱和（Ascend 扩展）
```

## care_padding（NPU 专属，默认 True）

| care_padding | other | padding 行为 | 性能 |
|:---:|:---:|------|------|
| True | None | 填 0 | MTE2 等 Vector 初始化 |
| False | None | 随机值 | MTE2/Vector 并行，访存密集提升 10%-30% |
| 任意 | 非 None | other 值 | care_padding 不生效 |

安全场景：element-wise（store 同 mask 过滤）、padding 被 where 覆盖、matmul K 维尾部。禁忌：sum/max 归约（改用 `other=0.0` / `-inf`）、中间结果依赖 padding、store 无 mask。

## 精度验证参数

| 场景 | rtol | atol |
|------|------|------|
| fp32 通用 | 1e-5 | 1e-5 |
| fp16/bf16 向量 | 1e-3 | 1e-3 |
| fp16/bf16 matmul | 2^-6 (~0.016) | 2^-6 |
| Softmax | 1e-5 | 1e-7（tl.exp 为近似计算） |
| int8 量化 | 0 | 0 |

matmul 分段验证（bf16/fp16）：

```python
mask = golden.abs() < 1.0
torch.testing.assert_close(result[mask], golden[mask], atol=2**-6, rtol=0)
torch.testing.assert_close(result[~mask], golden[~mask], atol=0, rtol=2**-6)
```

精度排查：1e-6 → 浮点顺序差异（正常）；1e-3 → 检查是否缺 `.to(tl.float32)`；1e-2 → matmul 正常，其他查累加器；差异大 → 查 int64/int32 标量退化、mask 逻辑，或 `TRITON_INTERPRET=1` CPU 基准对比。

## 910_95（dav-c310）特别注意

| 功能 | A2/A3 | 910_95 |
|------|:-----:|:------:|
| FP8 类型转换 | 不支持 | 支持 |
| FP8 dot_scaled | 不支持 | 支持（缩放张量 int8；fp4 不支持，bf16/fp16 支持） |
| FP8 直接 tl.dot | 不支持 | 不支持（先转 fp16/bf16） |

其他：UB 256KB（double buffer 时不超 128KB）；fixpipe L0C→UB 直通；multiBuffer 默认关闭需显式开启。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L288-L293](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L288-L293)
- [compiler.py#L835](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L835)
- [semantic.py#L61-L100](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L61-L100)
- [core.py#L1621-L1638](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1621-L1638)
- [semantic.py#L46-L100](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L46-L100)
