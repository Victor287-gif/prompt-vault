---
type: API
title: SIMT接口精度标准汇总
description: 本节按照数据类型和接口类型对SIMT浮点计算接口进行精度标准的分类汇总。对于非软仿接口，给出对应硬件指令的精度标准；对于软仿接口，由于软仿程度较低且使用到的硬件指令较少，给出其所用硬件指令的综合性精度标准。
tags:
- task.lookup
- topic.implementation
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/appendix/SIMT_interface_precision_standard_summary.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SIMT接口精度标准汇总

本节按照数据类型和接口类型对SIMT浮点计算接口进行精度标准的分类汇总。对于非软仿接口，给出对应硬件指令的精度标准；对于软仿接口，由于软仿程度较低且使用到的硬件指令较少，给出其所用硬件指令的综合性精度标准。

## bf16数据类型计算API精度标准

### 数学库函数

| 接口 | 硬件指令精度标准 | 软仿精度标准 |
| --- | --- | --- |
| [hexp](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hexp.md) | - | 1-ulp，Not Support denormalized numbers |
| [htanh](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/htanh.md) | - | 1-ulp，Not Support denormalized numbers |
| [hexp2](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hexp2.md) | - | 1-ulp，Not Support denormalized numbers |
| [hexp10](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hexp10.md) | - | 1-ulp，Not Support denormalized numbers |
| [hlog](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hlog.md) | - | 1-ulp，Not Support denormalized numbers |
| [hlog2](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hlog2.md) | - | 1-ulp，Not Support denormalized numbers |
| [hlog10](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hlog10.md) | - | 1-ulp，Not Support denormalized numbers |
| [hsqrt](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hsqrt.md) | - | 1-ulp，Not Support denormalized numbers |
| [hrsqrt](../simt_api/math_functions/bfloat16_type/bfloat16_math_functions/hrsqrt.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2tanh](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2tanh.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2exp](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2exp.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2exp2](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2exp2.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2exp10](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2exp10.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2log](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2log.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2log2](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2log2.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2log10](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2log10.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2sqrt](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2sqrt.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2rsqrt](../simt_api/math_functions/bfloat16_type/bfloat16x2_math_functions/h2rsqrt.md) | - | 1-ulp，Not Support denormalized numbers |

### 算术函数

| 接口 | 硬件指令精度标准 | 软仿精度标准 |
| --- | --- | --- |
| [\_\_habs](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_habs.md) | - | 0-ulp, Support denormalized numbers |
| [\_\_hadd](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hadd.md) | 0-ulp, Support denormalized numbers | - |
| [\_\_hmul](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hmul.md) | 0-ulp, Support denormalized numbers | - |
| [\_\_hdiv](../simt_api/math_functions/bfloat16_type/bfloat16_arithmetic_functions/_hdiv.md) | 1-ulp，Not Support denormalized numbers | - |
| [\_\_haddx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_haddx2.md) | 0-ulp, Support denormalized numbers | - |
| [\_\_hmulx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hmulx2.md) | 0-ulp, Support denormalized numbers | - |
| [\_\_hdivx2](../simt_api/math_functions/bfloat16_type/bfloat16x2_arithmetic_functions/_hdivx2.md) | 1-ulp，Not Support denormalized numbers | - |

## fp16数据类型计算API精度标准

### 数学库函数

| 接口 | 硬件指令精度标准 | 软仿精度标准 |
| --- | --- | --- |
| [hexp](../simt_api/math_functions/half_type/half_math_functions/hexp.md) | 1-ulp | - |
| [hexp2](../simt_api/math_functions/half_type/half_math_functions/hexp2.md) | - | 1-ulp，Not Support denormalized numbers |
| [hexp10](../simt_api/math_functions/half_type/half_math_functions/hexp10.md) | - | 1-ulp，Not Support denormalized numbers |
| [hlog](../simt_api/math_functions/half_type/half_math_functions/hlog.md) | 1-ulp，Not Support denormalized numbers | - |
| [hlog2](../simt_api/math_functions/half_type/half_math_functions/hlog2.md) | - | 1-ulp，Not Support denormalized numbers |
| [hlog10](../simt_api/math_functions/half_type/half_math_functions/hlog10.md) | - | 1-ulp，Not Support denormalized numbers |
| [hsqrt](../simt_api/math_functions/half_type/half_math_functions/hsqrt.md) | 1-ulp，Not Support denormalized numbers | - |
| [hrsqrt](../simt_api/math_functions/half_type/half_math_functions/hrsqrt.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2tanh](../simt_api/math_functions/half_type/half2_math_functions/h2tanh.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2exp](../simt_api/math_functions/half_type/half2_math_functions/h2exp.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2exp2](../simt_api/math_functions/half_type/half2_math_functions/h2exp2.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2exp10](../simt_api/math_functions/half_type/half2_math_functions/h2exp10.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2log](../simt_api/math_functions/half_type/half2_math_functions/h2log.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2log2](../simt_api/math_functions/half_type/half2_math_functions/h2log2.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2log10](../simt_api/math_functions/half_type/half2_math_functions/h2log10.md) | - | 1-ulp，Not Support denormalized numbers |
| [h2sqrt](../simt_api/math_functions/half_type/half2_math_functions/h2sqrt.md) | 1-ulp，Not Support denormalized numbers | - |
| [h2rsqrt](../simt_api/math_functions/half_type/half2_math_functions/h2rsqrt.md) | - | 1-ulp，Not Support denormalized numbers |

### 算术函数

| 接口 | 硬件指令精度标准 | 软仿精度标准 |
| --- | --- | --- |
| [\_\_hadd](../simt_api/math_functions/half_type/half_arithmetic_functions/_hadd.md) | 0-ulp, Support denormalized numbers | - |
| [\_\_hmul](../simt_api/math_functions/half_type/half_arithmetic_functions/_hmul.md) | 0-ulp, Support denormalized numbers | - |
| [\_\_hdiv](../simt_api/math_functions/half_type/half_arithmetic_functions/_hdiv.md) | 1-ulp，Not Support denormalized numbers | - |
| [\_\_haddx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_haddx2.md) | 0-ulp, Support denormalized numbers | - |
| [\_\_hmulx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hmulx2.md) | 0-ulp, Support denormalized numbers | - |
| [\_\_hdivx2](../simt_api/math_functions/half_type/half2_arithmetic_functions/_hdivx2.md) | 1-ulp，Not Support denormalized numbers | - |

## fp32数据类型计算API精度标准

| 接口 | 硬件指令精度标准 | 软仿精度标准 |
| --- | --- | --- |
| [expf](../simt_api/math_functions/float_math_functions/expf.md) | 1-ulp，Not Support denormalized numbers | - |
| [tanhf](../simt_api/math_functions/float_math_functions/tanhf.md) | - | 1-ulp，Not Support denormalized numbers |
| [tanpif](../simt_api/math_functions/float_math_functions/tanpif.md) | - | 1-ulp，Not Support denormalized numbers |
| [atanf](../simt_api/math_functions/float_math_functions/atanf.md) | - | 1-ulp，Not Support denormalized numbers |
| [atan2f](../simt_api/math_functions/float_math_functions/atan2f.md) | - | 1-ulp，Not Support denormalized numbers |
| [atanhf](../simt_api/math_functions/float_math_functions/atanhf.md) | - | 1-ulp，Not Support denormalized numbers |
| [expm1f](../simt_api/math_functions/float_math_functions/expm1f.md) | - | 1-ulp，Not Support denormalized numbers |
| [logf](../simt_api/math_functions/float_math_functions/logf.md) | - | 1-ulp，Not Support denormalized numbers |
| [log2f](../simt_api/math_functions/float_math_functions/log2f.md) | - | 1-ulp，Not Support denormalized numbers |
| [log10f](../simt_api/math_functions/float_math_functions/log10f.md) | - | 1-ulp，Not Support denormalized numbers |
| [log1pf](../simt_api/math_functions/float_math_functions/log1pf.md) | - | 1-ulp，Not Support denormalized numbers |
| [cosf](../simt_api/math_functions/float_math_functions/cosf.md) | - | 1-ulp，Not Support denormalized numbers |
| [acoshf](../simt_api/math_functions/float_math_functions/acoshf.md) | - | 1-ulp，Not Support denormalized numbers |
| [sqrtf](../simt_api/math_functions/float_math_functions/sqrtf.md) | 1-ulp，Not Support denormalized numbers | - |
| [rsqrtf](../simt_api/math_functions/float_math_functions/rsqrtf.md) | - | 1-ulp，Not Support denormalized numbers |
| [cbrtf](../simt_api/math_functions/float_math_functions/cbrtf.md) | - | 1-ulp，Not Support denormalized numbers |
| [rcbrtf](../simt_api/math_functions/float_math_functions/rcbrtf.md) | - | 1-ulp，Not Support denormalized numbers |
| [remquof](../simt_api/math_functions/float_math_functions/remquof.md) | - | 1-ulp，Not Support denormalized numbers |
| [fdividef](../simt_api/math_functions/float_math_functions/fdividef.md) | 1-ulp，Not Support denormalized numbers | - |
