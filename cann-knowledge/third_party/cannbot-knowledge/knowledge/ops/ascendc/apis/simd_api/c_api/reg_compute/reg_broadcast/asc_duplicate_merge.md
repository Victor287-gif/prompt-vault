---
type: API
title: asc_duplicate_merge
description: merge模式下，将src的最低位元素广播到dst中被mask筛选的位置，dst中未被mask筛选的元素保留原值。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/reg_compute/reg_broadcast/asc_duplicate_merge.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/reg_compute/compute/reg_broadcast.h
---

# asc_duplicate_merge

## 功能说明

头文件路径为：`"c_api/reg_compute/compute/reg_broadcast.h"`。

merge模式下，将src的最低位元素广播到dst中被mask筛选的位置，dst中未被mask筛选的元素保留原值。

## 函数原型

```cpp
__simd_callee__ inline void asc_duplicate_merge(vector_int8_t& dst, vector_int8_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_uint8_t& dst, vector_uint8_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_fp8_e8m0_t& dst, vector_fp8_e8m0_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_fp8_e5m2_t& dst, vector_fp8_e5m2_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_fp8_e4m3fn_t& dst, vector_fp8_e4m3fn_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_int16_t& dst, vector_int16_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_half& dst, vector_half src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_int32_t& dst, vector_int32_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask)
__simd_callee__ inline void asc_duplicate_merge(vector_float& dst, vector_float src, vector_bool mask)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输入/输出 | 目的操作数（矢量数据寄存器）。merge模式下，未被mask筛选的元素保留原值。 |
| src | 输入 | 源操作数（矢量数据寄存器）。 |
| mask | 输入 | 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。 |

矢量数据寄存器和掩码寄存器的详细说明请参见[reg数据类型定义](../../defs/type/data_type_definition.md)。

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
__simd_vf__ inline void duplicate_merge_vf(__ubuf__ half* dst_addr, __ubuf__ half* src_addr, uint32_t count, uint16_t one_repeat_size, uint16_t repeat_time)
{
    vector_half dst;
    vector_half src;
    vector_bool mask;
    for (uint16_t i = 0; i < repeat_time; ++i) {
        mask = asc_update_mask_b16(count);
        asc_loadalign(dst, dst_addr + i * one_repeat_size);
        asc_loadalign(src, src_addr + i * one_repeat_size);
        asc_duplicate_merge(dst, src, mask);
        asc_storealign(dst_addr + i * one_repeat_size, dst, mask);
    }
}
```
