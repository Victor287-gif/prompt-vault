---
type: API
title: asc_set_l13d_padding
description: 设置Pad属性描述，用于在调用asc_copy_l12l0a接口时配置填充数值。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/asc_set_l13d_padding.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_set_l13d_padding

## 功能说明

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

设置Pad属性描述，用于在调用[asc_copy_l12l0a](asc_copy_l12l0a/index.md)接口时配置填充数值。

## 函数原型

```cpp
__aicore__ inline void asc_set_l13d_padding(uint64_t config)
__aicore__ inline void asc_set_l13d_padding(half config)
__aicore__ inline void asc_set_l13d_padding(int16_t config)
__aicore__ inline void asc_set_l13d_padding(uint16_t config)
```

## 参数说明

**表1** 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| config | 输入     | Pad填充值的数值。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
constexpr uint64_t config = 0;
asc_set_l13d_padding(config);
```
