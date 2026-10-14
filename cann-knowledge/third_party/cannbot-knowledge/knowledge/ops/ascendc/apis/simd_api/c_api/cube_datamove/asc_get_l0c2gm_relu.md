---
type: API
title: asc_get_l0c2gm_relu
description: 数据搬运过程中进行随路量化时，通过调用该接口获取ReLU操作前矢量的起始地址。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/asc_get_l0c2gm_relu.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_get_l0c2gm_relu

## 功能说明

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

数据搬运过程中进行随路量化时，通过调用该接口获取ReLU操作前矢量的起始地址。

## 函数原型

```cpp
__aicore__ inline uint64_t asc_get_l0c2gm_relu()
```

## 参数说明

无

## 返回值说明

ReLU操作前矢量的起始地址。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint64_t relu_addr = asc_get_l0c2gm_relu();
```
