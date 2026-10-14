---
type: API
title: asc_set_l0c2gm_config
description: 数据搬运过程中进行随路量化时，通过调用该接口设置量化流程中的矢量量化参数。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/asc_set_l0c2gm_config.md
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

# asc_set_l0c2gm_config
## 功能说明

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

数据搬运过程中进行随路量化时，通过调用该接口设置量化流程中的矢量量化参数。

## 函数原型

```cpp
__aicore__ inline void asc_set_l0c2gm_config(uint64_t relu_pre, uint64_t quant_pre, bool enable_unit_flag)
```

## 参数说明

**表1** 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| relu_pre | 输入     | ReLU操作前矢量的起始地址。|
| quant_pre | 输入     | 量化操作前矢量的起始地址。|
| enable_unit_flag | 输入     | 是否启用unit_flag。unit_flag是一种矩阵计算指令和矩阵搬运指令细粒度的并行，开启该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于L0C Buffer累加的场景。|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

量化参数不能为inf/nan和非规格化数。

支持以下三种传参形式：
- 同时设置relu_pre和quant_pre。
- 仅传入relu_pre，quant_pre传入0。
- 仅传入quant_pre，relu_pre传入0。

## 调用示例

```cpp
constexpr uint64_t relu_pre = 0;
constexpr uint64_t quant_pre = 0x1000;// 假设量化操作有效地址为0x1000
asc_set_l0c2gm_config(relu_pre, quant_pre, true);
```
