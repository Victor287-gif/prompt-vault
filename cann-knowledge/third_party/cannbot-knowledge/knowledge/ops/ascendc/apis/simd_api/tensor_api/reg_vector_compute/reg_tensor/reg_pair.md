---
type: API
title: reg_pair
description: 头文件路径为："tensor_api/arch/vector/experimental/reg_tensor.h"，命名空间为asc::te::experimental。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/reg_vector_compute/reg_tensor/reg_pair.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/tensor_api/arch/vector/experimental/reg_tensor.h
---

# reg_pair

## 功能说明

头文件路径为：`"tensor_api/arch/vector/experimental/reg_tensor.h"`，命名空间为`asc::te::experimental`。

保存两个相同元素类型的`reg_tensor`，用于`deinterleave`等双结果接口。

## 定义原型

```cpp
template <typename DataType>
struct reg_pair {
    reg_tensor<DataType> first;
    reg_tensor<DataType> second;
};
```

## 参数说明

| 参数名 | 描述 |
| --- | --- |
| `DataType` | `reg_tensor<DataType> first`和`reg_tensor<DataType> second`的元素类型。 |

## 成员说明

| 成员名 | 描述 |
| --- | --- |
| `first` | 第一个矢量寄存器。 |
| `second` | 第二个矢量寄存器。 |

## 调用示例

```cpp
auto result = asc::te::experimental::deinterleave<float>(mask0, mask1);
asc::te::experimental::reg_tensor<bool> first = result.first;
asc::te::experimental::reg_tensor<bool> second = result.second;
```
