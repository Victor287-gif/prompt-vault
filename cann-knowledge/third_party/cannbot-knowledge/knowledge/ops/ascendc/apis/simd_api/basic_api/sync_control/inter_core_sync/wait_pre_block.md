---
type: API
title: WaitPreBlock
description: 通过读取全局内存判断前序核是否完成，阻塞当前核直至前序核通过NotifyNextBlock通知，与InitDetermineComputeWorkspace配合实现确定性计算。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/WaitPreBlock.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_determine_compute_sync_intf.h
---

# WaitPreBlock<a name="ZH-CN_TOPIC_0000001771590261"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_determine_compute_sync_intf.h"`。

WaitPreBlock和NotifyNextBlock是核间同步控制接口。NotifyNextBlock通过写全局内存来通知其他核当前核已执行完成，其他核可以继续往下执行；WaitPreBlock通过读取全局内存，判断当前核是否可以继续往下执行。

WaitPreBlock和NotifyNextBlock必须与[InitDetermineComputeWorkspace](init_determine_compute_workspace.md#)接口配合使用，这三个接口组合使用能够确保多个AIV核严格按照blockIdx的升序执行，适用于要求确定性计算的场景，确定性计算的具体含义请参考[原子操作-确定性计算](../../atomic_operations/key_features.md)。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void WaitPreBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| gmWorkspace | 输入 | 临时空间，通过读取gmWorkspace，判断当前核是否可以继续往下执行，类型为GlobalTensor。 |
| ubWorkspace | 输入 | 临时空间，用于操作gmWorkspace，类型为LocalTensor。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 要实现确定性计算，需要保证每个核调用NotifyNextBlock/WaitPreBlock接口的次数相同。如果调用了NotifyNextBlock没有与之配对的WaitPreBlock，会导致非确定性计算、输出结果可能不符合预期；如果调用了WaitPreBlock没有与之配对的NotifyNextBlock，会导致程序卡死。
- 使用接口前，请确保已经调用[InitDetermineComputeWorkspace](init_determine_compute_workspace.md)接口，初始化共享内存。
- gmWorkspace申请的空间最少要求为：GetBlockNum()*32Bytes，ubWorkspace申请的空间最少要求为：GetBlockNum()*32+32Bytes。
- 使用该接口进行多核同步时，仅对AIV核生效，WaitPreBlock和NotifyNextBlock之间仅支持插入矢量计算相关指令，对矩阵计算相关指令不生效。
- 使用该接口进行多核控制时，算子调用时指定的逻辑AI Core核数numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致核函数（Kernel）“卡死”现象。

## 调用示例<a name="section177231425115410"></a>

完整样例请参考[调用示例](init_determine_compute_workspace.md#section177231425115410)。
