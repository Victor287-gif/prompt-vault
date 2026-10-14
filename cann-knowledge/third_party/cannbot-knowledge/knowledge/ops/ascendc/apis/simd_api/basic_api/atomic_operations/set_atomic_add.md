---
type: API
title: SetAtomicAdd
description: 对后续搬运到GM的数据开启原子累加，将拷贝内容与GM已有值求和后写入。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.atomic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/atomic_operations/SetAtomicAdd.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_operator_set_atomic_intf.h
---

# SetAtomicAdd<a name="ZH-CN_TOPIC_0000001487536650"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_set_atomic_intf.h"`。

对后续目的地址为GM的数据搬运开启原子累加。原子累加过程：将待拷贝的内容和GM已有内容进行求和，然后将求和结果写入GM。SetAtomicAdd接口可通过模板参数设定不同的累加数据类型。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void SetAtomicAdd()
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 设定不同的原子累加操作数据类型。 |

## 数据类型


- Ascend 950PR/Ascend 950DT，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。



- Atlas 200I/500 A2 推理产品，支持的数据类型为int16_t、half、int32_t、float。



- Atlas 推理系列产品AI Core，支持的数据类型为int16_t、half、float。



- Atlas 训练系列产品，支持的数据类型为float。


## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>


- Ascend 950PR/Ascend 950DT，支持的数据通路为UB/L0C Buffer-\>GM。



- Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据通路为UB/L0C Buffer/L1 Buffer-\>GM。



- Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据通路为UB/L0C Buffer/L1 Buffer-\>GM。



- Atlas 200I/500 A2 推理产品，支持的数据通路为UB/L0C Buffer/L1 Buffer-\>GM。



- Atlas 推理系列产品AI Core，支持的数据通路为UB-\>GM。



- Atlas 训练系列产品，支持的数据通路为UB-\>GM。


- 累加操作完成后，建议通过[DisableDmaAtomic](disable_dma_atomic.md)关闭原子累加，以免影响后续相关指令功能。
- 该接口执行前不会自动清零GM数据。开发者需根据算子逻辑判断是否清零，如需清零，请在执行前手动完成。
- SetAtomicAdd内部已集成与[SetAtomicType](set_atomic_type.md)相同的功能。建议调用上述接口时，通过设置模板参数显式指定原子操作的数据类型，无需额外调用SetAtomicType接口。

## 调用示例<a name="section177231425115410"></a>

```cpp
AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
AscendC::LocalTensor<T> srcLocal = ubAllocator.Alloc<T, SIZE>();

AscendC::DisableDmaAtomic();
AscendC::DataCopy(srcLocal, srcGlobal, SIZE);
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
AscendC::SyncAll();

// 开启原子累加，将Unified Buffer（UB）数据原子累加到GM
AscendC::SetAtomicAdd<T>();
AscendC::DataCopy(dstGlobal, srcLocal, SIZE);

// 关闭原子累加
AscendC::DisableDmaAtomic();
```

完整样例请参考[DataMovementWithAtomicOperations样例](https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/06_atomic/data_movement_with_atomic_operations/data_movement_with_atomic_operations.asc)。


- [UnfoldGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/unfold_grad.md)
- [ThreeInterpolateBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/three_interpolate_backward.md)
- [CircularPad2dBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d_backward.md)
- [IndexFill A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/index/index_fill.md)
- [BlockSparseAttentionGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/block_sparse_attention_grad.md)
- [CircularPad3dBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_3d_backward.md)
- [InplaceIndexFill A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/index/inplace_index_fill.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [AddLora A2 ACLNN 算子知识](../../../../operators/aclnn/math/math/add_lora.md)
- [ChamferDistanceBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/loss/chamfer_distance_backward.md)
- [FlashAttentionScoreGrad 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attention_score_grad_950.md)
