---
type: API
title: Ascend 特有 Pass（ttir_to_linalg 的 12 步序列）
description: ttir_to_linalg() 的 Ascend Pass 执行序列与依赖：AutoBlockify、DAG 亲和性优化（dag-sync/scope/ssbuffer）、TritonToStructured（两次）、DiscreteMaskAccessConversion、TritonToAnnotation/Unstructured/HIVM/HFusion/LLVM、BubbleUpOperation
  的功能、选项与触发条件。
tags: [task.lookup, topic.compilation, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# Ascend 特有 Pass

`ttir_to_linalg()` 在 TTIR 优化之后执行以下序列（TTIR → Linalg IR 的适配层），末尾第 12 步 TritonToLinalg 的转换器细节见 [triton_to_linalg.md](triton_to_linalg.md)。

```
1. AutoBlockify                      add_auto_blockify(pm, auto_blockify_size)
2. DAG 亲和性（可选，add_auto_scheduling 控制）
   dag-sync → dag-scope → CSE/Canonicalize → dag-ssbuf → CSE/Canonicalize
3. TritonToStructured                add_triton_to_structure(pm, enable_mask_fallback_conversion, optimize_dynamic_offset)
4. DiscreteMaskAccessConversion      add_discrete_mask_access_conversion(pm, compile_on_910_95, force_simt_template)
5. TritonToAnnotation                add_triton_to_annotation(pm)
6. TritonToUnstructured              add_triton_to_unstructure(pm, compile_on_910_95, force_simt_template)
7. TritonToHIVM                      add_triton_to_hivm(pm)
8. TritonToHFusion                   add_triton_to_hfusion(pm)
9. TritonToLLVM                      add_triton_to_llvm(pm)
10. BubbleUpOperation                add_bubble_up_operation(pm)
11. TritonToStructured（二次）
12. TritonToLinalg                   add_triton_to_linalg(pm, False, named_ops, enable_nd2nz_on_vector, enable_select_analysis, compile_on_910_95)
```

**顺序约束**：DiscreteMaskAccessConversion 必须先于 TritonToUnstructured；TritonToStructured 执行两次（第二次处理前面 Pass 新引入的指针/mask 表达式）。

## 1. AutoBlockify（自动分块）

逻辑核合并到物理核，减少调度开销。流程：检查 `GetProgramIdOp` → 重算分块 ID → 替换 program_id → 生成上下界 mask → CSE/Canonicalize。

| 选项 | 默认 | 说明 |
|------|------|------|
| `auto_blockify_size` | 1 | 分块大小；`TRITON_ALL_BLOCKS_PARALLEL` 未设置时被强制为 1（不启用） |

启用条件：`TRITON_ALL_BLOCKS_PARALLEL=1` 且 `auto_blockify_size > 1`。限制：kernel 逻辑必须对执行顺序不敏感（否则死锁）；启用后允许 grid > 65535。

## 2. DAG 亲和性优化（TritonAffinityOpt，可选）

由 `add_auto_scheduling=True` 启用，面向 Cube+Vector 混合 kernel 的 CV 流水线（纯向量 kernel 不需要）：

| 子 Pass | TableGen 名 | 功能 |
|---------|-------------|------|
| DAGSync | `dag-sync` | 构建数据流 DAG，识别 Cube/Vector 亲和性，跨核心操作间插入 `hivm.sync_block_set/wait` 与 GM↔UB 搬运（基于 SharedMemoryAliasAnalysis） |
| DAGScope | `dag-scope` | 把操作按亲和性划分进 `scope.scope @aiv` / `@aic`，补充 scope 间同步点 |
| DAGSSBuffer | `dag-ssbuf` | Vector 操作转共享存储缓冲区操作，优化 UB 使用 |

## 3/11. TritonToStructured（指针/mask 线性化）

处理指针与 mask 表达式中的整除取余：分解为 `PtrState`/`MaskState`，升维消除 div/rem 后重建 load/store。

| 转换器 | 功能 / 局限 |
|--------|-------------|
| RewriteAddPtrOp / CreateAddpr | 指针表达式分解与重建；原始迭代轴必须能被分裂轴整除 |
| RewriteLoadOp / BuildMask / CreateLoad | mask 分解为 MaskState 并重建 load |
| RewriteStoreOp / CreateStore | store 同理 |
| RewriteAtomicRWMOp / RewriteAtomicCASOp | 原子操作指针线性化 |
| RewriteWhile / RewriteFor | 循环体内指针叠加；while 不支持循环内含 if |

选项：`enable_mask_fallback_conversion`（默认 False）、`optimize_dynamic_offset`（默认 False）。

## 4. DiscreteMaskAccessConversion（离散 mask 转换）

识别非规律/稀疏访存，为离散轴展开做准备：

| 转换器 | 变换 |
|--------|------|
| DiscreteMaskStoreConversion | 非连续 mask store → load→select→store |
| DiscreteMaskLoadConversion | 非连续 mask load → 全量 load→select(other) |
| DiscreteMaskAtomicAddConversion | 非连续 mask atomic_add → select→atomic_add |

选项：`compile_on_910_95`（自动检测）、`force_simt_template`（默认 False；`compile_mode="unstructured_in_simt"` 时为 True）。

## 5. TritonToAnnotation

`triton::AnnotationOp`（`al.compile_hint` 编译提示）→ `annotation::MarkOp`，指导后续硬件特定优化。

## 6. TritonToUnstructured（离散轴转标量循环）

`UnstructuredMemAccessConverter<{Load,Store,AtomicRMW,AtomicCAS}Op>`：把含离散轴的张量操作展开为 `scf.for` + `scf.if` 的多重循环标量访存，是 SIMT 混合模式的关键 Pass。

## 7-9. 方言转换

| Pass | 转换 |
|------|------|
| TritonToHIVM | `al.sync_block_all/set/wait` → HIVM 全局块同步 / 同步设置 / 同步等待 |
| TritonToHFusion | `triton::HistogramOp` → `hfusion::HistogramOp`（NPU 专用硬件加速） |
| TritonToLLVM | `triton::ElementwiseInlineAsmOp`（`tl.inline_assembly`）→ `LLVM::InlineAsmOp`，最终映射为 CCE Intrinsics |

## 10. BubbleUpOperation（操作上浮）

`BubbleUpExtract<tensor::ExtractOp/ExtractSliceOp>`：extract 顺序上移优化数据局部性，索引不依赖循环变量时可把 extract 移出循环，某些场景消除多余循环。

## 平台差异

- `force_simt_template` 影响第 4、6 步行为（混合路径）。
- 部分 Pass 在 910_95 平台行为有差异（经 `compile_on_910_95` 传入）。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [compiler.py#L95](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L95)
- [AutoBlockify.h](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/include/AutoBlockify/AutoBlockify.h)
- [AutoBlockify.cpp](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/lib/AutoBlockify/AutoBlockify.cpp)
- [Passes.h](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/include/TritonAffinityOpt/Passes.h)
- [DAG.cpp](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/lib/TritonAffinityOpt/DAG.cpp)
- [DAGSync.cpp](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/lib/TritonAffinityOpt/DAGSync.cpp)
- [DAGScope.cpp](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/lib/TritonAffinityOpt/DAGScope.cpp)
- [DAGSSBuffer.cpp](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/lib/TritonAffinityOpt/DAGSSBuffer.cpp)
