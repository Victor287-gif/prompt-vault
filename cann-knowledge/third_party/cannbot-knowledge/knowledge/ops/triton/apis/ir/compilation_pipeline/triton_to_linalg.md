---
type: API
title: TritonToLinalg 转换（TTIR → Linalg IR 与元数据解析）
description: TritonToLinalg Pass 的转换器全集（tt 操作 → linalg/memref/tensor/func）、named_ops
  等编译选项、mix_mode/parallel_mode 的确定与影响、_parse_linalg_metadata 提取的元数据字段与 kernel 命名规则。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# TritonToLinalg 转换

SIMD 路径核心转换：`ttir_to_linalg()` 序列的第 12 步（前 11 步见 [ascend_passes.md](ascend_passes.md)），用 MLIR DialectConversion 框架把 `tt` 操作映射为 linalg/memref/tensor/func。Linalg IR 是 Triton 前端与 BiSheng Compiler 的桥梁。

注意：`ttir_to_linalg()` 返回**字符串**而非 Module（后续 `linalg_to_bin_*` 要把 IR 写入临时文件传给外部进程 bishengir-compile）；`opt.debug` 时落盘 `kernel.ttadapter.mlir`。

## 转换器全集

| 转换器 | 转换目标 |
|--------|---------|
| StoreConverter | `tt.store` → `memref.copy` |
| LoadConverter | `tt.load` → `memref.copy` + `bufferization.to_tensor` |
| AddPtrConverter / AdvanceConverter | `tt.addptr` / `tt.advance` → `memref.reinterpret_cast` |
| GetProgramIDConverter / GetNumProgramsConverter | `tt.get_program_id` / `tt.get_num_programs` → 函数参数 |
| MatmulConverter / DotScaledConverter | `tt.dot` / `tt.dot_scaled` → `linalg.matmul` |
| ReduceConverter / ArgMinConverter / ArgMaxConverter | `tt.reduce` / `tt.argmin` / `tt.argmax` → `linalg.reduce` |
| SplatConverter / DenseConstantConverter | `tt.splat` / `arith.constant` → `linalg.fill` |
| ClampFConverter | `tt.clampf` → `tensor.empty` + `linalg.fill` |
| BroadcastConverter | `tt.broadcast` → `linalg.broadcast` |
| TransposeConverter | `tt.trans` → `linalg.transpose` |
| ReshapeConverter / ExpandDimsConverter | `tt.reshape` → `tensor.reshape`；`tt.expand_dims` → `tensor.expand_shape` |
| SplitConverter / JoinConverter / CatConverter | → `tensor.extract_slice` / `tensor.insert_slice` |
| MakeRangeConverter | `tt.make_range` → `linalg.generic` |
| AtomicRMWConverter / AtomicCASConverter | → `linalg.generic` |
| ScanConverter / GatherConverter / SortOpConverter / DeviceAssertConverter / DevicePrintConverter | → `func.call` / 生成 `func.func` 辅助函数 |
| GatherLoadConverter | `tt.gather_load` → `scf.for` |
| ExternElementwiseClOpConverter | `tt.extern_elementwise` → `linalg.map` |
| TritonMulhiuiConverter / TritonPreciseSqrtConverter / PreciseDivConverter | → `arith.mulsirextended` / `math.sqrt` / `arith.divf` |
| BitcastConverter | `tt.bitcast` → `arith.bitcast` |
| PtrToIntConverter / MakeTensorPtrConverter | `tt.make_tensor_ptr` → `arith.index_cast` 等 |
| LoopConverter\<scf.for/while\> / YieldConverter | scf 结构保留 |

## 编译选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `globalKernel` | False | 是否全局 kernel |
| `namedOps` | True（NPU 路径） | True 生成 `linalg.matmul` 等命名操作（BiSheng 可直接识别优化）；False 用 `linalg.generic`，表达更通用但优化空间小 |
| `enableNd2nzOnVector` | False | Vector 核心启用 ND→NZ 布局转换 |
| `enableSelectAnalysis` | True | 启用 select 分析 |
| `compileOn91095` | 自动检测 | 910_95 平台 |

## mix_mode / parallel_mode

由 TritonToLinalg 在转换中按操作类型确定，编码进函数属性并拼进 kernel 名：

| mix_mode | 含义 | BiSheng Compiler 行为 |
|----------|------|----------------------|
| `aiv` | Vector 核心（纯向量操作） | HFusion Vector 路径 |
| `aic` | Cube 核心（含矩阵乘） | HFusion Cube 路径，禁用 HFusion 向量化 |
| `mix_simd_simt` | 混合（含离散访存） | SIMD 部分走 HFusion，SIMT 部分走 TritonGPU |

`parallel_mode` 取值 `simd` / `simt` / `mix_simd_simt`。SIMT 直编路径（`ttir_to_npubin`）只支持 Vector kernel，`mix_mode` 固定 `aiv`。

## 元数据解析（_parse_linalg_metadata）

| 字段 | 正则来源 | 说明 |
|------|----------|------|
| `mix_mode` | `mix_mode = "..."` 函数属性 | 混合执行模式 |
| `parallel_mode` | `parallel_mode = "..."` | 并行模式 |
| `kernel_name` | `func.func @(\w+)` | kernel 函数名 |
| `tensor_kinds` | `tt.tensor_kind = N` | 0=输入，1=输出，供 BiSheng 内存规划 |
| `bitcodes` | `bitcode = "..."` | 需链接的 bitcode 路径 |
| `auto_tile_and_bind_subblock` | 存在 `hivm.disable_auto_tile_and_bind_subblock` 则禁用 | 自动分块绑定开关 |
| `required_ub_bits` | 置 0，后续由 bishengir-compile 输出回填 | UB 占用 |

kernel 命名：`name = kernel_name + "_" + mix_mode`（如 `add_kernel_aiv`、`matmul_kernel_aic`）；CANN 限制名长 ≤ 49 字符，由 `pack_metadata()` 截断。SIMT 路径对应 `_parse_ttir_metadata()`（从 TTIR 解析，mix_mode 固定 `aiv`）。

## 转换示例

TTIR 输入的 `tt.dot`：

```mlir
%dot = tt.dot %a, %b, %c : tensor<64x64xf32>
```

Linalg 输出（named_ops=True）：

```mlir
linalg.matmul ins(%A, %B : memref<64x64xf32>, memref<64x64xf32>)
              outs(%C : memref<64x64xf32>)
```

函数级形态：`!tt.ptr<f32>` 参数 → `memref<?xf32>`（带 `{tt.divisibility = 16, tt.tensor_kind = 0|1}` 属性），函数带 `attributes {mix_mode = "aiv", parallel_mode = "simd"}`；load → `memref.subview` + `bufferization.to_tensor`，逐元素加法 → `linalg.generic {indexing_maps, iterator_types = ["parallel"]}`。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [compiler.py#L95](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L95)
- [compiler.py#L288](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L288)
- [compiler.py#L342](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L342)
