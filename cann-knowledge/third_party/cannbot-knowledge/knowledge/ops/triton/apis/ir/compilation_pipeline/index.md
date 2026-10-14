# 编译流水线

## AscendNPU-IR 主流水线

- [overview](overview.md)：编译数据流全景：六个前端转换、两层方言变换与 SIMD/SIMT/MIX 三条后端路径
- [frontend_to_hfusion](frontend_to_hfusion.md)：arith/math/linalg/gpu/tensor/torch → HFusion 的转换 Pass 与执行顺序
- [hfusion_transforms](hfusion_transforms.md)：HFusion 40+ 变换 Pass：OpFusion/AutoSchedule/AutoVectorize 核心链路
- [hfusion_to_hivm](hfusion_to_hivm.md)：ConvertHFusionToHIVM 与 mm-map-mode 矩阵乘映射策略
- [hivm_transforms](hivm_transforms.md)：HIVM 70+ 变换 Pass：推断链、多缓冲、PlanMemory、同步链与 Mix 分裂
- [hivm_to_backend](hivm_to_backend.md)：HIVM→LLVM/TritonGPU 的 SIMD/SIMT/MIX 三条 lowering 路径
- [triton_ir_compilation](triton_ir_compilation.md)：Triton 原生编译链（TTIR/TTGIR）各阶段 Pass 与昇腾适配、bishengir-compile 的关系
- [pipeline_options](pipeline_options.md)：HIVMPipelineOptions 等全部字段、默认值与典型配置组合
- [pass_dependencies](pass_dependencies.md)：分阶段执行顺序、必须顺序执行的 Pass 对与 last_pass.mlir 阶段定位

## Triton-Ascend 前端编译链（Python → TTIR → Linalg → 二进制）

- [triton_compile_flow](triton_compile_flow.md)：新手向编译主线：AST→TTIR→Ascend Passes→Linalg→BiSheng，缓存与 dump 入门
- [triton_pipeline_overview](triton_pipeline_overview.md)：分析者向阶段划分：三路径对比、add_stages 注册、缓存键与 dump 文件清单
- [ttir_generation](ttir_generation.md)：AST→TTIR：CodeGenerator 映射规则、constexpr 求值与 Ascend 扩展注入
- [ttir_optimization](ttir_optimization.md)：TTIR 优化 Pass 与循环优化
- [ascend_passes](ascend_passes.md)：ttir_to_linalg 的 12 步 Ascend Pass 序列：AutoBlockify、DAG 亲和性、TritonToStructured 等
- [triton_to_linalg](triton_to_linalg.md)：TritonToLinalg 转换器全集、mix_mode/parallel_mode 与元数据解析
- [linalg_to_binary](linalg_to_binary.md)：Linalg→设备二进制：A2/A3 与 910_95 两条 bishengir-compile 路径、SIMT 直编与产物
- [triton_compile_options](triton_compile_options.md)：NPUOptions 全字段、compile_mode 语义、SIMT 位掩码与环境变量
