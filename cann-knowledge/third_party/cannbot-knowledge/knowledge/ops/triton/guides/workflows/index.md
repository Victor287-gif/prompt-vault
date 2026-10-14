# 工作流

端到端开发工作流指南。

## 入门与流程

- [triton_ascend_quickstart](triton_ascend_quickstart.md)：环境搭建命令序列（pip/源码/Docker）、版本兼容矩阵与第一个 kernel
- [optimization_flow](optimization_flow.md)：端到端性能优化流程：Profile→Bottleneck→Route→Implement→Re-profile→Validate
- [profiling](profiling.md)：msprof 采集/解析、关键指标解读与瓶颈判定
- [profiling_guide](profiling_guide.md)：msProf 上板与仿真流水图采集命令、op_summary 指标判读、理论耗时计算与六大性能反模式
- [shape_router](shape_router.md)：形状驱动算子决策树：从 PyTorch 算子代码自动路由到 ops/techniques/experiences

## GPU→NPU 迁移

- [gpu_migration_overview](gpu_migration_overview.md)：GPU SIMT vs NPU SIMD 架构差异、迁移检查清单与三阶段迁移工作流
- [migration_api_differences](migration_api_differences.md)：API 不支持项与替代方案、行为差异、Ascend 扩展（al.*）清单
- [migration_code_patterns](migration_code_patterns.md)：9 类代码改写模式 before/after（设备替换、Grid 对齐、care_padding 等）
- [migration_common_issues](migration_common_issues.md)：迁移常见问题 Q&A：UB 溢出、coreDim 超限、精度差异、标量退化
- [block_pointer_migration](block_pointer_migration.md)：make_block_ptr/advance 迁移差异：stride/order 规则、编译限制与检查清单

## 调试

- [debugging](debugging.md)：Triton Ascend kernel 调试方法与常见错误排查
- [debug_overview](debug_overview.md)：调试工具链总览：解释器模式/IR 转储/device_print/msProf 与场景速查
- [interpreter_mode](interpreter_mode.md)：TRITON_INTERPRET=1 CPU 解释执行、device_print/device_assert 用法
- [compile_error_debug](compile_error_debug.md)：编译错误分阶段定位（dump 文件判定）、IR 识读与十大常见编译错误
- [runtime_error_debug](runtime_error_debug.md)：运行时错误排查：UB 溢出、coreDim 超限、精度异常、死锁诊断命令
- [debugging_common_errors](debugging_common_errors.md)：17 类高频错误的报错-原因-修复三元组（含 910_95 平台差异）
- [error_codes](error_codes.md)：编译/运行时错误信息→原因→解决方案对照表与 NPU 执行错误码
- [debug_env_variables](debug_env_variables.md)：全部调试/编译控制/运行调度环境变量参考与场景组合速查
- [triton_ascend_faq](triton_ascend_faq.md)：安装/编译/运行时/精度/性能五类 FAQ 与版本兼容性速查
