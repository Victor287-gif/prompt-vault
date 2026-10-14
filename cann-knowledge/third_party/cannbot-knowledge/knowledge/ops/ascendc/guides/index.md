# Ascend C 开发指南

- [Ascend C多层级编程接口选择参考](asc_how_to_choose_api.md)：欢迎使用 Ascend C 进行昇腾AI处理器算子开发。Ascend C不仅致力于开放芯片完备编程能力支撑实现极致性能，同时通过多层级编程API设计，让您能够根据项目需求、团队技能与性能目标，灵活选择最合适的API，在开发效率与运行性能之间取得最佳平衡。
- [cross gen migration guide](cross_gen_migration_guide/index.md)：Ascend C的核心设计目标之一是实现跨代兼容。然而，每一代芯片在微架构上都存在独特的硬件特性，因此无法保证所有接口仅通过重编译就能达成完全的跨代兼容。基于这一前提，本手册旨在为Ascend C算子开发者提供系统化指导，帮助将算子Kernel代码从当前芯片平滑移植到目标代际芯片上运行，实现高效、可控的迁移。（共 16 张知识卡：arch22 → arch35 AscendC Migration and Backward-Generation Reference、Ascend C API兼容性策略、L1 迁移实施详细指南 等）
- [getting started](getting_started/index.md)：共 7 张知识卡（如 Ascend C概述、环境准备 等）。
- [msprof Agent 使用指南](msprof_agent_guide.md)：让 Agent 用 msprof 做性能分析而不撑爆 context 的指南：先用系统级 msprof 定位瓶颈，落到单算子再用算子级 msopprof（等同 msprof op）展开；核心是分层读取——优先只读 op_statistic 聚合 csv，再按需 grep op_summary 取关键指标，避免读二进制 trace。
- [operator practice](operator_practice/index.md)：共 129 张知识卡（如 本文档组织结构（算子实践）、异构计算（算子实践）、Host Tiling 单元测试开发 等）。
- [精度验证](precision/index.md)：共 44 张知识卡（如 aclnn 接口调用指南、激活参考实现的权重状态、边界与稳定公式、APACE 编码 golden 与容差边界 等）。
- [programming guide](programming_guide/index.md)：共 121 张知识卡（如 本文档组织结构（编程指南） 等）。
- [快速开始](quick_start.md)：根据本地是否有NPU设备和使用目标选择对应的环境准备方式：
- [AscendC SIMD Development Reference](simd_development_reference.md)：AscendC SIMD 开发参考，主张 arch35/A5 上多算子计算链默认走 Regbase MicroAPI 而非 Membase LocalTensor（后者每步 UB 往返加 PipeBarrier，实测拖累可达数十倍）；另含量化 kernel 常用 SIMD API 与 MXFP4 范式。
- [technical appendix](technical_appendix/index.md)：共 8 张知识卡，分布在 2 个子目录（concepts_and_terms、cpp_standard_support）。

- [regbase](regbase/index.md)：共 34 张知识卡（如 RegBase API 的操作数、掩码与存储合同、AscArange：固定 reference 调用约束、asc_bfloat162e1m2x2 (废弃)：固定 reference 调用约束 等）。

- [blaze](blaze/index.md)：共 21 张知识卡（如 Blaze 硬件基形与运行期尾块范围的区别、权重反量化的 Cast 顺序与 V→C 释放关系、Blaze 逐元素 Epilogue 的广播映射与加载边界 等）。
- [catlass](catlass/index.md)：共 17 张知识卡（如 CATLASS FA 接口映射及参考配方冲突、CATLASS 编译架构与启动资源合同、CATLASS 的 Tile、Block 与 Kernel 组件契约 等）。
- [diagnostics](diagnostics/index.md)：共 24 张知识卡（如 assert（AI CPU API）：固定 reference 调用约束、printf（AI CPU API）：固定 reference 调用约束、Printf：固定 reference 调用约束 等）。
- [memory](memory/index.md)：共 25 张知识卡（如 对齐单位、布局行距与填充值、BF16 别名缓冲与平台采样路径、Ascend C 缓冲所有权、流水交接与槽位布局 等）。
- [moe](moe/index.md)：共 19 张知识卡（如 MoE Dispatch 与 Combine 的元数据对应关系、MoE 容量公式与路由假设、MoE Combine 的可选路径与量化角色 等）。
- [reductions](reductions/index.md)：共 34 张知识卡（如 归约算法选择与数值边界、归约轴映射与全载、分载设计、Blaze Pure Reduce 的 partial 状态与模板缺口 等）。
- [vector](vector/index.md)：共 18 张知识卡（如 Ascend C API 家族与存储位置决定的搬运、字节 Mask、Bit Mask 与数值谓词、Ascend C 向量广播的地址语义与标量替换边界 等）。

- [layouts](layouts/index.md)：共 34 张知识卡（如 广播 Mask 极性、步长与暂存复用、重排路线启发式与连续搬运证据、Gather 尾轴模式与有符号索引规范化 等）。
- [shmem](shmem/index.md)：共 31 张知识卡（如 Blaze Heavy Kernel 覆盖与性能条件、Blaze-SHMEM Host Bootstrap 生命周期与失败传播、Blaze 本地 MX 模板的 L1 尺度窗口与事件 等）。
- [synchronization](synchronization/index.md)：共 22 张知识卡（如 向量递推中的真实依赖与别名冲突、APACE ReduceScatter 时序与通知计数、原子模式切换与首次覆盖顺序 等）。
- [torch extensions](torch_extensions/index.md)：共 9 张知识卡（如 PyTorch 自定义 ACLNN 算子的四层接口契约、ACLNN 扩展辅助层的符号解析与异步资源生命周期、PyTorch Ascend C 扩展的编译单元与启动 ABI 等）。

- [communications](communications/index.md)：共 28 张知识卡（如 APACE AllGather 预触发与 scale 等待边界、APACE AllGather 三区与输出归属、APACE AllToAll Matmul 的全局与局部坐标 等）。
- [matmul](matmul/index.md)：共 20 张知识卡（如 异步 Matmul 的结果空间与完成等待、Matmul 基本块流量模型与输出对齐、Batch/Group Matmul 分区身份与分支顺序 等）。
- [sorting](sorting/index.md)：共 9 张知识卡（如 排序聚合同索引更新的排列与所有权、硬件排序的行边界、索引与 Padding、多核排序的分层归并与工作区交接 等）。

- [simt](simt/index.md)：共 10 张知识卡（如 SIMT API 迁移的签名、lane 与版本边界、SIMT 调试上下文与缓存可见性、SIMT 调试输出的资源与执行影响 等）。

- [tiling](tiling/index.md)：共 39 张知识卡（如 自适应算法的退化域与 Fallback、自适应容量分块与多行缓冲预算、自适应分块搜索、可行性与安全容量 等）。

- [environments](environments/index.md)：共 17 张知识卡（如 API 示例清单与真实编译合同、Ascend C API 家族与文档版本的兼容边界、Blaze 设备地址空间与 VF 循环约束 等）。

- [registration](registration/index.md)：共 23 张知识卡（如 Add 注册模板的跨层 shape 与 dtype 合同、算子文档模板中的接口事实与占位值边界、DAG 输出与 Foreach Tensor 区间合同 等）。

- [attention](attention/index.md)：共 58 张知识卡（如 Attention 在线输出归一化与最终除法必须一致、反向 Cube 融合需保留全部梯度路径、因果窗口、GQA 与整块跳过的证明 等）。
- [performance](performance/index.md)：共 40 张知识卡（如 APACE Host 参数接线与实际计时、APACE 调优假设与负收益的适用范围、asys 系统采样的参数范围与 msprof 下发身份 等）。

- [pooling](pooling/index.md)：共 8 张知识卡（如 池化地址对齐守卫与算术反例、池化反向的梯度贡献与 gather/scatter 归属、深度池化 Direct Buffer 路径与同步修订 等）。

- [data movement](data_movement/index.md)：共 7 张知识卡（如 批量搬运、LoopMode 与交错 Scale 偏移、DataCopyPad 多行搬运的有效长度、padding 与物理行距、DataCopyPad 的版本、重载与尾部真实范围 等）。

- [testing](testing/index.md)：共 30 张知识卡（如 APACE 多rank启动与 context 复用范围、APACE 对标路径与失败证据边界、算子测试的行为断点与属性交叉覆盖 等）。

- [compilation](compilation/index.md)：共 2 张知识卡（如 APACE 构建根目录与版本化符号、Softmax 模板ABI与独立头文件依赖）。
