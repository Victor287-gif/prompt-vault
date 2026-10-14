# SIMT算子性能优化

- [control flow](control_flow/index.md)：同一Warp内线程应避免分支跳转以防止分支发散，减少掩码屏蔽等待，提升SIMT算子执行效率。
- [execution config](execution_config/index.md)：共 2 张知识卡（如 线程块数量配置优化、合理配置线程数避免寄存器溢出）。
- [instruction optimization](instruction_optimization/index.md)：【描述】SIMT编程模式下，多个线程更新同一个内存地址时需要使用原子操作接口保证更新结果正确。原子操作属于高开销指令，其耗时由两方面决定：一是同一地址上的原子操作只能排队依次执行，二是单条原子指令的处理开销。
- [memory access](memory_access/index.md)：共 5 张知识卡（如 避免UB的Bank冲突、DCacheAccess、MemoryAccessMerge 等）。
- [SIMT算子优化概述](overview.md)：SIMT编程模式以线程（Thread）为基本编程单位，由开发者编写单个线程的执行逻辑，再通过>>配置将大量线程组织为线程块（线程块）与线程网格（Grid），调度到硬件的Vector Core（物理核，即AIV核）上以Warp为单位并行执行。
