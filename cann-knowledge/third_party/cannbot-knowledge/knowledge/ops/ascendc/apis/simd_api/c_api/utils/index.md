# 工具接口

- [系统初始化](sys_init/index.md)：本接口将当前AI Core上可能由前序算子遗留的全局状态恢复为接口约定的初始状态，避免这些状态影响当前算子的计算结果。本接口仅初始化控制状态，不会清零Global Memory、Local Memory或用户数据。
- [系统其他操作](sys_misc/index.md)：生成一条空操作指令，不执行实际计算。每调用一次阻塞一拍流水。
- [系统资源和变量](sys_var/index.md)：共 15 张知识卡（如 asc_get_arch_ver、asc_get_core_id、asc_get_ffts_base_addr 等）。
