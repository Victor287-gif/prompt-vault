# 系统初始化

- [asc_init](asc_init.md)：本接口将当前AI Core上可能由前序算子遗留的全局状态恢复为接口约定的初始状态，避免这些状态影响当前算子的计算结果。本接口仅初始化控制状态，不会清零Global Memory、Local Memory或用户数据。
