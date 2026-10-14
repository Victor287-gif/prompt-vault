# 硬件参考

Ascend 910B/910_95/950PR/910C 硬件规格与强制约束速查。生成任何 Triton Ascend kernel 之前必须过一遍约束检查清单。

- [ascend_910b](ascend_910b.md)：910B2/910B3 核心架构、Grid 语义、内存层次、编译选项
- [ascend_910_95](ascend_910_95.md)：910_95/950PR/950DT（dav-c310，Reg-based A5 架构）核数、UB/L1/L0C 容量、专有数据通路与 910B 差异
- [NPU 内存层次与数据通路模型](npu_memory_model.md)：Ascend NPU 的 GM/L1/UB/L0A/L0B/L0C 存储层次、容量与对齐约束、MTE/FIX 数据通路及 tl.load/tl.store/tl.dot 的硬件映射。
- [constraints_cheatsheet](constraints_cheatsheet.md)：强制约束速查（UB/Gird/对齐） + 性能约束 + autotune 搜索空间
- [ascend_patterns](ascend_patterns.md)：Ascend 特有的编程模式与 GPU→Ascend 迁移差异
- [Ascend950PR (David V100) -- Target Platform](target_ascend950pr.md)：950PR（DAV_3510/arch35）完整平台规格：核数、UB/L1/L0 容量、6 管线与 Scalar 单元、SIMT+SIMD 模型、原子操作、跨核同步、Cube 数值行为与已知缺陷（手册+白皮书+实测）。
- [Ascend910C (DaVinci V220, dual-die) — Target Platform](target_ascend910c.md)：910C（A3）平台规格：SIMD-only、80~96 AIV、UB 192KB/AIV、L0C 128KB、SOC 字符串与构建参数、与 A5 移植差异及待实测项。
- [Ascend910B (DaVinci V220) — Target Platform](target_ascend910b.md)：910B（A2）平台规格：SIMD-only、B1–B4 SOC 型号、UB 192KB/AIV、L0C 128KB、构建参数与待上机验证项。

- [Ascend310P (dav-m200 / 200x) -- Target Platform](target_ascend310p.md)：Ascend310P（dav-m200/200x）目标平台参考：交叉编译限定的证据契约，含 SoC 身份、能力清单与证据 JSON 说明
