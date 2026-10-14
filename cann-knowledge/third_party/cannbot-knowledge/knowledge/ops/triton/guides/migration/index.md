# GPU→NPU 迁移对照

- [flash_attention_migration](flash_attention_migration.md)：Flash Attention（fwd/bwd）12 个迁移 Diff：Grid 收缩、mask 预计算、CV 流水线、propagate_nan
- [fused_matmul_migration](fused_matmul_migration.md)：Fused Matmul 四 kernel 10 个 Diff：bias 融合进 Cube 流水线、SPLIT_K 简化
- [fused_swiglu_migration](fused_swiglu_migration.md)：Fused SwiGLU 四 kernel Diff：libdevice 替换、autotune 空间重构、SPLIT_K 原子操作移除
- [rope_migration](rope_migration.md)：RoPE Diff：1D Grid 核内循环、Block Pointer 2D→3D、libdevice 路径变更
- [softcap_migration](softcap_migration.md)：Softcap（fwd/bwd）Diff：多块循环、autotune 空间重构、enable_vf_fusion
