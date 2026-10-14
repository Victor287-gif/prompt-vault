# asccopyl0c2gm

- [asc_copy_l0c2gm](asc_copy_l0c2gm_arch_2201.md)：矩阵计算完成后，对结果进行量化处理，之后将处理结果搬运到GM中。量化模式共分为9种，分别为： - NoQuant：不开启量化功能。 - F322F16：float量化成half。量化结果不支持INF_NAN模式。 - F322BF16：float量化成bfloat16_t。量化结果不支持INF_NAN模式。 - DEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。 - VDEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。 - QF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。 - VQF322B8_PRE：float量化成uint8_t/int8_t。矢量量化。 - REQ8：int32_t量化成uint8_t/int8_t。scalar量化。 - VREQ8：int32_t量化成uint8_t/int8_t。矢量量化。
- [asc_copy_l0c2gm_arch_3510](asc_copy_l0c2gm_arch_3510.md)：将矩阵计算结果从L0C Buffer搬运至Global Memory，搬运过程中可同步支持随路量化、随路激活、随路格式转换（Nz2ND/Nz2DN）等能力组合。
