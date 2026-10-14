# asccopyl0c2l1

- [asc_copy_l0c2l1](asc_copy_l0c2l1_arch_2201.md)：矩阵计算完成后，对结果进行量化处理，之后将处理结果搬运到L1 Buffer中。量化模式共分为9种，分别为： - NoQuant：不开启量化功能。 - DEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。 - F322BF16：float量化成bfloat16_t。量化结果不支持INF_NAN模式。 - F322F16：float量化成half。量化结果不支持INF_NAN模式。 - QF322B8_PRE：float量化成uint8_t/int8_t。scalar量化。 - REQ8：int32_t量化成uint8_t/int8_t。scalar量化。 - VDEQF16：int32_t量化成half。量化结果不支持INF_NAN模式。 - VQF322B8_PRE：float量化成uint8_t/int8_t。矢量量化。 - VREQ8：int32_t量化成uint8_t/int8_t。矢量量化。
- [asc_copy_l0c2l1](asc_copy_l0c2l1_arch_3510.md)：该接口用于将矩阵计算的结果从L0C Buffer搬运至L1 Buffer中，并且在搬运过程中支持与其它接口组合使用，配置多种随路能力。（目前针对量化模式的支持尚不完全，具体支持情况请以函数原型中包含的接口为准）
