---
type: "Runbook"
title: "dav-2201 Rsqrt 精度不足 — 改用 Sqrt 加标量倒数"
description: "RmsNorm 的 fp32 rstd 使用不带 config 的 Rsqrt 后呈现约 2^-9 量化偏差；改为 Sqrt、标量读回与 1.0f/s 后恢复 fp32 精度。"
tags: ["task.optimization", "topic.compute.normalization", "topic.compute.elementwise.math_function", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
aliases: ["KernelRmsNorm::Compute"]
---

# dav-2201 Rsqrt 精度不足 — 改用 Sqrt 加标量倒数

## 现象

RmsNorm 的 fp32 用例出现约 `1e-2` 最大输出差，而 fp16/bf16 因容差较宽可能通过。确定性 `x=1, gamma=1, eps=1e-6` 探针跨多个短、长归约长度都给出 `rstd=0.998046875`，期望约 `0.9999995`；随机输入也出现 `1.03515625` 对 `1.03701699` 的稳定偏差。整齐的二进制量化值是快速倒数精度不足的判别信号。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`AscendC::Rsqrt`、`AscendC::Sqrt`、`dav-2201`、`rstd` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

本轨迹中不带 config 的 `Rsqrt<float>` 在 dav-2201 上产生约 9-bit 量级的快速倒数结果。规约、均值和加 epsilon 链条随 D 正常变化，且固定 ones 输入对所有 D 得到相同低精度 rstd，排除了行索引、ReduceSum 除数和随机噪声。单指令近似精度不足是唯一根因。

## 解决配方

### before

```cpp
    __aicore__ inline void Compute(int64_t r)
    {
        (void)r;
        auto scalar = scalarBuf_.Get<float>();
        auto sq = sqBuf_.Get<float>();
        auto dummy = dummyBuf_.Get<float>();

        if constexpr (std::is_same<T, float>::value) {
            auto xFp32 = inQueueX_.DeQue<float>();
            AscendC::Mul(sq, xFp32, xFp32, numCol_);                              // x^2
            AscendC::ReduceSum<float>(scalar, sq, dummy, numCol_);               // sum(x^2)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scalar, scalar, 1.0f / static_cast<float>(numCol_), 1); // mean
            AscendC::Adds(scalar, scalar, epsilon_, 1);                         // mean + eps
            AscendC::Rsqrt<float>(scalar, scalar, 1);                           // rstd = 1/sqrt(mean+eps)
            AscendC::PipeBarrier<PIPE_V>();
            float rstd = scalar.GetValue(0);

            auto yOut = outQueueY_.AllocTensor<float>();
            AscendC::Muls(yOut, xFp32, rstd, numCol_);                          // x * rstd
            AscendC::Mul(yOut, yOut, gammaFp32Local_, numCol_);                 // * gamma
            outQueueY_.EnQue(yOut);
            inQueueX_.FreeTensor(xFp32);
        } else {
            auto xT = inQueueX_.DeQue<T>();
            auto xFp32 = xFp32Buf_.Get<float>();
            AscendC::Cast(xFp32, xT, AscendC::RoundMode::CAST_NONE, numCol_);  // x -> fp32
            inQueueX_.FreeTensor(xT);
            AscendC::Mul(sq, xFp32, xFp32, numCol_);                            // x^2
            AscendC::ReduceSum<float>(scalar, sq, dummy, numCol_);              // sum(x^2)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scalar, scalar, 1.0f / static_cast<float>(numCol_), 1); // mean
            AscendC::Adds(scalar, scalar, epsilon_, 1);                        // mean + eps
            AscendC::Rsqrt<float>(scalar, scalar, 1);                          // rstd = 1/sqrt(mean+eps)
            AscendC::PipeBarrier<PIPE_V>();
            float rstd = scalar.GetValue(0);

            AscendC::Muls(sq, xFp32, rstd, numCol_);                           // y = x * rstd (reuse sqBuf)
            AscendC::Mul(sq, sq, gammaFp32Buf_.Get<float>(), numCol_);         // * gamma
            auto yOutT = outQueueY_.AllocTensor<T>();
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(yOutT, sq, roundMode, numCol_);                      // y -> T
            outQueueY_.EnQue(yOutT);
        }
    }
```

### after

```cpp
    __aicore__ inline void Compute(int64_t r)
    {
        (void)r;
        auto scalar = scalarBuf_.Get<float>();
        auto sq = sqBuf_.Get<float>();
        auto dummy = dummyBuf_.Get<float>();

        if constexpr (std::is_same<T, float>::value) {
            auto xFp32 = inQueueX_.DeQue<float>();
            AscendC::Mul(sq, xFp32, xFp32, numCol_);                              // x^2
            AscendC::ReduceSum<float>(scalar, sq, dummy, numCol_);               // sum(x^2)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scalar, scalar, 1.0f / static_cast<float>(numCol_), 1); // mean
            AscendC::Adds(scalar, scalar, epsilon_, 1);                         // mean + eps
            AscendC::Sqrt(scalar, scalar, 1);                                  // sqrt(mean+eps)
            AscendC::PipeBarrier<PIPE_V>();
            float s = scalar.GetValue(0);
            float rstd = 1.0f / s;  // rstd = 1/sqrt(mean+eps); NaN/inf propagate (mean+eps >= eps > 0 for finite)

            auto yOut = outQueueY_.AllocTensor<float>();
            AscendC::Muls(yOut, xFp32, rstd, numCol_);                          // x * rstd
            AscendC::Mul(yOut, yOut, gammaFp32Local_, numCol_);                 // * gamma
            outQueueY_.EnQue(yOut);
            inQueueX_.FreeTensor(xFp32);
        } else {
            auto xT = inQueueX_.DeQue<T>();
            auto xFp32 = xFp32Buf_.Get<float>();
            AscendC::Cast(xFp32, xT, AscendC::RoundMode::CAST_NONE, numCol_);  // x -> fp32
            inQueueX_.FreeTensor(xT);
            AscendC::Mul(sq, xFp32, xFp32, numCol_);                            // x^2
            AscendC::ReduceSum<float>(scalar, sq, dummy, numCol_);              // sum(x^2)
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scalar, scalar, 1.0f / static_cast<float>(numCol_), 1); // mean
            AscendC::Adds(scalar, scalar, epsilon_, 1);                        // mean + eps
            AscendC::Sqrt(scalar, scalar, 1);                                 // sqrt(mean+eps)
            AscendC::PipeBarrier<PIPE_V>();
            float s = scalar.GetValue(0);
            float rstd = 1.0f / s;  // rstd = 1/sqrt(mean+eps); NaN/inf propagate (mean+eps >= eps > 0 for finite)

            AscendC::Muls(sq, xFp32, rstd, numCol_);                           // y = x * rstd (reuse sqBuf)
            AscendC::Mul(sq, sq, gammaFp32Buf_.Get<float>(), numCol_);         // * gamma
            auto yOutT = outQueueY_.AllocTensor<T>();
            constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(yOutT, sq, roundMode, numCol_);                      // y -> T
            outQueueY_.EnQue(yOutT);
        }
    }
```

## 设计决策

保留 fp32 规约与既有缓冲，只替换 rstd 的最后一步，避免把精度问题误治为 tiling 或 Cast 问题。有限输入下 `mean+eps > 0`，无需 `s>0` 保护；该保护会在 `s=NaN` 时错误选择 0，破坏整行 NaN 传播。

## 性能影响

新路线增加一次标量读回和标量除法，轨迹未提供配对性能数据；本卡只主张精度恢复，不主张性能收益。

## 预防

- 对快速倒数/倒数平方根 API 用 `x=1` 和非整平方输入做目标平台精度探针；
- 分 dtype 检查，避免低精度宽容差掩盖 fp32 的系统偏差；
- 不把其他平台或带 config 形态的精度承诺迁移到当前调用形态。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：default-eps fp32 有限值最大差 1.315689e-02，边界子集汇总 5 failed、5 passed。（来源平台：a3；证据等级: tool_observed）。
- 修复后：Sqrt+标量倒数实现及 NaN 传播修正后完整 NPU 测试 167 passed。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

该结论只在 a3 路径验证，不覆盖 a2、950、带精度 config 的 API 或其他 CANN 版本；扩展平台前须按目标设备精确判型并复测。

## 坏实践

- 看到 `Rsqrt<float>` 就假设结果具有完整 fp32 精度；
- 只跑 fp16/bf16 用例，用宽容差宣布数值路径正确；
- 用 `s>0 ? 1/s : 0` 保护有限域，却忽略 NaN 比较恒为 false 会破坏传播。

## 误诊

若 rstd 误差随 D、行号或输入地址随机变化，先查规约、同步和索引；本问题的特征是同一数学输入得到稳定、整齐的低精度量化值。