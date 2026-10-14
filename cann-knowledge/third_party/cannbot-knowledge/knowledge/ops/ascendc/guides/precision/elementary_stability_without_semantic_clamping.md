---
type: Guide
title: 初等函数稳定变换与语义保留
description: 纠正 Exp 范围、无条件 epsilon/截断及 Sinh 套用 Softmax 平移的错误。
tags:
- task.design
- topic.precision
- topic.compute.elementwise.math_function
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/common-traps.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/case-studies.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-precision-debug/references/debug-summary-template.md
  role: supporting
created_at: '2026-09-11T02:40:45Z'
updated_at: '2026-09-11T02:40:45Z'
platforms: []
---

# 初等函数稳定变换与语义保留

浮点范围与有效精度是不同限制。FP16最大有限值65504，ln(65504)约11.09；FP32最大有限值的自然对数约88.72。具体Exp溢出舍入边界由实现决定，但来源“FP16在x>88才溢出”显然混用了类型。log(0)与log(负数)也应按API/算子特殊值合同区分，不能一概只称无定义并任意替换。

对有限且有有效元素的Softmax，减max使Exp输入不为正，可避免该处正向指数溢出；分母仍可能在低精度长行求和时溢出或失去有效位。应按需要使用较宽累加和归一化，不能先在FP32求和又过早cast为half而丢失好处。NaN/Inf、全masked和空行另需定义。LogSumExp必须把max加回；源码half返回类型却只写外部output且无return的骨架不是完整API实现。

普通Sinh=(exp(x)-exp(-x))/2在接近0时会抵消，拓宽关键中间值可降低该误差，但不能恢复输入已丢失信息，也不保证全域准确。原总结模板对Sinh提出“先减最大值再Exp”却没补回缩放，改变了结果；Softmax的共同因子抵消不能直接套到独立Exp/Sinh。大幅值时中间Exp与最终Sinh的可表示范围也不同，需对完整公式处理范围。

sqrt(x+1)-sqrt(x)在实数x>=0可有理化为1/(sqrt(x+1)+sqrt(x))，1-cos(x)可改为2*sin(x/2)^2。它们在目标域的数学等价不保证有限精度、无穷值和溢出下逐位等价；评估实现可用的函数精度及中间范围。把已舍入half a/b拓宽后相减也不能恢复原始实数位数。

Sigmoid的1/(1+exp(-x))在负大x仍会产生大指数，可按符号选等价形式：x>=0使用原形式，x<0使用exp(x)/(1+exp(x))，再按特殊值合同核对。Exp(min(x,10))、分母加epsilon、用max(denom,eps)或小分母直接输出0都会改变原数学语义，尤其负分母；仅在算子定义要求时采用，epsilon还需检查目标dtype能否表示。

Sum/Mean存在累计误差，Max/Min不做加法但仍有NaN、tie/索引、padding和输入量化语义，不能说完全不受精度影响。来源SoftmaxV5短列失败不能证明所有Reduce最少8元素；应核对具体API、count/mask和tail实现。RoundMode的支持组合与语义也按具体接口核验，不把CAST_ODD/HYBRID等的来源概括或“多行相同必为ROUND错误”当普适结论。

验证小量、抵消、大幅值、长行、特殊值与原算子域，保留原有舍入点和完整输出。来源精度改善数字及案例根因没有在本批设备复现；Host数学检查仅确认本文所列数值范围和输入表示反例。
