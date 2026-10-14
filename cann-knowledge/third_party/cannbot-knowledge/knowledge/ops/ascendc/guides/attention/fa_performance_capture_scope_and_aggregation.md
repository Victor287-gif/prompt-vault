---
type: Guide
title: FA 性能采集范围与汇总口径
description: 保存跨框架计时陷阱，修正公平下界与唯一测量方法的泛化。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-perf-gate-evaluation.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/flash-attention.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-handcraft-pitfalls.md
  role: supporting
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# FA 性能采集范围与汇总口径

性能比较需同公式、输入、dtype、layout、设备/争用窗口和计时范围。device kernel执行、同stream事件区间及包含Host调度/布局转换的端到端时长是不同指标。Host初始化销毁和executor重建可显著扩大后两者，不能据纯device结果解释为Host框架开销带来的加速。

来源torch进程ctypes launch逐次event取min出现约1.6–1.7us恒值和391倍假加速，是事件未覆盖实际外来stream/操作等问题的线索。可用实际launch stream上的ACL事件组回放并与profiler核对；不能推广为全部torch event永远无法量外来kernel。分组3*10取min/均值/median都要标明聚合顺序，某一最小值尤其易放大漏测。

ATB executor重建例event4061us/device76us说明区间可能含Host造成的GPU/NPU空闲，选择profiler纯device可隔离该部分；端到端吞吐仍可合法用同步包围的多次回放，不能称墙钟一律无效。NNAL ATB环境脚本和torch_npu导出解析触发属于来源安装/版本要求，不是所有安装路径固定或仅export_chrome_trace一种导出。

profiler过滤只留AI_VECTOR_CORE/AI_CORE会漏MIX_AIC融合kernel；类型字段及名称按导出版本核验。磁盘满、trace截断、进程OOM和解析未完成都可造成异常短时长，不能只凭15us对72–78us差异认定工具盲区。独立进程逐case可降低累积内存影响，也会改变缓存/初始化条件。

多kernel链逐项device duration之和忽略Host间隙，但若核间/stream间重叠，它也不是必然的端到端公平下界；需区分工作量总和与时间线span。分解链与融合基线应显式命名，source33.5倍墙钟对1.16倍device不能当同一加速指标。

gate可用exp(mean(log(tRef/tCustom)))，要求各时长正且有限、逐case配对及权重明示；失败/缺测/0/0不能算通过或静默排除。算术均值对大ratio敏感，几何均值也不替代逐case回退/尾部风险。来源1.025、3.047、0.107–0.364等不同工程历史值不能作新实现门槛。

争用时将memset和kernel分事件区间可定位排队成本，但队列等待不保证只发生在首操作；若算法每launch确实需要清workspace，端到端仍应计入。kfc只清64KiB取决于实际系统区合同。两侧同窗口交替、多轮、同配置可降低环境偏差，不能保证完全相同争用。torch.cuda.Event并非NPU事件替代；同步墙钟均摊仍含Host部分，应与device口径分列。
