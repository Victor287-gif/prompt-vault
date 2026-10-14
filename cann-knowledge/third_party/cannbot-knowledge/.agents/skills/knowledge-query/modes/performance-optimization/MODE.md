# 性能优化

性能优化通常走审计路径，因为候选必须绑定 workload、baseline、平台、测量方法和实际 outcome。

## 查询路线

1. 先取得目标 platform、输入分布、基线和 Profiling 信号，不以“性能差”作为宽查询。
2. 按主导信号分别查询目标 technology 的 `optimizations`、`concepts`、相关 `apis` 和 `runbooks`。
3. 把搬运、计算、同步、片上容量、并行度、模板和精度路线拆成独立 decision；每轮只改变一个证据角度。
4. 对候选记录适用 workload、预期机制、风险、需要冻结的指标和回退条件。

## 补搜与使用

卡片中的优化只形成可测假设。未冻结基线、测量协议和正确性门禁时不宣称收益；每个实验结果通过 outcome manifest 绑定实际日志或报告。
