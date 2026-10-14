# 算子实现

完整算子实现默认走审计路径，把语义、API、Tiling/调度、数据搬运、边界、精度和验证拆成独立 decision。

## 查询路线

1. 先查目标 technology 的同名 `operators`、`concepts` 和目标 platform，明确公式、shape/dtype/layout 与边界。
2. 按每个实现 decision 选择有实际覆盖的 `apis`、`guides`、`examples`、`optimizations` 或 `runbooks`，避免一条宽查询承载全部设计。先完整读卡并沿已有链接补证，不对空分类机械轮询；Ascend C API 子决策复用专用 Skill。
3. 跨 technology 迁移只显式加入 `shared` 和匹配 source/target 的 `interoperability`；保留问题定义，重新推导目标 DSL 参数。
4. 将采用卡片和 receipt 冻结到 retrieval state，再开始形成需要交接的实现方案。

## 补搜与使用

已有生产卡和样例只是候选路线，不自动等于当前 workload 的最优实现。目标平台构建、正确性、边界和性能结果作为 outcome 绑定到原 decision。
