# API 选择

单个签名、参数或支持范围走轻量查询；多个候选 API、替代接口或工程集成决策走审计路径。

## 查询路线

Ascend C API 使用相邻 `ascendc-api-knowledge-query` 的专用消费流程，不在两个模式间重复跳转。以下路线也适用于其他 technology 的 API。

1. 用准确 API 名或连续报错短语做 literal 查询；只有能力描述时用 text。不要把“API 名 + 分散参数 + 自然语言问题”拼成必须连续出现的 literal。查询目标 technology 的 `apis`，主查询始终绑定 platform；命中后按正文核对已知 CANN version。
2. 精确 API 名在主查询零命中时，立即发起独立补搜：移除 platform，增加 `--include-unlocated --status draft --type apis --match literal`。补搜命中只表示存在平台未定位的 API 草稿，不保证来自官方来源；必须从正文和实际来源确认目标平台，不能与主查询共用 receipt 或直接视为可用。
3. 阅读选中卡的完整 Frontmatter 和正文，按缺口先追踪正文/相关中的已有链接；需要调用形态时看同 technology 的 `operators` 或已有 `examples`，需要使用步骤/机制时看 `guides`、`concepts`，失败经验看 `runbooks`。先确认这些 Profile 有实际覆盖，不反复查询空分类。
4. 比较签名、参数单位、dtype/layout、内存位置、同步语义、平台版本和失效边界，不因名称相似合并接口。
5. 正文未展开准确契约时沿固定 source 定向核验，不先全盘搜索本地 SDK。

## 补搜与使用

完成未定位 draft 补搜后仍零命中，或候选存在冲突时，再改变证据角度：API 名转参数/错误短语，接口卡转官方样例或版本迁移证据。输出区分规范事实、平台待核验候选、候选集成方式和必须在目标环境验证的行为。
