---
name: knowledge-query
description: 检索 cannbot-knowledge 中受治理的 CANN、算子、图、模型与 Runtime 知识。在技术问答、设计、实现、调试、优化、测试或跨技术适配的首次领域决策前使用；新证据缺口出现时定向重入。先确认平台，由脚本返回候选路径、title 和 description；Agent 选卡阅读全文，再按需沿链接阅读相关知识或 include。复杂任务支持检索审计。
---

# Knowledge Query

以当前问题为中心完成“确认目标平台 → 脚本返回候选路径、title、description → Agent 选卡阅读全文 → Agent 决定是否继续阅读相关内容或 include → 回答”。脚本只负责召回，选卡、阅读深度和证据采用由 Agent 决定。

## 1. 定位任务与检索边界

- 明确一个要回答的判断（decision），保留目标实体、准确 API 名或首发错误信号。复杂任务拆分判断，不把实现、调试和性能问题塞入一条 query。
- 先确定**任务目标平台**，不是默认采用 Agent 所在机器。只有型号、SocVersion、架构宏或配置时，读取[平台识别](references/platform-identification.md)；存在歧义或冲突时先确认，不猜测或移除平台条件。安装项目可在 `.cannbot/knowledge.env` 配置 `CANNBOT_KNOWLEDGE_PLATFORM` 作为缺省平台，省略 `--platform` 时由脚本回退到该值；任务目标与缺省值不一致时必须显式传 `--platform`。
- 明确必要的 domain、technology 与路径 Profile。只增加已知且有助区分的过滤项，不为了“最窄”叠满 task/tag；多个 tag 和 platform 都是 AND。共同适用知识用多平台过滤，平台差异分开查询。
- 首次进入一个知识仓时运行 `discover`，区分真实覆盖与注册/占位目录；同一轮任务、同一知识根且索引指纹未变时复用结果。它的各维度计数不证明某个组合已有覆盖。

按 [CLI](references/cli.md) 定位知识根、参数与输出。索引缺失、损坏或 stale 时停止，交由知识仓维护者或项目安装器准备；Query 不扫描全库兜底、不静默重建，也不自行下载资源。

使用安装器保留完整知识仓中的规范源与 Contract 依赖；仅配置数据根不能使独立复制的 Skill 获得全部能力。安装边界与维护者的索引准备入口均见 CLI。

## 2. 选择流程，只加载需要的参考

单 API、单错误码或窄概念问答走轻量路径：检索、读卡、补齐必要证据后直接回答，不强制创建审计文件。完整实现、多 API 选型、复杂故障、性能优化或正式交接走[查询与证据工作流](references/query-workflows.md)。

按需选一个主要侧面：[算子实现](modes/operator-implementation/MODE.md)、[API 选择](modes/api-selection/MODE.md)、[故障诊断](modes/failure-diagnosis/MODE.md)、[性能优化](modes/performance-optimization/MODE.md)、[技术问答](modes/operator-qa/MODE.md)。模式是内部参考，不是需要逐个执行的 Skill。

Ascend C API 契约与选型使用同装的 [ascendc-api-knowledge-query](../ascendc-api-knowledge-query/SKILL.md)，复用本 Skill 的平台、索引与审计，并补充重载区分和本地 `source_headers` 读取。已经从该流程进入 Query 时直接执行底层查询，不再回跳；其他技术的 API 使用通用 API 模式。

## 3. 构造能召回的查询

| 问题形态 | 初始 query 与匹配方式 |
|---|---|
| 已知准确 API、类型、错误码或连续日志短语 | 保留该标识/短语，使用 `--match literal`；不要追加自然语言问题或分散的参数 |
| 只有能力需求、机制或现象 | 使用“实体 + 一个关键约束”的短词组，使用 `--match text` |
| 同名 API、不同家族或调用方向 | 先查准确名字；按返回路径、完整声明和范围区分，再用已注册分类或正文术语定向补查 |

例如查询 API 契约：

```bash
python3 <skill-dir>/scripts/knowledge_query.py --knowledge-root <repo> search \
  --query "DataCopyPad" --domain ops --technology ascendc --type apis \
  --platform <confirmed-platform> --status stable --match literal -k 5
```

配置了项目缺省平台后，普通 search 可以省略 `--platform`；显式平台始终优先：

```bash
python3 <skill-dir>/scripts/knowledge_query.py search \
  --query "DataCopyPad" --domain ops --technology ascendc --type apis \
  --status stable --match literal -k 5
```

没有显式平台且项目未配置 `CANNBOT_KNOWLEDGE_PLATFORM` 时，search 仍以结构化错误退出，不猜测平台。

`literal` 匹配的是**完整查询串**，不是多个词的 AND；不要把 `DataCopyPad srcStride 非对齐` 当作一个必然存在的原文短语。`text` 按词法相关性排序，也不证明每个约束都满足。结果 `count` 是本次返回条数，受 k 限制，不是全库匹配总数。

初始只取少量候选；如果尚未找到目标定义且结果已达到 k，再提高 k 或按真实家族细化。精确标题优先于标题片段、摘要、正文中的提及，但排序不能代替声明核对。默认只查 `stable`；草稿、历史接口或 shared 需有明确问题理由，不把其他 DSL 的同名符号混进目标候选。

## 4. 阅读与补证，而不是交付搜索摘要

1. 先查看脚本返回的 `path` / `local_path`、`title` 和 `description`，由 Agent 决定哪些命中卡片值得继续阅读，不要求打开全部候选。对选中的卡片读取完整 Frontmatter 和正文；输出分页或截断时继续读到末尾。结果只返回最多两个 sources，完整来源必须回卡片读取。
2. 核对目标平台、status、版本、约束、失效条件和当前 decision。`stable` 是生命周期；`verified` 派生的 trust tier 影响排序，不证明技术结论正确。Query 不按 CANN 版本过滤；Runbook 卡不携带 `cann_versions`（版本边界写入正文），其他类型卡可能有该字段，仍须读正文的具体限制。
3. 全文读完后，由 Agent 根据当前需求决定是否查看相关内容或 include。正文足够时直接回答；需要参数/配套接口、机制、算子用法或失败经验时沿对应知识链接阅读，需要声明或实现信息时再看固定来源或 Ascend C `source_headers`。新卡片和头文件中的链接也可继续跟进，不限制为一跳，也不要求遍历全部链接。
4. 知识库链接是有效入口，直接沿链接打开，不需要先运行链接检查。普通相对链接以当前卡目录解析，前导 `/` 的知识链接以 `knowledge/` Bundle 根解析，`#锚点` 指向当前卡；跳到 `index.md` 后按需选择真实卡。Agent 仍需判断新卡内容的平台、版本和适用性。链接及其说明不替代目标卡的证据；列出的算子不表示完整调用集合。
5. 记录已读卡和未解决项以避免循环；读过的适用证据足够回答时停止。只看过的摘要、打不开的源码、未读头文件不能写成已核验。

不同本地路径的基准和缺失处理见 [CLI](references/cli.md)。普通读卡不要求 SDK 存在；只有依赖源码才能回答的部分才需要继续定位源码，不为一个已被正文充分回答的问题扫描整个 SDK。

## 5. 零命中、补搜与停止

- 先区分错误与零命中：退出码 `2` 是参数/索引/输入错误，应处理明确原因或报告阻塞；退出码 `0` 且无结果才是有效零命中。
- **精确 API 名在目标平台 stable 查询零命中时**，立即独立补搜未定位 draft：保留 API 名与 domain/technology，使用 `--type apis --match literal --include-unlocated --status draft`，不带 platform。它只提供平台待核验线索，不能混入原平台结果、共用 receipt 或代替最初的平台识别。
- 普通补搜保持已确定的目标平台和技术边界，只改变一个有依据的角度：准确标识 → 参数/错误信号，接口 → 使用指南/算子调用，现象 → 已验证修复。可移除没有证据支持的可选 tag/task，或选择更合适的 Profile；不要先放宽到所有技术、所有状态。
- 有名字空间或成员限定符但卡片只记短名时，可按短名补搜，再用声明验证家族；不能把短名的任意命中都算目标 API。
- 补搜无新增证据、来源仍冲突、目标仍不明确或预算已到时，给出已有结论与具体缺口。零命中不证明 API 或能力不存在；命中也不等于当前问题已覆盖。

`agnostic` 卡自动匹配明确的目标平台，不能用来表示平台未知。`platforms: []` 不满足普通过滤；未定位 draft 查询不能带 platform，结果必须单独说明。不能为了召回修改知识卡的平台或状态。

## 输出与权限边界

先直接回答当前问题，再给实际读过的卡片/来源位置和必要的范围、冲突或验证建议。区分文档事实、本机源码观察、工程推断与已做实验；不倾倒所有候选和导航。

检索保持只读；只在调用方指定且目录项尚不存在的路径创建审计产物，已有符号链接也须拒绝，不沿其写入其他文件。不覆盖旧 receipt、修改知识、索引、图谱、SDK 或全局环境。复杂任务的 state/outcome 绑定规则见[证据工作流](references/query-workflows.md)，不为轻量问答制造额外审计流程。
