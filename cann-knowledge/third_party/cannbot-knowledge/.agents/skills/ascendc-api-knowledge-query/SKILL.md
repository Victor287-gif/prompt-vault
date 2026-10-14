---
name: ascendc-api-knowledge-query
description: 专门检索 Ascend C API。先确认硬件平台，由脚本返回候选卡片的路径、title 和 description；Agent 选择需要阅读全文的卡片，读完后再决定是否沿链接阅读相关知识和本地 CANN include 头文件。不用于 ACLNN 接口检索或完整算子实现分析。
---

# Ascend C API 检索

完成 **确认平台 → 脚本召回摘要 → Agent 选卡全文阅读 → 按需沿链接/源码/include 补证 → 复核并回答**。复用相邻 [knowledge-query](../knowledge-query/SKILL.md) 的索引，不再启动另一套查询流程。

下文 `<api-skill-dir>` 是本 Skill 根目录，`<query-skill-dir>` 是相邻 `knowledge-query` 目录。查询自动从项目 `.cannbot/knowledge.env` 定位知识仓；仅在明确覆盖时传 `--knowledge-root <知识仓根>`。参数/配置错误查 [CLI](../knowledge-query/references/cli.md)，不要遍历目录找脚本。

## 1. 确认平台

按[平台识别](../knowledge-query/references/platform-identification.md)把用户的目标产品规范化为 Registry 平台 ID。只有架构宏、兼容枚举或其他歧义信号时，询问精确产品后结束，不默认平台、改求交集或补搜草稿。用户明确要求多平台共同适用才使用多个 `--platform`（AND）；比较差异则分别查询。

## 2. 召回候选摘要

首次进入知识仓执行 `python3 <query-skill-dir>/scripts/knowledge_query.py discover`；同根、同索引指纹的已有覆盖可复用。随后查询：

```bash
python3 <query-skill-dir>/scripts/knowledge_query.py search \
  --query "<准确 API 名>" --domain ops --technology ascendc --type apis \
  --platform <已确认平台> --status stable --match literal -k 5 --text
```

准确名字用 `literal`，只有能力描述时用 `text`；literal 不是关键词组合，不附加自然语言问题。`--type` 使用路径分类 `apis`、`operators` 等，不使用 OKF 类型 `API`、`Operator`。

脚本返回带本地链接的 title、path/local_path、description 和行数，由 Agent 选卡，不打开全部命中。`no matches` 才是零命中；其他卡片提及 API 不等于找到定义。缺定义且结果已满 k 时可细化或扩大 k；仍缺定义时独立补搜未定位 draft，stable 精确名查询零命中时必须补搜：

```bash
python3 <query-skill-dir>/scripts/knowledge_query.py search \
  --query "<准确 API 名>" --domain ops --technology ascendc --type apis \
  --match literal --include-unlocated --status draft -k 5 --text
```

只有未定位 draft 补搜不带 platform；其结果单独标为平台待核验线索，不能冒充正式支持或混入原 receipt。零命中不证明 API 不存在。正式审计才使用 JSON/receipt；普通问答直接消费 `--text`，不再接管道摘要。

**必要查询（包括 discover）被权限拒绝，或索引/知识根不可用时，报告阻塞并结束。** 不继续 search、改工具读技术卡、扫描知识树、重建索引或借 Skill 的导航回答技术问题。仅参数/配置错误可按 CLI 修正，不能删除普通查询的 platform 来规避。

## 3. 全文阅读选中卡片

```bash
python3 <api-skill-dir>/scripts/resolve_resources.py read --card "<选中 local_path>"
```

逐页执行输出的 `next_command`，直到同一 `file_sha256` 下第 1 行至末行连续读齐。结束行和续读位置只在页尾返回；看不到页尾就先补齐当前页，不推测下一页位置。不要给阅读命令接 `head`/`tail` 截断正文。文件改变就从头读。末页或 EOF 单独不能证明已读全文，分页指纹也不是 Query 审计的 `card_sha256`。

`index.md` 是唯一导航文件；其他 Markdown（包括汇总、Operator、Runbook）选中后都全文阅读。正文中的普通链接可帮助选下一张卡，不能用未读目标的标题或链接说明作佐证。核对当前卡的声明、平台、生命周期与限制；`stable` 不是技术正确性的证明。

链接原样交给工具：普通相对路径以当前卡目录为基准，`/ops/...` 等前导 `/` 路径以 `knowledge/` Bundle 为基准，`#章节` 指向当前卡。解析复用治理 Contract，不把 Bundle 根相对路径当作主机绝对路径；同页锚点不产生新卡或新的已读证据，避免循环阅读。

长卡可先取骨架再决定读法：

```bash
python3 <api-skill-dir>/scripts/resolve_resources.py outline --card "<选中 local_path>"
```

outline 返回正文标题、行号与总行数，只用于判断相关性和定位章节；它不提供正文内容，也不构成已读证据，采用本卡前仍须按上述规则全文读齐。

## 4. Agent 按问题缺口补证

读完后决定是否继续；证据足够就停止检索，不遍历所有邻接。首次需要补充材料时读[资源解析与证据核对](references/evidence-check.md)，按需使用以下入口：

| 所需证据 | 取证方式 |
|---|---|
| 相关 API、指南或问题经验 | 目标为知识卡时，原样复制链接，用 `read --card "<当前卡>" --link "<原始相对链接>"` 解析并读目标；续页使用返回的目标 `card`。目标为 `index.md` 时按补证参考用 `link` 解析，打开导航后再选择知识卡；`read` 不接受索引 |
| 实际调用算子 | 沿 API 相关链接选 **Operator 卡**并全文读完；无合适链接时以准确 API 名、当前平台及 `--type operators --match literal` 定向查询，再按下面的 source 命令读真实源码 |
| 本机声明、结构体、实现依赖 | 从已读卡 `source_headers` 选路径，确认用户包内 Ascend C **组件根**后按下面的 headers 命令读取；依赖按实际 include/构建搜索路径继续 |
| 组件版本或版本冲突 | 按补证参考核对对应组件的版本元数据及目标 include 安装归属；安装目录名、其他组件版本均不能代替版本证据 |

```bash
python3 <api-skill-dir>/scripts/resolve_resources.py source \
  --card "<已全文读过的 Operator 卡绝对路径>" "<其 sources[].resource 原值>"
python3 <api-skill-dir>/scripts/resolve_resources.py headers \
  --package-root "<用户 CANN 包内已确认的组件绝对路径>" "<source_headers 中的 include/... 原值>"
```

这两个命令只解析资源；Agent 还须打开返回的 `absolute_path`，读实际调用/声明和必要上下文。源码位于知识仓而非 Bundle/SDK；include 相对组件根，值已含 `include/`，只拼接一次。Runbook 摘录、SDK 示例和卡片里的源码摘录不代替本轮读取 Operator 声明的固定源码。“部分算子”只代表注明范围的展示样本，不是完整调用集合。

源码/include 缺失时说明未读部分，不扫描其他安装目录、自动下载或执行包内脚本；已有正文足以回答的部分仍可回答。只问声明时无需额外调查版本，注明本机版本未核验。资源路径错误按补证参考定位，不手算多层 `../` 或找同名文件冒充目标。

## 5. 复核并回答

将拟答复视为待审草稿，用已读原文反查每项技术判断，考虑原文允许的反例；用户没有给出实现时，判断方法能否满足目标，不预测假想程序分支。复核由当前 Agent 完成，无需另建文件、委派或重读整批材料。

跨卡采用参数/公式前，用 `compare --card "<已读 API 卡>" --card "<已读补充卡>" --field "<准确字段>"` 集中对照原文，按补证参考核对原始赋值和单位。**先处理冲突，再形成结论**；缺少 SDK 不阻碍按文中已有单位换算。冲突原式只能作为待纠正项说明，不能在前文采用、末尾才写待核验。

只回答原问题，附实际读过的证据链接及影响结论的范围/缺口。引用复制查询标题链接或阅读输出的“引用此卡”；未读导航不作佐证，免责声明也不能把它变成证据。参数问题给字段、数值和单位，代码仅按已读声明作必要改动，不补充未核对的替代接口、重载或版本。最终只交更正后的答案，不展示内部草稿或逐步自检。

检索全程只读，不修改知识、索引、SDK 或全局环境，不下载、编译或执行上游代码。复杂多决策与正式交接再使用 Query 的[证据工作流](../knowledge-query/references/query-workflows.md)。
