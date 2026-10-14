# CLI

需要组合过滤条件、解释输出或定位知识根时读取本文件。

## 知识根

Query 按 `--knowledge-root`、`CANNBOT_KNOWLEDGE_ROOT`、当前目录祖先中的 `.cannbot/knowledge.env`、仓内脚本位置依次解析知识仓。全局参数可以放在子命令前后；配置文件同时接受 `export CANNBOT_KNOWLEDGE_ROOT=...` 和不带 `export` 的形式。

这是数据根解析规则，不是独立 Skill 打包接口。完整安装使用仓库 `init.sh` 保留到规范源的链接，脚本由此可找到同一 checkout 的 `governance/contracts/`。单独复制 Query 目录可能仍能做普通索引搜索，但索引构建、完整校验和依赖 Contract 的 API 阅读等流程不完整，不能据此宣称已安装可用的全部能力。

## 索引

项目安装器默认先恢复文档与固定提交源码，再验证并按需构建 `artifacts/indexes/knowledge.sqlite3`。知识仓维护者在卡片、Registry、Profile、Schema 或 Contract 变化后执行：

```bash
python3 <skill-dir>/scripts/knowledge_index.py --knowledge-root <repo> build
python3 <skill-dir>/scripts/knowledge_index.py --knowledge-root <repo> verify --level full
```

Query 只读索引；索引构建通过统一 Contract 拒绝重复 YAML key 等不合法卡片。普通调用执行 quick stale check，写 search receipt 和校验 retrieval state 时还会把 SQLite document/posting 行逐项绑定到知识卡，并继续使用已校验的同一只读连接。缺失、损坏、卡片集合/时间戳变化、治理指纹变化或查询表不一致都失败关闭，不退回逐文件搜索。

## 本地来源

阅读本地证据时先区分路径基准，不把所有相对路径都拼在同一个根下：

| 位置 / 字段 | 路径基准与读取方式 |
|---|---|
| 搜索结果 `path` / `local_path`、`title`、`description` | 先据此选择相关候选；`local_path` 已是绝对路径，选中后打开并阅读全文 |
| 卡片正文中的本地 Markdown / 源码链接 | 普通相对路径以当前卡目录解析；前导 `/` 的知识链接以 Bundle 根解析；`#锚点` 指向当前卡。源码仍须满足登记的固定提交来源规则 |
| Frontmatter `sources[].resource` 的本地路径 | 知识仓根；按 Registry 核对归档或仓库、commit，不相对当前工程解析 |
| Ascend C API 的 `source_headers` | CANN 包内对应的 Ascend C 组件根目录；值已带 `include/`。Agent 决定需要查看时再拼接绝对路径并读取 |

选中卡片后阅读全文，再决定是否查看其相关内容或 include。知识库链接是有效入口，不需要另跑链接检查；`index.md` 只是导航，Agent 可继续沿其中的链接选卡。链接有效不等于内容适用于当前问题，仍需结合平台、生命周期和版本判断。

`sources[].resource` 中的登记路径相对于知识仓根（不是调用方工程或卡片目录）；四个源码仓统一使用 `cann-ops-raw/ascendc/<仓库名>/<40位commit>/...`。仓库 Registry 保存远程 repository 与允许的 revision；需要远程核对时可还原为 `<repository>/blob/<commit>/<path>`。API 的“源码中使用此 API 的算子”区块链接对应 Operator 卡和源码行，超出自动展示上限标注“部分算子”；仅有正文证据的示例单独标注。两者都是导航，不替代读取源码。

这里的知识仓根是 `discover` 输出的 `root`，包含 `governance/` 与 `knowledge/`；`knowledge/` 本身是 Bundle root，不是源码路径基准。Ascend C API 流程使用其[路径工具](../../ascendc-api-knowledge-query/SKILL.md)解析选定卡片的原始 source 或相对链接，再读取绝对路径；工具复用同一根解析与 Contract，不扫描替代文件。

资源未恢复时，Query 不自动联网或退回扫描源码。向用户说明缺口，获准后运行仓库登记的资源获取入口；安装器支持显式离线模式，但不能声称已核对不可用源码。索引只收录知识卡正文，不收录源码树；普通卡片链接与正文一起参与检索。

## Discover

```bash
python3 <skill-dir>/scripts/knowledge_query.py --knowledge-root <repo> discover [--json]
```

输出实际 Concept 总数及 domain、technology/scope、OKF type、路径 Profile 和 status 计数。结构占位目录和只有 Registry 的分类不计入覆盖。

这些是各维度的独立计数，不是平台/技术/Profile 的交叉覆盖矩阵。同一知识根、同一任务中可复用已有 discover；后续 search 仍会检查索引有效性，语料或治理指纹变化后再刷新覆盖判断。

## Search

```bash
python3 <skill-dir>/scripts/knowledge_query.py --knowledge-root <repo> search \
  --query <text> [--domain ops] [--technology ascendc] [--type apis] \
  [--task implementation] [--tag topic.data_movement] \
  --platform a2 [--include-shared] \
  [--source-technology triton] [--target-technology ascendc] \
  [--status stable|draft|deprecated|all] [--match text|literal] \
  [--decision <id>] [-k 10] [--json|--text] [--receipt <new-json>]
```

`--type` 过滤路径 Profile ID（如 `apis`），输出中的 `type` 是 OKF type（如 `API`）。`-k`、`--k`、`--limit` 等价。Search 默认输出 JSON；`--text` 供 Agent 或人工阅读精简候选：带本地 Markdown 链接的标题、路径、摘要和行数，不改变过滤与排序。普通问答选卡后阅读全文，采用该卡时可直接引用标题链接；状态、平台和复核事件回到正文核对。评分、信任层级及正式审计仍使用完整 JSON/receipt，不把文本输出当作审计载荷。

安装器可把项目缺省平台写入 `.cannbot/knowledge.env` 的 `CANNBOT_KNOWLEDGE_PLATFORM`，也可直接在进程环境中设置同名变量。普通 search 省略 `--platform` 时使用该值；显式 `--platform` 优先。`discover`、平台识别查询和 `--include-unlocated` 不使用该回退。没有缺省平台时，普通 search 仍要求显式 `--platform`。

### 过滤与匹配

- 重复的 tag 和 platform 使用 AND。
- task 自动规范为 `task.*` 并校验 task 维度。
- domain、route、platform、tag 和跨技术端点必须注册；Profile 必须受支持。
- 普通搜索必须有目标 platform，可显式传 `--platform`，或由 `CANNBOT_KNOWLEDGE_PLATFORM` 提供缺省值。`--technology platforms` 可在映射产品平台时不传；未定位 draft 使用 `--include-unlocated --status draft|all`，且不能同时传 platform。
- `--include-shared` 只与具体 technology 一起使用，并只附加同领域 shared。
- Search 当前不提供 CANN 版本过滤；版本字段仍可随命中返回，具体适用边界由正文核对。
- `text` 对 title、description、tags/aliases 和正文加权；`literal` 要求完整查询串出现在这些字段中，并按精确标题、标题片段、摘要、正文的顺序降权。普通正文链接与其他正文一起进入索引，不按专用关系块标记排除内容。
- 默认 status 为 `stable`，limit 必须大于零。

### Receipt

`--receipt` 原子写入 schema、完整请求、canonical 请求指纹、知识仓状态、索引版本、语料/治理指纹、命中卡片指纹和原始结果。目标目录项必须不存在；已有文件、目录和符号链接（包括悬空链接）均拒绝，不沿目标链接创建其他文件。这一约定同时适用于 search、state 和 outcome receipt。审计流程的 `--decision` 使用稳定 ID，并把文件放在调用方实验目录。State 校验会重新计算请求指纹，要求每条 receipt 的平台集合与 state 目标完全相同，核对完整当前索引 provenance，并按原请求重放过滤、评分、排序和命中；ready receipt 绑定每个 search receipt 的文件 SHA-256，outcome 校验会再次确认这些文件未变化。

每个命中包含 rank、score、Concept ID、绝对 `local_path`、正文行数、OKF type、Profile、trust tier、范围、生命周期、卡片 SHA-256 及最多两个 source。顶层 `results` 是规范字段，`hits` 是迁移兼容别名。退出码 `0` 包括零命中；非法过滤、索引陈旧、知识根不可用或 receipt 已存在时退出码为 `2`。失败时同一结构化 error JSON 同时写入 stdout 和 stderr，便于管道消费方发现参数或索引错误。

`count` 仅表示本次返回的结果数，受 limit 限制；CLI 不返回全量匹配数。达到 limit 且目标定义仍未出现时，可扩大 k 或按问题细化，不能把前 k 项说成全部 API/全部调用者。

## 审计工具

两个审计入口共用 `scripts/retrieval/root.py` 的 `Audit`、JSON 对象读取和非空字符串校验；路径要求、错误结构和拒绝覆盖语义保持一致。

冻结 `cannbot-retrieval-state.v1` 后校验 receipt、decision、当前索引和证据卡片，并只在全部 decision 为 `covered` 时生成 ready receipt：

```bash
python3 <skill-dir>/scripts/retrieval_state.py <absolute-state.json> \
  --knowledge-root <repo> --require-ready --receipt <new-ready.json>
```

工程验证结束后创建 `cannbot-retrieval-outcome.v1` manifest，绑定 state、ready receipt、每个 decision 的 outcome 与实验产物 SHA-256：

```bash
python3 <skill-dir>/scripts/retrieval_outcome.py <absolute-outcome.json> \
  --require-complete --receipt <new-outcome-receipt.json>
```

完整 state/outcome 形状与停止条件见[查询与证据工作流](query-workflows.md)。
