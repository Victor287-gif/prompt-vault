# cannbot-knowledge

CANN 领域知识库面向 [CANN](https://hiascend.com/software/cann) 社区，沉淀 AscendC/PyPTO/TileLang/Triton 等语言的算子开发、模型迁移、推理优化，以及图模式、Runtime 等开发场景中的可复用知识。知识库通过标准化知识卡、统一治理和检索能力，支持快速接入不同 Agent 框架。

本仓（cannbot-knowledge）是独立维护的 CANN 领域知识仓，提供知识卡、来源治理和检索能力，可作为不同 Agent 框架的公共知识底座。知识库架构如下：

![CANN 知识库架构](docs/figures/cann-knowledge-architecture.png)

本仓不把算子工程或模型代码作为知识正文提交，而是把经过整理、能够追溯到登记的文档来源、固定版本源码或卡内原始观测的工程知识保存成 Markdown 知识卡，例如：

- 某个 CANN API 的语义和使用限制；
- 某种算子 DSL 的编程模型；
- 一个算子的设计、实现或优化方法；
- 某类编译、运行、精度或性能问题的排查步骤。

各类 Agent 可通过本仓提供的 Skill 生产、检索和检查这些知识，从而在回答问题或执行工程任务时复用经过治理、可以追溯来源的结论。

## 设计参考

本仓知识正文的文件化设计参考 Google 的 [Open Knowledge Format（OKF）v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/fe3268a70e8ca5110a43a8f1dfdf6d1a458cf79f/okf/SPEC.md)。知识使用带 YAML Frontmatter 的 Markdown 表达，通过目录和 `index.md` 渐进浏览，通过普通 Markdown 链接形成可重建关系，并使用 Git 分发和审计。

`knowledge/` 是本仓唯一且符合 OKF v0.2 的 Knowledge Bundle 根目录，其下所有卡片共享同一个 Concept ID 命名空间。`common/`、`ops/` 等只是 Bundle 内的分类，不是独立 Bundle；仓库根目录的 Skill、治理、测试和日志设施不属于 Bundle。详细边界见 [OKF Bundle 与 CANNBot Profile](governance/specs/okf.md)。

## 这个仓库如何工作

```text
官方文档 / 固定版本源码 / 可复现实验
                    │
                    ▼
       Ingest 与专用 producer
                    │
                    ▼
               Markdown 知识卡
                 │        │
                 │        └── knowledge-lint 检查结构和治理规则
                 ▼
            knowledge-query
                    │
                    ▼
          CANNBot 读取正文并核对来源
```

本仓维护七项 Skill：六项可安装到业务项目，`ops-knowledge-trace-ingest` 只用于本知识仓内的 trace → Runbook 生产。

| Skill | 作用 | 是否修改知识仓 |
|---|---|---|
| `knowledge-query` | 查看实际知识覆盖，检索知识卡，并可保存不可覆盖的审计 receipt | 除指定 receipt 外只读 |
| `ascendc-api-knowledge-query` | 复用 Query 按硬件平台检索 Ascend C API，阅读全文并按需追踪相关卡片和本地 CANN 头文件 | 否 |
| `knowledge-ingest` | 统一路由新来源、增量、版本升级和通用建卡，支持按需链接图谱 | 创建候选卡；构图只读正文，产物始终由 Git 忽略 |
| `knowledge-lint` | 检查目录、Frontmatter、来源、正文、导航和日志；整理 Issue 材料 | 否 |
| `ops-knowledge-trace-ingest` | 从算子开发 trace 蒸馏 Runbook，并按当前 Schema、导航、Lint 和检索门禁收口 | 仓内使用；只写经审查的 Runbook、导航和日志 |
| `ops-knowledge-vv-ingest` | 从固定版本源码提取不含可达 Cube/AIC 或 MIX_CV 节点的完整 Operator 卡 | 只写任务指定的最终卡 |
| `ops-knowledge-cv-ingest` | 从固定版本源码提取含可达 Cube/AIC 或 MIX_CV 节点的完整 Operator 卡 | 只写任务指定的最终卡 |

## 知识如何组织

大多数知识卡使用下面的路径：

```text
knowledge/<领域>/<技术或作用域>/<路径 Profile>/<知识卡>.md
```

例如，当前 API 卡可以采用分层主题目录：

```text
knowledge/ops/ascendc/apis/utils_api/cpp_stdlib/math_functions/ceil_division.md
```

其中：

- `ops` 表示算子领域；
- `ascendc` 表示这条知识只适用于 Ascend C；
- `apis` 是路径 Profile，表示它是一张 API 知识卡；
- `utils_api/cpp_stdlib/math_functions` 是便于浏览的主题 section；
- bundle 内的 `ops/ascendc/apis/utils_api/cpp_stdlib/math_functions/ceil_division` 是 OKF Concept ID；移动卡片会改变该标识；
- Frontmatter `type: API` 是路径 Profile 的确定性投影，供 OKF consumer 读取。

不同 DSL 的专属知识分开保存。只有确实不依赖某个 DSL 的结论才能进入 `shared/`；两种技术之间哪些内容可以复用、哪些必须重新推导，则写入 `shared/interoperability/`。

Operator 卡按 `operators/<aclnn|direct>/<cv|math|nn|transformer>/<category>/<operator>.md` 组织；`aclnn` 保存公开 ACLNN/API 到 Kernel 的完整链路，`direct` 保存独立 Direct Invoke 路线，category 沿用 Registry 登记的官方源码一级分类。

每张知识卡的正文解释结论、适用条件和失效边界；文件开头的 Frontmatter 保存检索、平台、版本、生命周期和来源等结构化信息。详细规则见[治理规范](#治理规范)。

## 仓库目录

| 路径 | 用途 |
|---|---|
| `knowledge/` | 知识卡和逐层导航，是唯一知识正文目录 |
| `governance/schemas/` | 机器可读的字段、路径 Profile 和受控值规则 |
| `governance/specs/` | 对治理规则的完整文字说明 |
| `governance/contracts/` | Ingest、Query、Lint 和图谱共同使用的校验实现 |
| `.agents/skills/` | 三项基础 Skill、Ascend C API 专用消费流程、仓内 trace producer 与两项源码级 Operator producer 的规范源 |
| `docs/` | 知识库设计原则、安装与使用指南 |
| `recipes/` | 面向具体任务的知识库最佳实践 |
| `evals/` | 检索、治理、文档和安装回归 |
| `OAT.xml` | GitCode 开源扫描使用的 CANN 2.0 许可策略和 matcher |
| `install.sh` | 自动取得知识仓、补齐运行依赖、恢复资源、准备索引并把 Skill 安装到目标项目 |
| `check.sh` | 运行仓库完整质量检查 |
| `artifacts/` | 本地生成且可重建的产物，默认不提交 |
| `logs/` | 已实施的知识和治理变更记录 |

## 安装与使用

安装和使用方法见[安装与使用指南](docs/installation_and_usage.md)。

设计取舍见[设计原则](docs/design_principles.md)；常见任务见[形成可追溯回答](recipes/answer_with_evidence.md)、[贡献知识](recipes/contribute_knowledge.md)、[查询 API](recipes/api_search.md)和[使用知识库进行算子开发](recipes/operator_development.md)。

## 贡献和检查

贡献前阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，按贡献类型准备材料：

- **知识建设（缺陷修复除外）**：先提交 Issue，说明知识缺口、生产方法和预期效果；完成知识卡与效果验证后提交关联 PR，由 committer 检视后合入。
- **公共组件新增、优化或主动行为调整（如检索 Skill，缺陷修复除外）**：遵循“扩展优先、保持兼容”，先提交 RFC，说明当前问题、修改必要性和接入方案；主动改变既有行为时补充影响、迁移与回退说明。完成 A/B test 和原有路径/新增能力验证后提交关联 PR，由 committer 检视后合入。
- **缺陷修复与轻量修订**：知识错误、检索脚本 bug 等可直接提交 PR，无需前置 Issue/RFC 或完整 A/B 实验；提供修复依据与定向验证结果，满足适用检查并由 committer 检视后合入。纯排版、错别字和不改变组件行为的说明文档修订也适用；包含新增能力或主动流程调整时应拆分贡献。

贡献指南提供可填写的 Issue、RFC、PR 正文，以及 A/B 实验和合入要求；写卡操作可参照[贡献知识](recipes/contribute_knowledge.md)。

提交前运行：

```bash
bash check.sh
```

该入口执行知识 Lint、导航检查、索引与检索回归、单元测试、Ruff、Shell 语法及归档镜像差异检查，不主动获取或严格复验滚动官方来源。图谱和 API 引用建议按需生成，不属于入库门禁。

## 治理规范

- [OKF Bundle 与 CANNBot Profile](governance/specs/okf.md)：Bundle 边界、OKF 对应关系和严格作者门禁。
- [知识路径](governance/specs/paths.md)：domain、technology/scope、路径 Profile、Concept ID 和索引规则。
- [Frontmatter 与证据](governance/specs/frontmatter.md)：字段组合、生命周期、来源、标签、平台和版本。
- [Schema、Registry 与 Contract](governance/specs/schemas.md)：机器事实源、受控词表和规则执行关系。
- [Knowledge Skills](governance/specs/skills.md)：Query、通用 Ingest、仓内 trace producer、Operator producer、Lint 和安装模式。

本仓采用 [CANN Open Software License Agreement 2.0](LICENSE)。
