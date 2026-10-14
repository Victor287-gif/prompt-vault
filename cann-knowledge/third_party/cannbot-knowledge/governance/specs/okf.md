# OKF Bundle 与 CANNBot Profile

本仓使用 Google [Open Knowledge Format（OKF）v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/fe3268a70e8ca5110a43a8f1dfdf6d1a458cf79f/okf/SPEC.md) 表达和交换知识，并在其上增加面向 CANN 工程知识的严格 Profile。当前基线固定到上游 commit `fe3268a70e8ca5110a43a8f1dfdf6d1a458cf79f`。

OKF 规定通用交换格式；CANNBot Profile 规定本仓允许哪些路径、字段和值，以及 Query、Ingest 和 Lint 如何使用它们。Profile 可以比 OKF 更严格，但不能改变 OKF 同名字段的语义。

## 术语

| 术语 | 本仓含义 |
|---|---|
| OKF Concept | `knowledge/` 中任意一张非 `index.md` 知识卡，与它的具体 `type` 无关 |
| OKF `type` | Frontmatter 中供 OKF consumer 识别的类型；`Concept` 只是其中一个取值 |
| 路径 Profile | `concepts/`、`apis/`、`runbooks/` 等目录分类，负责选择 OKF `type` 和字段组合 |
| CANNBot Profile | 本仓在 OKF 之上增加的整套路径、字段、受控值和作者门禁 |
| 安装模式 | `consumer` 或 `contributor` 的 Skill 组合，不属于知识 Profile |

因此，`type: API` 的卡片仍然是一个 OKF Concept document；只有位于 `concepts/` 的卡片使用 `type: Concept`。

## 唯一 Bundle 边界

`knowledge/` 是本仓唯一的 OKF Knowledge Bundle 根目录，其下所有知识卡共享同一个 Concept ID 命名空间。

`common/`、`ops/`、`graph/`、`model/` 和 `runtime/` 只是 Bundle 内的知识分类，不是独立 Bundle，也不重复声明 `okf_version`。仓库根目录不是 Bundle root；`governance/`、`.agents/`、`evals/` 和 `logs/` 属于生产、消费与治理设施，其中的 Markdown 不作为 OKF Concept。

| 本仓对象 | OKF 对应 | 约定 |
|---|---|---|
| `knowledge/` | Knowledge Bundle | 唯一正文根，可作为一个整体浏览和分发 |
| 非 `index.md` Markdown | Concept document | YAML Frontmatter 加 Markdown 正文 |
| Bundle 内相对路径 | Concept ID | 去掉 `.md`，不带 `/` 或 `knowledge/` 前缀 |
| Markdown 链接 | Link | 支持相对路径，以及形如 `/ops/...` 的 Bundle 根绝对路径 |
| `index.md` | Index file | 用于逐层浏览，不参与知识搜索 |

例如 `knowledge/ops/ascendc/apis/datacopypad_gm_to_ub.md` 的 Concept ID 是 `ops/ascendc/apis/datacopypad_gm_to_ub`。

Concept 间的本地 Markdown 链接必须解析到同一 `knowledge/` Bundle；索引也只能导航 Bundle 内文件。知识卡可额外链接 Registry 登记的固定提交本地源码文件（可带 `#L行号`），这类链接只是来源导航，不是 Concept Link，也不生成源码节点或图谱边。

其他 Bundle 外材料使用 HTTP(S) 来源，不用仓库相对路径把 Concept 绑定到治理设施。行内代码和代码围栏中的 Markdown-like 片段不构成 Link；列表容器缩进后的围栏同样按代码处理。

正文图片不形成 Concept Link 或图谱边，必须使用带 40 位 commit 和具体文件路径的 HTTPS Git raw URL；Lint 仅离线校验 URL 形状。

`knowledge/` 本身可以按 OKF 格式浏览或分发；使用 CANNBot 的 Query、Ingest 和 Lint 时，还需要配套版本的 Skill 与仓库根目录 `governance/`。这些运行设施位于 Bundle 外，不改变 Bundle 边界。

如果未来需要按领域或 DSL 单独交付，应从这个规范源生成独立发布 Bundle，并处理 shared/common 依赖；不要在当前知识树内声明互相嵌套的 Bundle。

## OKF 对应关系

| OKF v0.2 语义 | CANNBot Profile 约束 |
|---|---|
| Concept 必须有非空 `type` | 路径 Profile 为每类知识确定唯一 `type`，Contract 检查两者一致 |
| `title`、`description`、`tags` 可选 | 为保证检索和可读性，本仓全部设为必填 |
| `sources[].resource` 表达 provenance | 保留字段名和语义，接受 HTTP(S) 或安全的仓库相对资源，并允许可选 `role` 扩展 |
| 生命周期使用 `draft`、`stable`、`deprecated` | 使用相同值，并要求每张卡显式填写 |
| `verified` 记录 actor 和时间 | 支持一个或多个事件，Query 据此派生信任层级 |
| Concept ID 是 Bundle 内路径 | Query、图谱和 `replaced_by` 使用同一种 Bundle 相对 ID |
| 根索引可声明 `okf_version` | 只有 `knowledge/index.md` 声明 `okf_version: "0.2"` |
| Index 可选 | 本仓要求每个实际目录都有只列直接子项的索引和摘要 |

`created_at`、`updated_at`、`platforms`、`cann_versions`、`source_headers`、`aliases`、`source_technology`、`target_technology`、`replaced_by` 和 `sources[].role` 是 CANNBot Profile 扩展。它们不重定义 OKF 核心字段，并且都有明确消费者。其中 `cann_versions` 仅用于非 Runbook 卡，Runbook 不声明版本字段（版本边界写入正文失效边界）。

本仓暂不使用 OKF 的 `generated`、`stale_after`、Bundle 内 `log.md`、`references/` 和 Attested Computation。没有可靠事实时不通过 Git 时间或经验规则补造这些字段。

## 严格作者门禁

OKF 通用 consumer 应容忍未知 `type`、扩展字段、断链和缺失索引。本仓工具只面向 CANNBot Profile：

- Ingest 只生成 CANNBot Profile 接受的候选 Concept；
- Query 在加载时拒绝不符合 CANNBot Profile 的卡片，不承诺读取任意第三方 OKF Bundle；
- Lint 检查字段白名单、受控路径、索引完整性和链接可达性，严格程度高于 OKF 最小要求；
- 本地图谱按需从 Concept 之间的普通 Markdown 链接生成，仅包含 Concept 节点和有向 `links_to` 边，包括 API 与算子卡片之间的链接；Index 只用于导航。构图不依赖关系判定缓存，不维护专用关系块，不回写 Concept，也不是入库门禁。图谱和可视化全部位于 Git 忽略的 `artifacts/graphs/`，不提交。

这些限制是作者质量门禁，不应被描述成 OKF 对所有 consumer 的要求。

## 规范演进

变更 OKF 核心字段、Concept ID、根索引或链接解析时，必须同时更新本仓机器声明、Contract、受影响的 Skill、知识卡和 conformance 回归。

OKF 上游仍在演进。本仓只对上述固定 commit 做过审核；更新参考版本时必须重新检查 required 字段、保留文件、provenance、trust、lifecycle、link 和 consumer conformance，不能静默跟随 `main`。
