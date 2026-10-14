# 知识路径

路径表达知识属于哪个领域、适用于哪种技术或作用域，以及采用哪种路径 Profile。它同时生成 OKF Concept ID；domain、technology/scope 和 Concept ID 不在 Frontmatter 重复保存。

## 基本形态

知识卡只采用两种路径：

```text
knowledge/<domain>/<technology-or-scope>/<profile>/[<section>/...]<slug>.md
knowledge/<domain>/<direct-profile>/[<section>/...]<slug>.md
```

第一种经过 technology 或 scope 路由；第二种只用于 Registry 明确允许由领域直接承载的 Profile。算子知识必须声明 DSL 或使用 `shared`，不能省略这一层。

- Concept ID 是路径相对 `knowledge/` 去掉 `.md` 后的值，例如 `ops/ascendc/apis/datacopypad_gm_to_ub`。
- Concept ID 不带开头 `/`、`knowledge/` 前缀或 `.md` 后缀。移动卡片会改变 ID，必须同步正文链接和 `replaced_by`。
- 文件名使用小写 `snake_case`；集合型目录使用复数。已注册技术 ID、scope 和行业缩写保持注册形式。
- 路径变化由 Git 历史审计，不另设旧路径映射。

## Domain 与 route

route 是 technology 或 scope 的统称。当前允许的组合如下：

| Domain | Technology/scope route | Direct profile |
|---|---|---|
| `common` | `platforms`, `toolchains`, `observability` | `glossaries` |
| `contrib` | `ascendc` | 无 |
| `ops` | `ascendc`, `triton`, `tilelang`, `pypto`, `shared` | 无 |
| `graph` | `torch_compile`, `ge`, `shared` | 无 |
| `model` | `inference`, `training`, `spatial`, `embodied` | 无 |
| `runtime` | 无 | `concepts`, `apis`, `runbooks` |

route 后必须出现路径 Profile，例如 `knowledge/common/platforms/concepts/`。`interoperability` 只能放在同一 domain 的 `shared/` 下。

Registry 允许某种组合，只表示工具能够识别它，不表示仓库已有相应知识。Contract 要求 Registry 中注册的所有 domain 根目录始终存在；设置 `precreate_routes: true` 的 `ops` 和 `model` 还预建各自注册 route。

`common`、`contrib`、`graph`、`runtime` 不预建没有真实知识的下一层路径，由领域维护者随首张卡物化。所有 index-only 入口都不计入 Query 覆盖，route 下的具体 Profile 也不全量预建。

当前路径 Profile 是全局集合：经过 route 的路径默认可以选择任一已注册 Profile，但内容必须符合该 Profile 的职责。机器规则另行限制 `interoperability` 只能位于 `shared`，并限制 Operator 与 Runbook 的附加层级。若以后需要禁止某个 domain-route 与 Profile 的组合，应在 Registry、Contract 和接受/拒绝测试中明确增加规则，不能只写在文档里。

## 路径 Profile

| Profile | 保存内容 | 不保存 |
|---|---|---|
| `concepts/` | 稳定概念、职责边界、执行模型、编程模型 | 单个 API 参数表、故障处置步骤 |
| `apis/` | 接口签名、语义、约束和适用范围 | 完整算子实现 |
| `examples/` | 能说明输入、输出和验证方式的最小样例 | 缺少上下文的代码片段 |
| `guides/` | 围绕一个开发任务组织的原理、步骤、约束和验证入口 | 单个 API 参数表或按症状处置的 Runbook |
| `operators/` | 完整算子的语义、设计、Tiling、Kernel 和工程职责 | 跨算子的通用原理 |
| `optimizations/` | 可复用的优化方法、触发条件、收益来源和失效边界 | 未归因的单次性能数据 |
| `runbooks/` | 按症状或任务组织的诊断、验证和处置步骤 | 只有背景概念、没有操作闭环的说明 |
| `interoperability/` | 两种技术间可复用和必须重推导的映射 | 单一技术的实现细节 |
| `glossaries/` | 术语、稳定定义和检索别名 | 教程、API 手册或经验步骤 |

路径 Profile 和 OKF `type` 是两个层次：前者组织仓库并选择字段组合，后者保存在 Frontmatter 供 OKF consumer 使用。两者的唯一映射见 [Frontmatter](frontmatter.md)。

不单设 `patterns/`：原理进入 `concepts/`，完整算子进入 `operators/`，性能方法进入 `optimizations/`，处置步骤进入 `runbooks/`。版本差异由适用范围和生命周期表达，不建立 `migrations/` Profile。

## Profile 附加层级

`concepts`、`apis`、`examples`、`guides` 和 `optimizations` 可以按真实主题增加任意深度的小写 `snake_case` section，用来保持大型官方参考集可浏览。section 只是导航，不是新的 domain、route 或 Profile，也不进入 Frontmatter；每一级都必须有完整的 `index.md`。禁止使用来源批次、日期或发布版本作为 section。

Operator 和 Runbook 的附加层级仍是受控值：

```text
knowledge/ops/<technology>/operators/<call-mode>/<family>/<category>/<operator>.md
knowledge/<domain>/<technology-or-scope>/runbooks/<area>/<runbook>.md
knowledge/runtime/runbooks/<area>/<runbook>.md
```

- Operator `call-mode`：`aclnn` 保存公开 ACLNN/API 到 Host、Tiling、Kernel 的完整链路；`direct` 保存独立 Direct Invoke 路线。
- Operator `family`：`cv`、`math`、`nn`、`transformer`，对应四个 CANN 官方算子仓。
- Operator `category`：按官方算子仓一级源码目录登记，且由 family 约束：

  | family | category |
  |---|---|
  | `cv` | `image`, `objdetect` |
  | `math` | `conversion`, `math`, `random` |
  | `nn` | `activation`, `conv`, `foreach`, `index`, `loss`, `matmul`, `norm`, `optim`, `pooling`, `quant`, `rnn`, `vfusion` |
  | `transformer` | `attention`, `ffn`, `gmm`, `mc2`, `mhc`, `moe`, `posembedding` |
- Runbook `area`：`compilation`、`runtime`、`precision`、`performance`、`validations`。

这些值来自 `registries.yaml` 的 `paths` 分区，不使用未注册的近义目录。

## Index 导航

- `index.md` 只存在于 `knowledge/`，不作为 Concept 参与检索。
- 只有 Bundle 根 `knowledge/index.md` 带 `okf_version: "0.2"` Frontmatter；其他索引不得带 Frontmatter。
- 每个目录都必须有索引，并完整、无重复地列出本层直接子目录和知识卡。
- 每个条目包含简短摘要，索引不能越级链接，也不得导航 Bundle 外的本地源码。知识卡中 Registry 登记的固定提交源码文件链接是来源导航例外，不建立源码 index 或 Concept。
- 普通知识目录的同步保留人工摘要；标题或 description 变更后需核对对应条目。导航覆盖检查不判断摘要是否仍准确表达卡片，不能把无缺项视为摘要已更新。
- Domain 根和 `ops`/`model` 的预建 route 允许只有说明文字的 index；其他目录必须包含合法直接卡或带合法 index 的直接子目录，普通占位文件不算知识内容。
- Index-only 入口只保证结构可浏览，不表示存在 Concept、来源覆盖或摄入 watermark。

## 示例

合法路径：

```text
knowledge/common/platforms/concepts/ai_core_execution_architecture.md
knowledge/ops/ascendc/operators/aclnn/nn/activation/gelu.md
knowledge/ops/ascendc/apis/basic_api/data_copy/datacopy.md
knowledge/ops/ascendc/guides/programming_guide/kernel_call.md
knowledge/ops/triton/runbooks/compilation/missing_symbol.md
knowledge/ops/shared/interoperability/triton_to_ascendc_tiling.md
knowledge/runtime/runbooks/runtime/device_reset.md
```

非法路径：

```text
knowledge/ops/apis/data_copy.md                     # 缺少 technology/scope
knowledge/common/platforms/chip.md                  # 缺少路径 Profile
knowledge/ops/ascendc/patterns/double_buffer.md     # 未注册 Profile
knowledge/ops/ascendc/runbooks/timeout.md           # 缺少 area
knowledge/ops/ascendc/glossaries/source/api.md       # Glossary 不允许附加 section
knowledge/graph/ascendc/concepts/kernel.md           # route 不属于 graph
```
