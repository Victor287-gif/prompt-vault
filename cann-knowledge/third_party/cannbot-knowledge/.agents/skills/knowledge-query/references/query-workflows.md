# 查询与证据工作流

复杂任务使用本文件，将检索召回和工程判断分开。单一事实问答使用主 Skill 的轻量路径即可；这里不要求额外 Agent，也不授权执行构建或实验。

## 什么时候重入检索

首次领域决策前确认目标并查覆盖；后续复用已读卡和判型证据，只在出现新 API、错误信号、平台/版本变化或尚未解决的决策时重入。每轮说明要补的缺口、检索角度和新增证据；没有变化时不重复 discover、设备探测或同义改写查询。

## 审计查询

1. 建立目标边界：实体、platform、CANN version、工程形态、dtype/layout、workload 和验证边界。
2. 列出 decisions；每个 decision 只回答一个判断，例如参数单位、错误信号是否有已验证修复、性能瓶颈的主导机制。
3. 为每个 decision 运行窄查询，保存独立 receipt。互不依赖的 decision 可以并行，但 receipt 路径不能复用。
4. Agent 先根据脚本返回的路径、title 和 description 选择候选，再阅读全文；读完后决定是否沿链接查看相关卡、固定来源或 include，不要求读完所有命中或链接。记录事实、适用边界、negative evidence、conflict 和 gap。没有搜索命中不是 negative evidence；只有来源明确说明不支持、限制或失败，才能作为否定性技术证据。
5. 主任务将 decision 标为 `covered`、`conflicting`、`insufficient` 或 `gap`。只对未解决项补搜；补搜应从实体转向错误信号、从机制转向 API 契约或从现象转向验证过的修复路径。
6. 无新增证据或达到预算时停止。把 target、decision、receipt 和采用的卡片路径/指纹冻结为 `cannbot-retrieval-state.v1`；每条 receipt 必须使用与 target 完全相同的平台集合。全部 covered 后运行 `retrieval_state.py --require-ready --receipt`，它会核对完整索引 provenance 并重放过滤、评分、排序和命中；生成的 ready receipt 会绑定全部 search receipt 的 SHA-256。
7. 实现后的正确性、性能和平台实验写入 `cannbot-retrieval-outcome.v1`，绑定 state、ready receipt 和真实产物 SHA-256，再运行 `retrieval_outcome.py --require-complete --receipt`。不得修改原始 search receipt；任何改动都会使 outcome 链失效。

### 决策状态与停止

| 状态 | 证据情况 | 后续 |
|---|---|---|
| `covered` | 已读证据足以回答本决策，范围匹配且关键冲突已处理 | 可进入 ready 校验；不等于已做工程实验 |
| `conflicting` | 已读来源对同一范围给出不兼容结论 | 明确版本/重载/适用条件，未消除前不能 ready |
| `insufficient` | 有相关线索，但缺关键参数、范围或验证依据 | 只补缺失部分；停止时如实交付缺口 |
| `gap` | 当前决策没有可采用证据 | 保留有效零命中记录或具体阻塞，不制造 covered |

### 关联阅读如何进入审计链

- 沿链接读到的新卡不会自动进入原 search receipt。需要采用为 state 的 `evidence` 时，用该卡的准确标题/独特术语、适合的 Profile 和**同一目标平台集合、同一 decision ID**定向查询，保存新 receipt；实际命中该卡后才引用返回的 `path` 和 `card_sha256`。不能手填未出现在绑定 receipt 中的卡片，或修改旧 receipt 加入命中。
- 未定位 draft 补搜没有目标平台，与平台识别查询一样只作独立线索记录；不能绑定到有明确目标平台的 state 来生成 ready。无法获得符合目标边界的证据时，保持 insufficient/gap，不改写卡片或审计文件绕过校验。
- 不同平台的差异比较分别建立目标与证据链；共同适用查询使用完整的平台集合，每条 receipt 必须一致，不能把 A2 与 A3 单独结果拼成一份共同支持证明。
- 本机 SDK 头文件、固定源码与实验产物不是 Concept，不放入 state 的卡片 evidence。保留其真实路径、版本/commit、已读范围及它支撑的 decision；只有真实发生的工程验证才记录到 outcome。不要为记录“读过头文件”伪造 verified 实验。

已读内容无法闭合决策时，输出部分结论和缺口即可；不运行 ready 来“强行完成”，不把结构校验成功当作技术正确性证明。

最小 state：

```json
{
  "schema_version": "cannbot-retrieval-state.v1",
  "target": {"entity": "DataCopyPad", "platforms": ["a2"]},
  "decisions": [{
    "id": "api-contract",
    "question": "目标接口契约是什么？",
    "status": "covered",
    "receipts": ["/absolute/search.json"],
    "evidence": [{"path": "ops/ascendc/apis/example", "card_sha256": "<sha256>"}]
  }]
}
```

最小 outcome manifest：

```json
{
  "schema_version": "cannbot-retrieval-outcome.v1",
  "state": "/absolute/retrieval_state.json",
  "state_sha256": "<sha256>",
  "ready_receipt": "/absolute/retrieval_ready.json",
  "ready_receipt_sha256": "<sha256>",
  "events": [{
    "decision": "api-contract",
    "outcome": "verified",
    "summary": "目标平台验证通过",
    "observed": ["correctness passed"],
    "artifacts": [{"path": "/absolute/test.log", "sha256": "<sha256>"}]
  }]
}
```

## 输出建议

```markdown
# 结论
- <当前 decision 的结论；区分来源事实与工程判断>

## 证据
- `<card>#<anchor>`：<事实、revision 与适用边界>

## 冲突或缺口
- <只有存在时输出>

## 建议验证
- <知识证据不能替代的目标平台检查>
```

不要倾倒所有候选。正式交接可附 `cannbot-retrieval-review.v1`，其中 `state_sha256` 绑定 state、`verdict` 为 `approve|reject`；reviewer 只检查覆盖、冲突、边界、receipt 与引用文件一致性，不重新生成工程方案。
