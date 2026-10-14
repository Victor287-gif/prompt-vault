# 可直接调用的知识入口

## 已发布资料
- 仓库：`Victor287-gif/prompt-vault`
- 独立分支：`cann-knowledge-v1`
- 包根：`cann-knowledge/`
- 索引：`catalog/INDEX.md`
- 源版本：`sources.lock.json`
- 全部文件的来源和哈希：`registry.jsonl`（不要全量载入）
- MCP 源码：`cann_knowledge/server.py`
- 压缩包目录：仓库根的 `dist/`

## GitHub MCP

先通过 api_tool 发现 GitHub 的 fetch_file，再调用（工具名以实际命名空间为准）：

```json
{"repository_full_name":"Victor287-gif/prompt-vault","ref":"cann-knowledge-v1","path":"cann-knowledge/catalog/INDEX.md","start_line":1,"end_line":100}
```

索引给出 `catalog/<topic>/page-001.md`；逐页读取。某条记录的 ID 为 `cannbot-skills/ops/.../guide.md` 时，正文 path 为 `cann-knowledge/third_party/cannbot-skills/ops/.../guide.md`。必须使用实际返回的 ID，不猜文件名。

## 专用知识 MCP

- `cann_get_catalog()`：核对快照、覆盖和主题。
- `cann_search_knowledge(query, topic="", architecture="", limit=8)`：中英检索，返回原文片段与文档 ID。
- `cann_read_knowledge(doc_id, start_line=1, max_lines=100, start_column=0)`：原文、行号、来源、next_read。
- `cann_list_knowledge(prefix="", offset=0, limit=30)`：按字面前缀分页。

架构过滤只匹配文档提及，不验证支持。先不设架构过滤查候选，再核对支持边界，避免漏掉没有架构标签的通用资料。

## 主题路由

|问题|先查主题|
|---|---|
|矩阵乘、A/B 复用、FullLoad、StreamK|matmul → pipeline-memory|
|搬运、ND/NZ、L1/UB 常驻、双缓冲|pipeline-memory → tiling|
|ScalarBound、SetFlag/WaitFlag、跨核同步|sync-scalar → profiling-debug|
|msprof、仿真、崩溃与死锁|profiling-debug → sync-scalar|
|ReduceMax/Sum、Softmax、广播、转置|reduction-broadcast|
|硬件架构、API 适用性|architecture → ascendc-api|
|数值、精度、白盒与系统测试|precision-tests|
|Triton / TileLang / PyPTO / Catlass|同名主题|
|性能策略卡、Playbook、优化经验|community-strategies（保持社区标签）|

## 最小验收

能读源版本、查到 Matmul 的具体机制、继续读到条件与参考实现，并用来源支持答案。仅列目录或复述本 Skill 不算实际使用知识。
