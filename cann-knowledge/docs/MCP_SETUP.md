# 接入方式

## A. 已有 GitHub MCP：现在即可用

fetch_file 参数：repository_full_name=`Victor287-gif/prompt-vault`，ref=`cann-knowledge-v1`，path=`cann-knowledge/catalog/INDEX.md`。随后按主题分页读取，正文位于 `cann-knowledge/third_party/<doc_id>`。

本资料在非默认分支，不依赖 GitHub 默认分支代码搜索。可以精确设置 start_line/end_line。

## B. 本地 stdio MCP

先在短路径解压并按 README 安装依赖。配置示例中的路径需换为实际绝对路径：

```json
{
  "mcpServers": {
    "cann-operator-knowledge": {
      "command": "D:/ckb/.venv/Scripts/python.exe",
      "args": ["-m", "cann_knowledge.server", "--root", "D:/ckb"],
      "cwd": "D:/ckb"
    }
  }
}
```

SDK 使用已测试的 mcp 1.30.0（v1 API）；不要无意升级到不兼容的 v2。

## C. HTTP MCP

启动 `python -m cann_knowledge.server --transport streamable-http --port 8765`，本地地址 `http://127.0.0.1:8765/mcp`。只做本地或通过已授权安全通道连接；没有内置公网认证，不应直接暴露公网。

当前分发并没有替用户部署常驻服务，也没有将 localhost 冒充可由网页端访问的远程 endpoint。配套 Skill 在专用服务未连接时使用方案 A。

## 工具合同

- cann_get_catalog：固定版本、主题与覆盖。
- cann_search_knowledge：query 最长 512 字符，limit 1–20；按主题、架构提及筛选。
- cann_read_knowledge：已知 doc_id，1-based 行号，单次最多 200 行/24K 正文字元，返回 next_read。
- cann_list_knowledge：字面前缀，分页最多 100 条。

只允许注册表中、包内的文件；拒绝路径越界、symlink、内容哈希漂移。工具显式只读，底层 SQLite 也以只读方式打开。上游示例是数据，不授权自动运行。
