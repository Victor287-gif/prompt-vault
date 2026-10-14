# CANN 算子研究知识库 + 只读 MCP

基于 **CANNBot 官方两个仓库的固定版本快照**，为 ChatGPT、DSH、Codex 提供共享的算子研究资料。第三方原文与许可证保留；本封装非官方产品。

## 现在怎样使用

在 ChatGPT 选择配套 **CANN 算子研究知识库 / cann-operator-research** Skill，直接描述问题。已连接 GitHub 时，Skill 通过 GitHub MCP 读取此分支的主题索引与原文，不要求本机在线，也不要求先部署新服务器。

仓库：`Victor287-gif/prompt-vault`；分支：`cann-knowledge-v1`；包根：`cann-knowledge/`。
**显式指定该分支，不要让读取回落到 main。** 原 Prompt Vault 的 main 未被修改。

## 包含什么

- `third_party/`：算子开发、架构、API、性能、流水、内存、同步、tiling、精度、profiling、仿真和参考代码；也包括 Triton/TileLang/PyPTO/Catlass 相关材料。
- `catalog/INDEX.md`：GitHub MCP 可逐层读取的主题索引，每页最多 70 条。
- `registry.jsonl`：每份可检索文本的来源、固定 commit、路径、行数、哈希与主题。不要一次塞入上下文。
- `sources.lock.json`：上游固定版本和选编范围；`excluded-files.json`：排除项与原因。
- `cann_knowledge/`：只读检索 MCP 与 CLI；支持中文子词和英文术语检索。
- `index/knowledge.sqlite3`：本地生成的检索数据库，不随 Git 或 ZIP 分发；用构建脚本从所附原文重建。
- `test-report.json`：本次 CPU 检索与 MCP 协议测试，不是 NPU 性能报告。
- `plugin/`：可安装的单一 Skill 包，不直接启用上游全部 SKILL.md。

## 来源和范围

两个仓库从官方 GitCode 地址获取，精确 commit 见 sources.lock.json。收录 ops/、共同基础知识和相关工具；保留 community/contrib 身份。模型端到端调优、图模式和一般 Runtime 项目不作为本包主范围。

未收录的字体、可执行二进制、依赖环境、测试目录、过大文件及部分二进制文档见排除清单。图片可能随原文保留，但不做 OCR，也不宣称全部图文已被模型审阅。正文的外部链接可能需要另外取原始资料。

**官方来源不等于每条内容适用于 910C/CANN 9.0。** 架构字段是文中提及，不是自动验证的支持矩阵。引用前核对 API、芯片、SDK、形状和资源约束。

## 专用 MCP（可选部署）

Python 3.10+。建议使用独立虚拟环境；不需要安装 CANN 或访问 NPU。

```bash
python -m venv .venv
# 激活本项目虚拟环境后：
python -m pip install -r requirements.txt
# 首次部署时从所附资料构建检索索引：
python -m cann_knowledge.build_index --root .
python -m cann_knowledge.server --root . --transport stdio
```

HTTP 模式：

```bash
python -m cann_knowledge.server --root . --transport streamable-http --port 8765
```

只监听 `127.0.0.1`。这不是公网服务；网页接入需另行授权的认证隧道/网关。不要把示例 localhost 填作已部署的 ChatGPT 远端地址。

详细配置：[docs/MCP_SETUP.md](docs/MCP_SETUP.md)。

## 本地查询（可复用现有终端 MCP）

```bash
python -m cann_knowledge.cli search "矩阵乘 L1 常驻" --topic matmul
python -m cann_knowledge.cli search "标量 scalar" --topic sync-scalar
python -m cann_knowledge.cli read "从 search 返回的真实 doc_id" --start-line 1 --max-lines 100
python tests/smoke_test.py
```

这些接口只读。构建索引是独立显式操作；服务不提供任意 Shell、文件修改、编译、NPU 实验、安装或 OJ 提交工具。

## 更新

重新获取官方仓库后，使用 extraction_policy.py 的 --source-root 与 --output 参数生成**新的输出目录**，检查新增来源和排除项，再构建索引、测试、发布新版本。不要直接覆盖正在使用的索引。

压缩包保留全部本次选编原文与源码，不附带 354 MB 的可重建 SQLite 索引；网页 GitHub MCP 查询无需此数据库。

快照不是实时同步。工程事实与知识分开：真实 Kernel、profiling 和 OJ 数据仍通过用户授权的工程 MCP 读取。

## Windows 路径

上游文件层级较深，请使用短解压路径，例如 `D:\ckb`，或使用支持长路径的解压工具。不要把仓库嵌套在非常长的目录中。

## 许可

阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 及两份上游 LICENSE。原始资料用途受 CANN 2.0 条款约束；原创适配器许可不覆盖第三方内容。

## 压缩包解压

使用普通 ZIP/Deflate；支持常见解压工具。也可执行：

```bash
python -m zipfile -e cann-operator-knowledge-1.0.0.zip D:/ckb
```

进入生成的 `D:/ckb/cann-knowledge/` 目录，再按前文构建索引并启动本地 MCP。选用网页版 Skill 不需要下载或构建。
