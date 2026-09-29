# Prompt Vault

一个自带 12 个中文 Prompt 的 Skill 插件。安装插件后，在 ChatGPT 网页端或 Codex 中说“打开 Prompt 工具箱”，即可浏览、挑选并直接使用；日常调用不需要 GitHub 连接器或 MCP。

## 插件内容

```text
plugin.json
skills/prompt-vault/
├── SKILL.md
├── agents/openai.yaml
└── references/
    ├── index.yaml
    └── prompts/                 # 12 个独立模板
```

目录为每个模板说明适用场景、处理方式、产出和例子。浏览时只读取目录，选定后才读取对应正文。多轮模板按步骤逐问推进。

## 使用

- “打开 Prompt 工具箱，让我挑一个。”
- “推荐一个适合我当前问题的 Prompt。”
- “用第 9 个模板分析我的两个选择。”
- “把第 2 个 Prompt 原文给我复制。”

GitHub 保存插件源码。已安装的插件不会因为仓库更新而自动更新，需要发布新版本。Codex 也可单独安装 `skills/prompt-vault` 目录。

Prompt 整理自《12个最常用Prompt（via 数字生命卡兹克）》PDF，做了文字规范化和 Markdown 排版。原始 PDF 留在本地，没有上传到公开仓库。

模板涉及事实核查或实时研究时，是否能联网取决于当前对话可用的工具；没有检索能力时会说明限制。
