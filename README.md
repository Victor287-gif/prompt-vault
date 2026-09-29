# Prompt Vault

一个自带 12 个中文 Prompt 的 Skill。安装后说“打开 Prompt 工具箱”或调用 `$prompt-vault`，即可浏览、挑选并直接使用；使用时不需要 GitHub 连接。

## 内容

```text
skills/prompt-vault/
├── SKILL.md
├── agents/openai.yaml
└── references/
    ├── index.yaml
    └── prompts/                 # 12 个独立模板
```

Prompt 整理自《12个最常用Prompt（via 数字生命卡兹克）》PDF，做了文字规范化和 Markdown 排版。原始 PDF 留在本地，没有上传到公开仓库。

## 安装

- **Codex**：从本仓库的 `skills/prompt-vault` 安装 Skill。安装后即可在任意项目中使用 `$prompt-vault`。
- **ChatGPT**：如果当前账号或工作区提供 Skills，在 **Plugins → Skills → Create → Upload from your computer** 上传完整的 `prompt-vault` Skill 包。安装后即可在对话中直接调用，无需每次指定本仓库。ChatGPT 和 Codex 的安装分别管理。
- **更新**：GitHub 保存 Skill 源码；更新 Prompt 后，需要把新版 Skill 重新安装到对应产品。GitHub 不会自动同步已安装的 Skill。

## 使用示例

- “打开 Prompt 工具箱，让我挑一个。”
- “推荐一个适合我当前问题的 Prompt。”
- “用 Prompt Vault 的第 9 个模板分析我的选择。”
- “把第 2 个 Prompt 原文给我复制。”

浏览只读取目录，选定后才读取对应正文。多轮模板会逐问推进。

## 边界

Skill 是完整的浏览和调用入口。模板中要求事实核查、实时研究时，能否联网取决于当前对话可用的工具；没有检索能力时会说明限制。
