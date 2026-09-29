# Prompt Vault

这个 GitHub 仓库是 12 个 Prompt 的统一来源。原始材料是《12个最常用Prompt（via 数字生命卡兹克）》PDF；`prompts/` 中的 Markdown 做了文字规范化和排版，内容以原 PDF 为准。

## 仓库结构

```text
index.yaml                    # 目录：编号、用途、文件路径
prompts/                      # 12 个独立的 Prompt 正文
skills/prompt-vault/SKILL.md  # 浏览、选择、调用规则
```

## 在 ChatGPT 中使用

在 ChatGPT 的 Apps / Plugins 中连接 GitHub，并授权访问本仓库。然后在支持 GitHub 的对话中说：

> 请读取 `Victor287-gif/prompt-vault` 的 `index.yaml`，列出 Prompt 供我选择；我选定后，再读取对应的 `prompts/*.md` 并按原文执行。

也可以直接指定：“从 `Victor287-gif/prompt-vault` 读取 `prompts/09-steelman.md`，结合当前对话使用。”ChatGPT 的 GitHub 连接是按需读取仓库内容；普通聊天是否能使用 GitHub，取决于账号和产品界面。仓库里的 `SKILL.md` 不会因为连接 GitHub 就自动安装成 ChatGPT Skill。

## 在 Codex 中使用

把仓库 clone 到本机后，在该仓库工作区中说“读取 `index.yaml`，打开 Prompt 工具箱”或“使用第 9 个 Prompt”。Codex 可以直接读取本地文件，不需要自建 MCP。若想在任意工作区通过 `$prompt-vault` 调用，可以另行安装 `skills/prompt-vault`，并让它能够访问这份仓库。

## 设计边界

这版用 GitHub 管理和分发内容，不需要自建 MCP。GitHub 仓库提供读取与版本管理；ChatGPT 中的“选一个就用”由对话完成，不会自动生成卡片按钮。如果以后确实需要可点击的选择器，再单独开发界面。

