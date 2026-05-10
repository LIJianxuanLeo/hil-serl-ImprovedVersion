# CLAUDE.md — 本仓库的 Claude 协作约定

> 这是给 Claude Code 看的项目级备忘。仓库整体说明请看 [`README.md`](README.md)，
> 技术文档索引见 [`docs/README.md`](docs/README.md)。

## 工作流规则

### 改完直接推 GitHub

完成用户布置的整段任务后，**自动 `git add` → `git commit` → `git push origin <当前分支>`**，
不需要再问一次"要不要推"。

- 适用范围：用户对仓库内容（代码 / 配置 / 文档）的任何修改任务，且涉及多文件改动时
- commit message 用一句中文概括"做了什么 + 为什么"，按本仓库已有风格
- 推送前先 `git status` 确认没把秘密文件、`academic/`、`.DS_Store`、临时产物带进去
- 推送目标默认 `origin` + 当前分支（一般是 `main`）；若遇到非 fast-forward，先 `git pull --rebase` 再推
- 仅在以下情况停下来确认：
  - 出现冲突 / pre-commit hook 失败
  - 工作树有用户未提及的本地修改
  - 需要 force-push、删除分支、改写历史等破坏性操作

### 文档组织

`docs/` 下按 `01-架构与设计 / 02-改进记录 / 03-部署与训练指南 / 04-远程算力与传输 / 05-结果分析`
五个目录分组。新增文档先归入对应类别；如开新类别，同步更新
[`docs/README.md`](docs/README.md) 索引和根 README 的"Where to look"表。
