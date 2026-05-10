# 文档索引（docs/）

整合后按"用途"分为 5 个目录。每个目录下按子主题进一步细分。

| 目录 | 用途 | 包含文档 |
|---|---|---|
| [01-架构与设计](01-架构与设计/) | 跨变体的架构总览、与 hil-serl-sim 上游的设计对齐、超参差异表 | architecture-overview / sim_architecture_alignment / 参数对比 |
| [02-改进记录](02-改进记录/) | 相对上游 baseline 的所有修改项与原因（V1 → V2.0 → V2.1） | CHANGES |
| [03-部署与训练指南](03-部署与训练指南/) | 端到端部署、训练剧本、Touch 触觉干预的操作手册 | DEPLOY_GUIDE / 实验合作内容 / TOUCH_INTERVENTION_GUIDE |
| [04-远程算力与传输](04-远程算力与传输/) | 远程 GPU 选型 / 云平台对比 / 国内算力机的代码加速拉取 | 算力与远程训练方案 / 远程算力机加速拉取方案 |
| [05-结果分析](05-结果分析/) | 上游/复现实验跑出来的实测数据与诊断结论 | hil-serl-sim 复现结果分析 |

## 文档清单

### 01-架构与设计

- [architecture-overview.md](01-架构与设计/architecture-overview.md) — 四个变体（V1/V2 × PyTorch/JAX）的 2×2 矩阵、不变量与差异
- [sim_architecture_alignment.md](01-架构与设计/sim_architecture_alignment.md) — 我们 PyTorch 实现与 hil-serl-sim 在 adopted / upgraded / kept 三个层面的对齐说明
- [参数对比_V1_V2_vs_hil-serl-sim.md](01-架构与设计/参数对比_V1_V2_vs_hil-serl-sim.md) — V1 / V2 / 上游 sim 的逐参数差异表（SAC、视觉、训练循环等）

### 02-改进记录

- [CHANGES.md](02-改进记录/CHANGES.md) — 训练不收敛 7 项根因分析 + 各项修复 + V2.0/V2.1 hybrid reward 重构（§8、§8b）

### 03-部署与训练指南

- [DEPLOY_GUIDE.md](03-部署与训练指南/DEPLOY_GUIDE.md) — Ubuntu 22.04 + RTX 3060 + Geomagic Touch 全流程：环境、OpenHaptics、数据采集、训练、监控、调参、FAQ
- [实验合作内容.md](03-部署与训练指南/实验合作内容.md) — A100 80GB 协作训练剧本（重型版：REDQ-10 + UTD=8 + 解冻编码器 + wandb）
- [TOUCH_INTERVENTION_GUIDE.md](03-部署与训练指南/TOUCH_INTERVENTION_GUIDE.md) — Touch 设备按键定义与三阶段干预策略

### 04-远程算力与传输

- [hil-serl-sim_算力与远程训练方案.md](04-远程算力与传输/hil-serl-sim_算力与远程训练方案.md) — 显存解构、GPU 选型、云平台对比、远程训练完整工作流
- [远程算力机加速拉取方案.md](04-远程算力与传输/远程算力机加速拉取方案.md) — 国内服务器拉 GitHub 的 5 种加速方案（ghproxy / Gitee / rsync / OSS / proxychains）

### 05-结果分析

- [hil-serl-sim_复现结果分析.md](05-结果分析/hil-serl-sim_复现结果分析.md) — 协作者 AutoDL 上跑完 hil-serl-sim 的诊断报告（220K 步只到 10% 成功率）
