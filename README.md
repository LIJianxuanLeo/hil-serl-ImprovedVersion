# HIL-SERL — Improved Version

> A consolidated workbench of four parallel HIL-SERL implementations
> (PyTorch / lerobot × JAX / serl_launcher, each in V1 sparse and
> V2 hybrid-shaping reward variants) plus the unified technical
> documentation, training results, and launch scripts that tie them
> together.
>
> 一个把四个并行的 HIL-SERL 实现（PyTorch/lerobot × JAX/serl_launcher，
> 每条线包含 V1 稀疏奖励与 V2 混合 shaping 奖励两个版本）汇总到单一
> 仓库的工程工作台，附带统一的技术文档、训练结果与启动脚本。

[English](#english) · [中文](#中文)

---

<a id="english"></a>
## English

### Why this repo exists

HIL-SERL ([Luo et al. 2024](https://arxiv.org/abs/2410.21845)) reports
near-100% success rates for sample-efficient real-robot manipulation. A
public reproduction the collaborator ran on AutoDL servers reached only
**10% success after 220 K steps / 3.83 hours of GPU time** — a clear
reproducibility gap. This repo holds the chain of fixes, ablations, and
tooling we built to close it.

The core split is reward-design × RL framework:

|  | **PyTorch / lerobot** | **JAX / serl_launcher** |
|---|---|---|
| **V1 (sparse reward)** | [`v1-pytorch/`](v1-pytorch) — our first improved fork | [`v1-jax/`](v1-jax) — upstream baseline (GeorgeAuburn/hilserl_sim_v1) |
| **V2 (hybrid sparse + dense shaping)** | [`v2-pytorch/`](v2-pytorch) — V2.1 reward, F7 logging | [`v2-jax/`](v2-jax) — JAX mirror of V2.1, byte-identical math |

The four cells are **deliberately byte-aligned** on the dimensions we
hold constant, so a comparison across any axis isolates exactly one
variable.

### Reward-design lineage

| | V1 sparse | V2.0 dense (deprecated) | **V2.1 hybrid (current)** |
|---|---|---|---|
| Per-step reward | 0 | 0–10 (`r_reach + r_grasp + r_lift`) | clip(`1·r_succ + 0.05·r_app + 0.1·r_grasp + 0.05·r_lift − 0.01·r_pen`, −1, 2) |
| Success bonus | +1 (override) | +10 (additive) | +1 (additive) |
| Approach | n/a | continuous `1/(1+5d)` | **differential** `clip(d_prev−d_now, ±0.02)` |
| Grasp | n/a | smooth ramp 8 → 1 cm | **event-based**, fires once per episode |
| Lift | n/a | linear `lift / 0.10` | **clipped** `clip(z − z_pick_ref, 0, 0.05)` |
| Episode reward range | {0, 1} | 15–60 | 0.0–0.6 (fail) / 1.2–1.8 (success) |
| Anti-farming guarantees | trivially safe | vulnerable to hover / light-touch / lift-farming | bounded by construction on all three |

The V2.0 → V2.1 refactor is the substantive engineering contribution:
the design comes from
`hilserl_抓取任务_稀疏奖励替换为稠密奖励_完整实现流程.docx`
(2026-05-04 engineering scheme), which we ported with minor grasp-event
and lift-clip refinements. Ablation guidance: see
[`docs/02-改进记录/CHANGES.md`](docs/02-改进记录/CHANGES.md) §8b.

### F6 + F7 logging

Both PyTorch variants ship a logger (`lerobot/src/lerobot/rl/training_logger.py`)
that emits, in addition to the standard SAC/REDQ metrics:

- **F6** (training): 32 columns including `q_target_mean/std`,
  `td_error_mean/std/max`, `critic_disagreement`, `policy_entropy_raw`,
  `actor_loss_q_term`, `actor_loss_entropy_term`, `step_time_ms`,
  `gpu_mem_peak_mb`, …
- **F6** (per-episode): `is_success`, `rolling_success_rate_50`,
  `rolling_intervention_rate_50`, `rolling_policy_only_success_20`,
  `episode_length`, `termination_reason`, …
- **F7** (V2.1 only — per-stage reward attribution):
  `r_sub_success / r_sub_approach / r_sub_grasp / r_sub_lift /
  r_sub_penalty / grasp_event_fired`. Lets the A/B/C ablation in
  scheme §11 distinguish "policy completed the task" from
  "policy ground out shaping reward".

All columns also stream to wandb (`wandb.enable=true` in the
`train_config_*.json`).

### Layout

```
hil-serl-ImprovedVersion/
├── README.md              ← this file (bilingual)
├── docs/                  ← all unified technical docs live here, grouped by purpose
│   ├── README.md                          ← docs index
│   ├── 01-架构与设计/                       ← architecture & design alignment
│   │   ├── architecture-overview.md           — 2x2 matrix navigation
│   │   ├── sim_architecture_alignment.md      — PyTorch sim ↔ JAX sim alignment
│   │   └── 参数对比_V1_V2_vs_hil-serl-sim.md   — hyperparameter delta table
│   ├── 02-改进记录/                         ← changes against upstream
│   │   └── CHANGES.md                         — V1 → V2.0 → V2.1 changelog
│   ├── 03-部署与训练指南/                    ← deployment & training playbooks
│   │   ├── DEPLOY_GUIDE.md                    — end-to-end environment setup
│   │   ├── 实验合作内容.md                      — collaborator playbook
│   │   └── TOUCH_INTERVENTION_GUIDE.md        — touch-haptic teleop wiring
│   ├── 04-远程算力与传输/                    ← remote compute & transfer
│   │   ├── hil-serl-sim_算力与远程训练方案.md    — compute / remote training plan
│   │   └── 远程算力机加速拉取方案.md             — faster `git pull` from China-region pods
│   └── 05-结果分析/                         ← experiment results
│       └── hil-serl-sim_复现结果分析.md         — upstream-run diagnostic notes
│
├── results/               ← shared task results (4090 smoke runs, figures)
├── academic/              ← LOCAL-ONLY (private working materials)
│
├── v1-pytorch/            ← V1 PyTorch — sparse reward (lerobot)
│   ├── lerobot/           ← entry point: lerobot/run_learner.sh
│   └── SurRoL_v2/         ← vendored sim
│
├── v2-pytorch/            ← V2 PyTorch — hybrid V2.1 reward + F7 logging (lerobot)
│   ├── lerobot/           ← entry point: lerobot/run_learner.sh
│   └── SurRoL_v2/
│
├── v1-jax/                ← upstream JAX baseline, unmodified
│   └── hil-serl-sim/      ← entry point: hil-serl-sim/run_learner.sh
│
└── v2-jax/                ← V2 JAX — byte-identical mirror of v2-pytorch's reward
    └── hil-serl-sim/
```

All technical documentation is at the repo root in `docs/` — the
variant subdirectories hold only code (no per-variant `docs/` folder).
Run paths inside each guide are written relative to a variant root,
so prefix them with `cd v2-pytorch/` (or whichever variant you're
running) before following the steps.

`academic/` is in `.gitignore` and never published. See
[`academic/README.md`](academic/README.md) (locally) for what lives there.

### Quick start

```bash
# 1. Clone
git clone https://github.com/LIJianxuanLeo/hil-serl-ImprovedVersion.git
cd hil-serl-ImprovedVersion

# 2. Pick a variant — example: V2 PyTorch (recommended for new work)
cd v2-pytorch

# 3. Set up Python env (one-time)
bash setup.sh                # installs SurRoL_v2, lerobot extras, gym_hil

# 4. Configure wandb (one-time, optional but recommended)
wandb login

# 5. Two-terminal training: learner first
cd lerobot && ./run_learner.sh

# 6. Then in a second terminal — wait for "[LEARNER] gRPC server started"
cd lerobot && ./run_actor.sh
```

For headless / SSH / no-GUI machines, swap `./run_learner.sh` →
`./run_learner.sh headless` and add `MUJOCO_GL=egl`.
Detailed instructions: [`docs/03-部署与训练指南/DEPLOY_GUIDE.md`](docs/03-部署与训练指南/DEPLOY_GUIDE.md).

### Where to look for what

Full docs index: [`docs/README.md`](docs/README.md).

| I want to … | Go to |
|---|---|
| Train V2.1 hybrid reward (recommended) | `v2-pytorch/lerobot/run_learner.sh` |
| Compare with sparse baseline | `v1-pytorch/lerobot/run_learner.sh` |
| Reproduce the 10%-success run that started this work | `v1-jax/hil-serl-sim/run_learner.sh` |
| JAX-vs-PyTorch comparison with same reward | `v2-jax/` paired with `v2-pytorch/` |
| Full reward-redesign rationale | [`docs/02-改进记录/CHANGES.md`](docs/02-改进记录/CHANGES.md) §8 / §8b |
| Operational playbook for collaborator | [`docs/03-部署与训练指南/实验合作内容.md`](docs/03-部署与训练指南/实验合作内容.md) |
| End-to-end environment setup | [`docs/03-部署与训练指南/DEPLOY_GUIDE.md`](docs/03-部署与训练指南/DEPLOY_GUIDE.md) |
| Touch-haptic intervention guide | [`docs/03-部署与训练指南/TOUCH_INTERVENTION_GUIDE.md`](docs/03-部署与训练指南/TOUCH_INTERVENTION_GUIDE.md) |
| Hyperparameter delta across variants | [`docs/01-架构与设计/参数对比_V1_V2_vs_hil-serl-sim.md`](docs/01-架构与设计/参数对比_V1_V2_vs_hil-serl-sim.md) |
| Compute selection / remote training | [`docs/04-远程算力与传输/hil-serl-sim_算力与远程训练方案.md`](docs/04-远程算力与传输/hil-serl-sim_算力与远程训练方案.md) |
| Faster `git pull` for China-region pods | [`docs/04-远程算力与传输/远程算力机加速拉取方案.md`](docs/04-远程算力与传输/远程算力机加速拉取方案.md) |
| Architecture: how the 4 variants relate | [`docs/01-架构与设计/architecture-overview.md`](docs/01-架构与设计/architecture-overview.md) |
| Upstream reproduction analysis (10% success) | [`docs/05-结果分析/hil-serl-sim_复现结果分析.md`](docs/05-结果分析/hil-serl-sim_复现结果分析.md) |
| Numerical results (smoke runs, figures) | [`results/4090_smoke_run/`](results/4090_smoke_run/) |

### License

The bridging code (consolidation scripts, top-level `docs/`, `results/`)
is released under MIT. Each variant subdirectory inherits any upstream
licenses that originally applied to the code it contains.

---

<a id="中文"></a>
## 中文

### 这个仓库为什么存在

HIL-SERL（[Luo et al. 2024](https://arxiv.org/abs/2410.21845)）声称能让真实
机器人操控以接近 100% 的成功率高样本效率收敛。但合作者在 AutoDL 服务器上
跑公开复现，跑了 **220 K 步 / 3.83 小时 GPU 时间，最终只有 10% 成功率** ——
这就是复现差距。本仓库整合了我们为弥合这个差距而做的修复链、消融实验和
工具基础设施。

整体按 **奖励设计 × RL 框架** 切成 4 格：

|  | **PyTorch / lerobot** | **JAX / serl_launcher** |
|---|---|---|
| **V1（稀疏奖励）** | [`v1-pytorch/`](v1-pytorch) —— 我们的第一个改进 fork | [`v1-jax/`](v1-jax) —— 上游 baseline（GeorgeAuburn/hilserl_sim_v1） |
| **V2（稀疏 + 密集 shaping 混合）** | [`v2-pytorch/`](v2-pytorch) —— V2.1 奖励 + F7 日志 | [`v2-jax/`](v2-jax) —— V2.1 在 JAX 端的字节对齐镜像 |

四格在我们要保持一致的所有维度上 **刻意做了字节级对齐**，所以沿任何一个
轴做对照都只剩一个变量。

### 奖励设计演进

| | V1 sparse | V2.0 dense（已弃用） | **V2.1 hybrid（当前）** |
|---|---|---|---|
| 每步 reward | 0 | 0–10（`r_reach + r_grasp + r_lift`）| clip(`1·r_succ + 0.05·r_app + 0.1·r_grasp + 0.05·r_lift − 0.01·r_pen`, −1, 2) |
| 成功奖励 | +1（覆盖） | +10（加法） | +1（加法） |
| 接近 | 无 | 连续 `1/(1+5d)` | **差分** `clip(d_prev−d_now, ±0.02)` |
| 抓取 | 无 | 平滑渐变 8 → 1 cm | **事件型**，单 episode 只发一次 |
| 抬起 | 无 | 线性 `lift / 0.10` | **截断** `clip(z − z_pick_ref, 0, 0.05)` |
| Episode 累积 reward 量级 | {0, 1} | 15–60 | 0.0–0.6（失败）/ 1.2–1.8（成功） |
| 反 reward-hacking 保证 | 天然安全 | 暴露于悬停 / 轻触 / 抬高 三种刷分 | 三种风险均按构造性闭合 |

V2.0 → V2.1 的重构是本项目实质性的工程贡献：设计来自
`hilserl_抓取任务_稀疏奖励替换为稠密奖励_完整实现流程.docx`
（2026-05-04 工程方案），我们在抓取事件检测与抬升 clip 上做了少量精化后
完整移植。消融指引见 [`docs/02-改进记录/CHANGES.md`](docs/02-改进记录/CHANGES.md) §8b。

### F6 + F7 日志

两个 PyTorch 变体都自带日志器（`lerobot/src/lerobot/rl/training_logger.py`），
除了标准的 SAC/REDQ 指标外还会输出：

- **F6**（训练，32 列）：`q_target_mean/std`、`td_error_mean/std/max`、
  `critic_disagreement`、`policy_entropy_raw`、`actor_loss_q_term`、
  `actor_loss_entropy_term`、`step_time_ms`、`gpu_mem_peak_mb`……
- **F6**（每 episode）：`is_success`、`rolling_success_rate_50`、
  `rolling_intervention_rate_50`、**`rolling_policy_only_success_20`**
  （仅统计无干预 episode，用于看 policy 真实自主能力）、
  `episode_length`、`termination_reason`……
- **F7**（仅 V2.1，每 episode 分阶段 reward 归因）：
  `r_sub_success / r_sub_approach / r_sub_grasp / r_sub_lift /
  r_sub_penalty / grasp_event_fired`。这让方案 §11 的 A/B/C 对照能区分
  "policy 真正完成了任务" 和 "policy 把 shaping 项刷满"。

所有字段同时同步到 wandb（`train_config_*.json` 里
`wandb.enable=true`）。

### 目录结构

```
hil-serl-ImprovedVersion/
├── README.md              ← 本文件（中英对照）
├── docs/                  ← 所有统一技术文档，按用途分组
│   ├── README.md                          ← 文档索引
│   ├── 01-架构与设计/                       ← 架构与设计对齐
│   │   ├── architecture-overview.md           — 2x2 矩阵跨变体导航
│   │   ├── sim_architecture_alignment.md      — PyTorch 仿真 ↔ JAX 仿真对齐
│   │   └── 参数对比_V1_V2_vs_hil-serl-sim.md   — 超参差异表
│   ├── 02-改进记录/                         ← 相对上游的修改
│   │   └── CHANGES.md                         — V1 → V2.0 → V2.1 变更日志
│   ├── 03-部署与训练指南/                    ← 部署与训练剧本
│   │   ├── DEPLOY_GUIDE.md                    — 端到端环境部署
│   │   ├── 实验合作内容.md                      — 协作者训练剧本
│   │   └── TOUCH_INTERVENTION_GUIDE.md        — Touch 触觉介入接线
│   ├── 04-远程算力与传输/                    ← 远程算力与传输
│   │   ├── hil-serl-sim_算力与远程训练方案.md    — 算力选型与远程训练方案
│   │   └── 远程算力机加速拉取方案.md             — 国内服务器加速 git pull
│   └── 05-结果分析/                         ← 实验结果
│       └── hil-serl-sim_复现结果分析.md         — 上游复现诊断报告
│
├── results/               ← 共享任务结果（4090 smoke 数据、图表）
├── academic/              ← 仅本地（私人工作材料）
│
├── v1-pytorch/            ← V1 PyTorch —— 稀疏奖励（lerobot）
│   ├── lerobot/           ← 入口：lerobot/run_learner.sh
│   └── SurRoL_v2/         ← vendored 仿真
│
├── v2-pytorch/            ← V2 PyTorch —— V2.1 hybrid 奖励 + F7 日志（lerobot）
│   ├── lerobot/           ← 入口：lerobot/run_learner.sh
│   └── SurRoL_v2/
│
├── v1-jax/                ← 上游 JAX baseline，未做修改
│   └── hil-serl-sim/      ← 入口：hil-serl-sim/run_learner.sh
│
└── v2-jax/                ← V2 JAX —— v2-pytorch reward 的字节镜像
    └── hil-serl-sim/
```

所有技术文档都在仓库根的 `docs/` —— 各变体子目录只放代码（没有变体内的
`docs/` 子文件夹）。文档里的 run 路径都是相对变体根写的，跑前先
`cd v2-pytorch/`（或别的变体）即可。

`academic/` 在 `.gitignore` 里，永不上传 GitHub。本地说明见
[`academic/README.md`](academic/README.md)。

### 快速上手

```bash
# 1. clone
git clone https://github.com/LIJianxuanLeo/hil-serl-ImprovedVersion.git
cd hil-serl-ImprovedVersion

# 2. 选一个变体 —— 推荐新工作用 V2 PyTorch
cd v2-pytorch

# 3. 装环境（一次性）
bash setup.sh                # 装 SurRoL_v2 / lerobot extras / gym_hil

# 4. 登录 wandb（一次性，建议）
wandb login

# 5. 双终端训练 —— learner 先
cd lerobot && ./run_learner.sh

# 6. 第二个终端 —— 看到 "[LEARNER] gRPC server started" 后再启动 actor
cd lerobot && ./run_actor.sh
```

无显示器 / SSH 远程跑：把 `./run_learner.sh` 换成
`./run_learner.sh headless` 并加 `MUJOCO_GL=egl`。
完整指引：[`docs/03-部署与训练指南/DEPLOY_GUIDE.md`](docs/03-部署与训练指南/DEPLOY_GUIDE.md)。

### 想找什么去哪里

完整文档索引：[`docs/README.md`](docs/README.md)。

| 我想…… | 去 |
|---|---|
| 跑 V2.1 hybrid 奖励（推荐） | `v2-pytorch/lerobot/run_learner.sh` |
| 跑稀疏 baseline 做对照 | `v1-pytorch/lerobot/run_learner.sh` |
| 复现合作者那次 10% 成功率 | `v1-jax/hil-serl-sim/run_learner.sh` |
| 同 reward 下 JAX vs PyTorch 对比 | `v2-jax/` 与 `v2-pytorch/` 配合 |
| 看 reward 重构完整原理 | [`docs/02-改进记录/CHANGES.md`](docs/02-改进记录/CHANGES.md) §8 / §8b |
| 合作者训练 / 协作剧本 | [`docs/03-部署与训练指南/实验合作内容.md`](docs/03-部署与训练指南/实验合作内容.md) |
| 端到端部署 | [`docs/03-部署与训练指南/DEPLOY_GUIDE.md`](docs/03-部署与训练指南/DEPLOY_GUIDE.md) |
| Touch 触觉干预指引 | [`docs/03-部署与训练指南/TOUCH_INTERVENTION_GUIDE.md`](docs/03-部署与训练指南/TOUCH_INTERVENTION_GUIDE.md) |
| 各变体超参差异表 | [`docs/01-架构与设计/参数对比_V1_V2_vs_hil-serl-sim.md`](docs/01-架构与设计/参数对比_V1_V2_vs_hil-serl-sim.md) |
| 算力选型 / 远程训练方案 | [`docs/04-远程算力与传输/hil-serl-sim_算力与远程训练方案.md`](docs/04-远程算力与传输/hil-serl-sim_算力与远程训练方案.md) |
| 国内服务器加速 git pull | [`docs/04-远程算力与传输/远程算力机加速拉取方案.md`](docs/04-远程算力与传输/远程算力机加速拉取方案.md) |
| 架构：四个变体如何关联 | [`docs/01-架构与设计/architecture-overview.md`](docs/01-架构与设计/architecture-overview.md) |
| 上游复现实测结果（10% 成功率诊断） | [`docs/05-结果分析/hil-serl-sim_复现结果分析.md`](docs/05-结果分析/hil-serl-sim_复现结果分析.md) |
| 数值结果（smoke run、图表） | [`results/4090_smoke_run/`](results/4090_smoke_run/) |

### 许可

整合层代码（顶层 `docs/`、`results/`、整合脚本）按 MIT 发布。各变体子目录
继承其代码原本适用的上游许可。
