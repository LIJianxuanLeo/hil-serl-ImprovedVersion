# HIL-SERL SurRoL (Improved)

基于 [HIL-SERL](https://github.com/rail-berkeley/hil-serl) 的人类在环强化学习框架，在 [SurRoL v2](https://github.com/med-air/SurRoL) 仿真环境中训练 Franka Panda 机械臂完成拾取任务。

本仓库基于 [GeorgeAuburn/hilserl-surrol](https://github.com/GeorgeAuburn/hilserl-surrol) 改进，修复了导致训练不收敛的关键配置问题，并新增阶段式密集奖励、训练日志存储、可视化等功能。

## 主要改进

### 超参数修复（解决不收敛问题）

| 改进项 | 原始值 | 改进值 | 影响 |
|--------|--------|--------|------|
| SAC 温度 | 0.01 | **1.0** | 修复探索能力丧失（核心问题） |
| 折扣因子 | 0.97 | **0.99** | 有效视野 3.3s → 10s |
| 温度学习率 | 0.0003 | **0.001** | 加快自适应调节 |
| Actor 学习率 | 0.0003 | **0.0001** | 稳定策略更新 |
| 梯度裁剪 | 10.0 | **1.0** | 防止梯度爆炸 |
| Batch Size | 256 | **128** | 防止 RTX 3060 OOM |
| 动作范围 | [-0.4, 0.4] | **[-1.0, 1.0]** | 匹配实际数据 |
| 硬编码路径 | `/home/zjj/...` | **自动配置** | 可移植部署 |

### 奖励函数改进（加速收敛）

将原始的**稀疏奖励**（仅成功时 reward=1.0）替换为**阶段式密集奖励**（StagedRewardWrapper），提供连续梯度信号：

| 阶段 | 奖励范围 | 条件 |
|------|---------|------|
| 接近 (Approach) | 0 ~ 0.25 | `exp(-10*dist)` 靠近方块 |
| 抓取 (Grasp) | 0 ~ 0.25 | 抓住方块 = 0.25，靠近 = 0.10 |
| 抬起 (Lift) | 0 ~ 0.50 | 按抬起高度比例给分 |
| 成功 | 1.0 | 完成任务 |

### 训练日志存储

训练过程自动保存 CSV 格式日志，无需 WandB 也可回看训练进度：

```
output_dir/training_logs/
├── training_metrics.csv     # 训练指标（loss, temperature, grad_norm...）
├── episode_metrics.csv      # Episode 指标（reward, intervention_rate...）
└── training_summary.json    # 训练总结（最佳奖励、总步数、训练时长...）
```

**可视化训练曲线**：
```bash
python plot_training.py outputs/train/franka_sim_sac_touch
```
生成 `training_curves.png`，包含 Loss、Temperature、Buffer Size、Reward、Intervention Rate 五张曲线图。

## 快速开始

```bash
# 克隆（包含子模块）
git clone --recursive https://github.com/LIJianxuanLeo/hilserl-surrol-improved.git
cd hilserl-surrol-improved

# 一键部署
bash setup.sh

# 激活环境
conda activate hilserl
cd lerobot

# 终端1: 启动 Learner
python -m lerobot.rl.learner --config_path train_config_gym_hil_touch.json

# 终端2: 启动 Actor（等 Learner 启动后）
python -m lerobot.rl.actor --config_path train_config_gym_hil_touch.json
```

## 训练监控

### 方式一：CSV 日志（默认启用）

训练自动生成 CSV 日志，训练结束后或中途查看：

```bash
# 查看最新训练指标
tail -5 outputs/train/franka_sim_sac_touch/training_logs/training_metrics.csv

# 查看最新 episode 奖励
tail -10 outputs/train/franka_sim_sac_touch/training_logs/episode_metrics.csv

# 查看训练总结
cat outputs/train/franka_sim_sac_touch/training_logs/training_summary.json

# 绘制训练曲线
python plot_training.py outputs/train/franka_sim_sac_touch
```

### 方式二：WandB（可选）

```bash
pip install wandb
wandb login
# 修改 train_config_gym_hil_touch.json 中 "wandb": { "enable": true }
```

## 系统要求

- Ubuntu 22.04
- NVIDIA GPU (RTX 3060 12GB 或更高)
- Geomagic Touch 触觉设备 + OpenHaptics SDK
- Conda (Python 3.12)

## 文档

- **[部署与训练指南](docs/DEPLOY_GUIDE.md)** - 完整的中文安装、配置、训练、调参指南
- **[改进记录](docs/CHANGES.md)** - 所有修改项的技术分析

## 项目结构

```
hilserl-surrol-improved/
├── setup.sh                          # 一键部署脚本
├── docs/
│   ├── DEPLOY_GUIDE.md               # 详细部署指南（中文）
│   └── CHANGES.md                    # 改进记录
├── lerobot/                          # 修改版 LeRobot（SAC + HIL）
│   ├── train_config_gym_hil_touch.json       # 训练配置（已修复）
│   ├── env_config_gym_hil_touch_record.json  # 数据采集配置
│   ├── env_config_gym_hil_il.json            # 键盘采集配置
│   ├── plot_training.py                      # 训练曲线可视化工具
│   ├── franka_sim_touch_demos/               # 预采集演示数据（30 episodes）
│   └── src/lerobot/
│       ├── rl/
│       │   ├── learner.py            # Learner（含 CSV 日志集成）
│       │   ├── actor.py              # Actor
│       │   ├── training_logger.py    # CSV 训练日志记录器
│       │   ├── staged_reward_wrapper.py  # 阶段式密集奖励
│       │   ├── gym_manipulator.py    # 环境封装
│       │   └── wandb_utils.py        # WandB 日志工具
│       ├── policies/sac/             # SAC 策略（含离散动作修改）
│       ├── teleoperators/touch/      # Touch 触觉设备集成
│       └── processor/hil_processor.py # HIL 干预处理器
└── SurRoL_v2/                        # SurRoL 仿真环境
    ├── surrol/                       # 仿真核心（PyBullet）
    ├── haptic_src/                   # Touch 设备 SWIG 绑定
    └── ext/                          # 子模块（bullet3, pybullet_rendering）
```

## 训练架构

```
[Touch 触觉设备] → [Actor] ←gRPC:50051→ [Learner]
                      ↓                      ↓
              [SurRoL 仿真环境]         [SAC 策略更新]
              [PyBullet + Panda]        [Replay Buffer]
                      ↓                      ↓
              [episode_metrics.csv]     [training_metrics.csv]
                                             ↓
                                    [training_summary.json]
```

### 人工干预（HIL）

- **Button 1**：按住 = 人工接管，松开 = 策略自主
- **Button 2**：切换夹爪开合
- 人工干预 transitions 同时进入 online/offline buffer，batch 50/50 混合采样

### 收敛预期（使用阶段式奖励 + RTX 3060）

| 训练阶段 | 预期表现 |
|----------|---------|
| 0~5K 步 | reward 开始 > 0.05，学会移动方向 |
| 5K~15K 步 | reward 达到 0.35+，开始抓取 |
| 15K~30K 步 | reward 出现 0.50+，成功抬起 |
| 30K+ 步 | 成功率 > 80% |

## License

Apache-2.0
