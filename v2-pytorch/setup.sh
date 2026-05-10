#!/usr/bin/env bash
# HIL-SERL SurRoL 一键部署脚本
# 适用于 Ubuntu 22.04 + NVIDIA GPU + Geomagic Touch
set -e

echo "============================================"
echo "  HIL-SERL SurRoL 部署脚本"
echo "============================================"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
HAPTIC_MODULE_PATH="${PROJECT_ROOT}/SurRoL_v2/haptic_src"

echo ""
echo "[1/7] 检测系统环境..."
echo "  项目目录: ${PROJECT_ROOT}"
echo "  Python:   $(python3 --version 2>/dev/null || echo 'NOT FOUND')"
echo "  CUDA:     $(nvcc --version 2>/dev/null | head -1 || echo 'NOT FOUND')"

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "  [WARN] nvidia-smi 未找到，请确保已安装 NVIDIA 驱动"
fi

echo ""
echo "[2/7] 创建 conda 环境..."
if conda info --envs 2>/dev/null | grep -q "hilserl"; then
    echo "  环境 'hilserl' 已存在，跳过创建"
else
    conda create -n hilserl python=3.12 -y
    echo "  环境 'hilserl' 创建完成"
fi

echo ""
echo "[3/7] 安装 SurRoL_v2..."
cd "${PROJECT_ROOT}/SurRoL_v2"

# Initialize and update submodules
if [ ! -f "ext/bullet3/setup.py" ]; then
    echo "  初始化 git submodules..."
    cd "${PROJECT_ROOT}"
    git submodule update --init --recursive
    cd "${PROJECT_ROOT}/SurRoL_v2"
fi

eval "$(conda shell.bash hook)"
conda activate hilserl

pip install -e . 2>&1 | tail -3
echo "  SurRoL_v2 安装完成"

echo ""
echo "[4/7] 编译 Touch 触觉设备驱动..."
cd "${PROJECT_ROOT}/SurRoL_v2"
if [ -f "haptic_src/_touch_haptic.cpython-312-x86_64-linux-gnu.so" ] || \
   ls haptic_src/_touch_haptic.*.so 2>/dev/null | grep -q ".so"; then
    echo "  Touch 驱动已编译，跳过"
else
    echo "  检查 OpenHaptics SDK..."
    if [ -d "/opt/OpenHaptics" ] || [ -d "/usr/local/OpenHaptics" ]; then
        echo "  编译 Touch SWIG 绑定..."
        bash setup_haptic.sh
        echo "  Touch 驱动编译完成"
    else
        echo "  [WARN] OpenHaptics SDK 未安装"
        echo "         请参考文档 docs/DEPLOY_GUIDE.md 安装 OpenHaptics"
        echo "         安装后运行: cd SurRoL_v2 && bash setup_haptic.sh"
    fi
fi

echo ""
echo "[5/7] 安装 LeRobot..."
cd "${PROJECT_ROOT}/lerobot"
pip install -e ".[sac]" 2>&1 | tail -3

# Install additional dependencies
pip install grpcio protobuf 2>&1 | tail -1
echo "  LeRobot 安装完成"

echo ""
echo "[6/7] 配置路径..."
HAPTIC_PATH_ESCAPED=$(echo "${HAPTIC_MODULE_PATH}" | sed 's/\//\\\//g')

# Update all config files with correct haptic path
for config_file in "${PROJECT_ROOT}/lerobot/"*config*.json; do
    if [ -f "$config_file" ]; then
        sed -i "s/__HAPTIC_MODULE_PATH__/${HAPTIC_PATH_ESCAPED}/g" "$config_file"
        echo "  更新: $(basename $config_file)"
    fi
done
echo "  路径配置完成"

echo ""
echo "[7/7] 链接数据集..."
DATASET_SRC="${PROJECT_ROOT}/lerobot/franka_sim_touch_demos"
DATASET_DST="$HOME/.cache/huggingface/lerobot/local/franka_sim_touch_demos"

if [ -d "$DATASET_SRC" ]; then
    mkdir -p "$(dirname $DATASET_DST)"
    if [ -L "$DATASET_DST" ] || [ -d "$DATASET_DST" ]; then
        echo "  数据集链接已存在，跳过"
    else
        ln -s "$DATASET_SRC" "$DATASET_DST"
        echo "  数据集已链接到 HuggingFace 缓存"
    fi
else
    echo "  [INFO] 未找到预采集数据集"
    echo "         需要先运行数据采集步骤（参考 docs/DEPLOY_GUIDE.md）"
fi

echo ""
echo "============================================"
echo "  部署完成！"
echo "============================================"
echo ""
echo "使用方法："
echo "  conda activate hilserl"
echo ""
echo "  # 步骤1: 采集演示数据（如已有数据集可跳过）"
echo "  cd ${PROJECT_ROOT}/lerobot"
echo "  python -m lerobot.rl.gym_manipulator --config_path env_config_gym_hil_touch_record.json"
echo ""
echo "  # 步骤2: 启动 Learner（终端1）"
echo "  python -m lerobot.rl.learner --config_path train_config_gym_hil_touch.json"
echo ""
echo "  # 步骤3: 启动 Actor（终端2）"
echo "  python -m lerobot.rl.actor --config_path train_config_gym_hil_touch.json"
echo ""
