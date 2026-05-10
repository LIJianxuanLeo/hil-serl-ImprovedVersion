SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 强制按 DISPLAY 选渲染后端（不允许外部残留 MUJOCO_GL 干扰）：
#   - 有 DISPLAY → glfw（笔记本本地：弹 mujoco 窗口可视化 + pynput 听键）
#   - 无 DISPLAY → egl（headless 服务器：离屏渲染）
if [ -n "${DISPLAY:-}" ]; then
    export MUJOCO_GL=glfw
    unset PYOPENGL_PLATFORM
else
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
fi
echo "[run_actor.sh] MUJOCO_GL=$MUJOCO_GL  DISPLAY=$DISPLAY"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
if [ -z "${CUDA_ROOT:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_ROOT=/usr/local/cuda
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
        export CUDA_ROOT="$CONDA_PREFIX"
    else
        export CUDA_ROOT=/tmp
    fi
fi

if [ -z "${RUN_NAME:-}" ] && [ -f .last_run_name ]; then
    RUN_NAME="$(cat .last_run_name)"
fi
RUN_NAME="${RUN_NAME:-pick_cube_sim_$(date +%Y%m%d_%H%M%S)}"
echo "Actor run name: $RUN_NAME"

# 过小 (如 .1) 在 6GB 级显卡上易导致 cuDNN 初始化失败 CUDNN_STATUS_INTERNAL_ERROR
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" && \
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}" && \
export WANDB_MODE="${WANDB_MODE:-offline}" && \
export WANDB_API_KEY="${WANDB_API_KEY:-local-anonymous-not-used}" && \
export WANDB_DISABLED="${WANDB_DISABLED:-true}" && \
python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --actor