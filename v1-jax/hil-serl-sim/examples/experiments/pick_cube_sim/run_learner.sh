SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 部分云镜像 /lib 下 libcuda.so 为 0 字节占位，动态链接会误加载导致 JAX/cuDNN 初始化失败；
# 须让真实驱动用户态库（与 nvidia-smi 驱动版本一致）优先于占位路径。
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 让脚本不依赖 .bashrc：headless 渲染 + jax cuda_nvcc workaround + tf 静音
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
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

RUN_NAME="${RUN_NAME:-pick_cube_sim_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$RUN_NAME}"
echo "$CHECKPOINT_PATH" > .last_run_name
echo "Training checkpoint path: $CHECKPOINT_PATH"

# 优先 PICK_CUBE_DEMO_PKL；否则取 demo_data/ 下最新一个 .pkl（ls -t 按修改时间倒序）
DEMO_PKL="${PICK_CUBE_DEMO_PKL:-}"
if [ -z "$DEMO_PKL" ]; then
    DEMO_PKL="$(ls -1t demo_data/*.pkl 2>/dev/null | head -n 1)"
fi
if [ -z "$DEMO_PKL" ] || [ ! -f "$DEMO_PKL" ]; then
    echo "[run_learner.sh] 未找到 demo .pkl：请把 demo_data/*.pkl 准备好，或 export PICK_CUBE_DEMO_PKL=/abs/path/xxx.pkl" >&2
    exit 1
fi
echo "Using demo: $DEMO_PKL"

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}" && \
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}" && \
# W&B：默认开启 online。关闭请 export WANDB_DISABLED=1
# 认证（任选其一）：
#   1) 运行一次 wandb login（写入 ~/.netrc）
#   2) export WANDB_API_KEY=...（勿写入 git）
#   3) 在本目录创建 .wandb_api_key 文件（一行 key，已加入 .gitignore）
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${SCRIPT_DIR}/.wandb_api_key" ]; then
    export WANDB_API_KEY="$(tr -d '\n\r' < "${SCRIPT_DIR}/.wandb_api_key")"
fi
export WANDB_MODE="${WANDB_MODE:-online}" && \
export WANDB_DISABLED="${WANDB_DISABLED:-0}" && \
python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --demo_path="$DEMO_PKL" \
    --learner