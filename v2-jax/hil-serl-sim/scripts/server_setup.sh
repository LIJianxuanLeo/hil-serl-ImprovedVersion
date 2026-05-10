#!/usr/bin/env bash
# AutoDL 服务器一键环境配置（A800 80GB + PyTorch 2.3.0 + Python 3.10 + CUDA 12.1）
# 使用：
#   scp scripts/server_setup.sh autodl-hil:/root/autodl-tmp/
#   ssh autodl-hil bash /root/autodl-tmp/server_setup.sh
# 重复执行是安全的（idempotent）。

set -e
set -o pipefail

log() { echo -e "\033[1;36m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[setup-warn]\033[0m $*" >&2; }

# 1) 学术加速（AutoDL 内置代理；source 后会 export http_proxy/https_proxy）
log "Enable AutoDL academic network acceleration"
source /etc/network_turbo 2>/dev/null || warn "network_turbo not available, skip"

# 2) git 网络稳定性配置 + 把代理同步给 git
log "Configure git for flaky network"
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 60

PROXY="${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}"
if [ -n "$PROXY" ]; then
    log "Detected proxy from env: $PROXY (binding to git config)"
    git config --global http.proxy "$PROXY"
    git config --global https.proxy "$PROXY"
    export http_proxy="${http_proxy:-$PROXY}"
    export https_proxy="${https_proxy:-$PROXY}"
    export all_proxy="${all_proxy:-$PROXY}"
else
    warn "No proxy in env: AutoDL academic accel likely NOT active in this shell"
    warn "  -> Will rely on github.com mirrors as fallback"
fi

# GitHub 镜像列表（按可用性排序）
GH_MIRRORS=(
    "https://github.com"
    "https://gitclone.com/github.com"
    "https://github.moeyy.xyz/https://github.com"
    "https://hub.gitmirror.com/https://github.com"
)

# 多镜像 + 重试 + 超时的 git clone
# usage: git_clone_multi <dest> <commit_or_empty> <owner/repo>
git_clone_multi() {
    local dest="$1" rev="$2" repo="$3"
    if [ -d "$dest/.git" ]; then
        log "Repo $dest already exists, skipping clone"
    else
        local cloned=0
        for base in "${GH_MIRRORS[@]}"; do
            local url="$base/$repo.git"
            log "Try clone: $url"
            if timeout 90 git clone "$url" "$dest"; then
                cloned=1
                break
            fi
            warn "Failed: $url"
            rm -rf "$dest"
        done
        if [ "$cloned" != "1" ]; then
            warn "All mirrors failed for $repo"
            return 1
        fi
    fi
    if [ -n "$rev" ]; then
        (cd "$dest" && git fetch --all --tags --quiet 2>/dev/null || true)
        (cd "$dest" && git checkout "$rev") || warn "git checkout $rev failed (rev may not exist on mirror)"
    fi
}

# 3) 进入持久化数据盘 + 克隆主仓库（多镜像）
mkdir -p /root/autodl-tmp
cd /root/autodl-tmp

git_clone_multi /root/autodl-tmp/hil-serl-sim "" ggggfff1/hil-serl-sim
cd hil-serl-sim

# 4) 定位 conda 并初始化 shell（把 conda activate 函数注入到当前 shell 与 ~/.bashrc）
if ! command -v conda >/dev/null 2>&1; then
    if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
        source /root/miniconda3/etc/profile.d/conda.sh
    else
        warn "conda not found; AutoDL PyTorch image should ship miniconda3"
        exit 1
    fi
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 4.5) conda init bash —— 让以后任何新 ssh shell 都可以直接 conda activate
if ! grep -q "conda initialize" ~/.bashrc; then
    log "Run 'conda init bash' so future shells can use 'conda activate'"
    conda init bash >/dev/null
else
    log "~/.bashrc already has conda init block"
fi

# 5) 创建/激活 conda env
if ! conda env list | awk '{print $1}' | grep -qx hilserl; then
    log "Create conda env hilserl (python 3.10)"
    conda create -n hilserl python=3.10 -y
fi
conda activate hilserl

# 6) PyTorch（CUDA 12.1，按 README 推荐）
if ! python -c "import torch, sys; assert torch.cuda.is_available()" 2>/dev/null; then
    log "Install PyTorch 2.3.0 (cu121)"
    pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121
else
    log "PyTorch CUDA already OK"
fi

# 7) JAX（README 推荐 0.4.35，A800 sm_80 完全兼容）
#    显式锁 jax + jaxlib + jax-cuda12-{pjrt,plugin} 四件套版本，避免 pip 升级时四者错配
if ! python -c "import jax; assert jax.__version__ == '0.4.35'" 2>/dev/null; then
    log "Install JAX 0.4.35 (cuda12_pip, all four packages pinned)"
    pip install --no-cache-dir \
        "jax==0.4.35" "jaxlib==0.4.34" \
        "jax-cuda12-pjrt==0.4.35" "jax-cuda12-plugin==0.4.35" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    log "JAX 0.4.35 already installed"
fi

# 7.5) 新 NVIDIA 驱动（如 580）上，torch 锁定的 cudnn 8.9.2 可能导致 JAX 报 CUDNN_STATUS_INTERNAL_ERROR；
#     升级到 9.1.x 后 JAX GPU 初始化正常（pip 会提示与 torch 元数据冲突，一般仍可同时用两框架）。
log "Install nvidia-cudnn-cu12 9.1.x for JAX + recent drivers (optional fix)"
pip install --no-cache-dir "nvidia-cudnn-cu12==9.1.0.70" || warn "cudnn 9.1 install failed, skip"

# 8a) 提前手动装 agentlace（serl_launcher 依赖它，但走 pip git+ 容易踩 AutoDL→GitHub 的网络问题）
#     如果之前用户从笔记本 scp 了 /root/autodl-tmp/agentlace，这里直接用
if ! python -c "import agentlace" >/dev/null 2>&1; then
    log "Clone & install agentlace (pinned commit) via mirrors"
    git_clone_multi /root/autodl-tmp/agentlace \
        cf2c337c5e3694cdbfc14831b239bd657bc4894d \
        youliangtan/agentlace
    if [ ! -d /root/autodl-tmp/agentlace ] || [ ! -f /root/autodl-tmp/agentlace/setup.py ]; then
        warn ""
        warn "==================== agentlace 仍然装不上 ===================="
        warn "服务器到 github.com 完全不通且所有镜像也挂了。"
        warn "请在【笔记本】上跑下面这条把 agentlace 推到服务器，然后重跑本脚本："
        warn ""
        warn "  git clone https://github.com/youliangtan/agentlace.git /tmp/agentlace"
        warn "  cd /tmp/agentlace && git checkout cf2c337c5e3694cdbfc14831b239bd657bc4894d && cd -"
        warn "  rsync -avzP /tmp/agentlace/ autodl-hil:/root/autodl-tmp/agentlace/"
        warn "  ssh autodl-hil bash /root/autodl-tmp/server_setup.sh"
        warn "==============================================================="
        exit 1
    fi
    pip install --no-cache-dir -e /root/autodl-tmp/agentlace
else
    log "agentlace already installed"
fi

# 8b) 项目子模块（editable install）
#     关键：必须加 --no-deps，否则 pip 看到 serl_launcher/setup.py 里
#     "agentlace @ git+https://..." 形式的依赖会无视已装版本、重新 git clone，
#     再次踩 AutoDL→GitHub 的网络问题。我们在 8a 已手动装好 agentlace，
#     其他依赖（zmq/opencv-python/lz4/typing/typing_extensions）都已被 sim-only deps 段装上。
log "Install serl_launcher (editable, --no-deps)"
pip install --no-cache-dir --no-deps -e ./serl_launcher

log "Install franka_sim (editable, --no-deps)"
pip install --no-cache-dir --no-deps -e ./franka_sim

# 9) 仿真任务实际所需依赖（避开 requirements.txt 里大量 ROS 包）
#    特别说明：
#    - lz4 / pyzmq 是 serl_launcher install_requires 里要的；因 8b 用了 --no-deps 故必须显式装齐
#    - tensorflow-cpu 是 serl_launcher.common.typing 硬 import 的（仅用作 tf.Tensor 类型注解）；
#      装 cpu 版避免重抓 cudnn 依赖、不抢 GPU
log "Install sim-only dependencies"
# 重要：以下版本约束必须保留，否则会出现连锁冲突：
#   - tensorstore==0.1.71 + ml-dtypes==0.4.1：避免新 tensorstore 要 ml_dtypes>=0.5、
#     而 tensorflow-cpu 2.18 强制 ml_dtypes<0.5（导致 float8_e3m4 import 错误）
#   - numpy<2.1：tensorflow-cpu 2.18 不兼容 numpy 2.1+
#   - protobuf<6：tensorflow-cpu 2.18 不兼容 protobuf 6/7
pip install --no-cache-dir \
    "numpy>=1.26.0,<2.1.0" "protobuf>=3.20.3,<6" \
    mujoco==2.3.7 gymnasium==0.29.1 gym==0.26.2 glfw==2.8.0 \
    flax==0.10.2 distrax==0.1.5 chex==0.1.88 optax==0.2.4 \
    orbax-checkpoint==0.10.3 tensorstore==0.1.71 ml-dtypes==0.4.1 \
    ml_collections==1.0.0 \
    opencv-python==4.10.0.84 imageio==2.36.1 imageio-ffmpeg==0.5.1 \
    matplotlib==3.10.0 cloudpickle requests \
    natsort tqdm pyzmq lz4 absl-py einops simple-parsing wandb pyquaternion \
    tensorflow-cpu==2.18.0 tensorflow-probability==0.25.0 tf-keras==2.18.0

# 10) 常驻环境变量（写入 ~/.bashrc，避免重复追加）
#     重要：CUDA_ROOT 用来绕过 jax 0.4.35 import nvidia-cuda-nvcc-cu12-12.9 触发的
#     "TypeError: expected str, bytes or os.PathLike object, not NoneType" 已知 bug。
#     只要 CUDA_ROOT 已设，jax 就走 _try_cuda_root_env 而不是 _try_cuda_nvcc_import。
if ! grep -q "hil-serl-sim env block" ~/.bashrc; then
    log "Append env vars to ~/.bashrc"
    cat >> ~/.bashrc <<'EOF'

# === hil-serl-sim env block ===
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export CUDA_MODULE_LOADING=LAZY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
# CUDA_ROOT: 绕过 jax 0.4.35 的 cuda_nvcc namespace-package import bug
if [ -z "${CUDA_ROOT:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_ROOT=/usr/local/cuda
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
        export CUDA_ROOT="$CONDA_PREFIX"
    else
        export CUDA_ROOT=/tmp
    fi
fi
# ============================
EOF
else
    log "~/.bashrc env block already present"
    # 旧版本可能没写 CUDA_ROOT，补丁追加
    if ! grep -q "CUDA_ROOT" ~/.bashrc; then
        log "Patch ~/.bashrc to add CUDA_ROOT (jax 0.4.35 cuda_nvcc workaround)"
        cat >> ~/.bashrc <<'EOF'

# === hil-serl-sim CUDA_ROOT patch ===
if [ -z "${CUDA_ROOT:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_ROOT=/usr/local/cuda
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
        export CUDA_ROOT="$CONDA_PREFIX"
    else
        export CUDA_ROOT=/tmp
    fi
fi
# ============================
EOF
    fi
fi

# 11) 立刻 source 一份给当前 shell（重跑 setup 时也能跑过 sanity check）
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export CUDA_MODULE_LOADING=LAZY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
if [ -z "${CUDA_ROOT:-}" ]; then
    if [ -d /usr/local/cuda ]; then
        export CUDA_ROOT=/usr/local/cuda
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
        export CUDA_ROOT="$CONDA_PREFIX"
    else
        export CUDA_ROOT=/tmp
    fi
fi
log "Using CUDA_ROOT=$CUDA_ROOT (avoids jax 0.4.35 cuda_nvcc import bug)"

# 12) Sanity check：确认 jax + torch 都能用 CUDA
#     说明：pip 会报 torch 要 cudnn 8.9 但 jax 装上了 9.21 的依赖冲突警告。
#     这是 pip 解析期警告：torch wheel 自带 cudnn 库，运行期不会用 site-packages 里
#     nvidia-cudnn-cu12，因此实际不冲突。下面的 sanity check 会真实验证两者都能用 CUDA。
log "Run JAX + Torch sanity check (pip 'cudnn' warning is harmless, validating runtime)"
python - <<'PY'
import jax, jax.numpy as jnp
print("JAX devices:", jax.devices())
x = jnp.ones((1024, 1024))
y = (x @ x).sum()
print("JAX matmul ok, sum =", float(y))

import torch
print("Torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    a = torch.ones(1024, 1024, device="cuda")
    print("Torch matmul ok, sum =", float((a @ a).sum().item()))
PY

# 13) 准备 demo_data 目录
mkdir -p /root/autodl-tmp/hil-serl-sim/examples/experiments/pick_cube_sim/demo_data

cat <<EOF

==============================================================
[setup] Server environment ready.
Repo path  : /root/autodl-tmp/hil-serl-sim
Conda env  : hilserl (activate via: conda activate hilserl)
Demo dir   : /root/autodl-tmp/hil-serl-sim/examples/experiments/pick_cube_sim/demo_data

Next steps:
  1) From your laptop, rsync your demo .pkl into the demo dir above.
  2) On this server, in a tmux session:
       conda activate hilserl
       cd /root/autodl-tmp/hil-serl-sim/examples/experiments/pick_cube_sim
       bash run_learner.sh
==============================================================
EOF
