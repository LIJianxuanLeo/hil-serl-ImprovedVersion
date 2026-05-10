# 在激活 conda 后: source 本文件（路径按你本机仓库修改）
# 解决 JAX 0.4.35 在 pip CUDA 元包下 import 失败；有本机 CUDA 时应指向真实路径，避免运行期选错 cuDNN。
if [ -n "${CUDA_ROOT:-}" ]; then
  :
elif [ -d /usr/local/cuda ]; then
  export CUDA_ROOT=/usr/local/cuda
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
  export CUDA_ROOT="${CONDA_PREFIX}"
else
  export CUDA_ROOT=/tmp
fi
