#!/usr/bin/env python3
"""
对 run 目录下每个 checkpoint_* 加载策略，在仿真中跑 N 局，统计成功率，导出 CSV。

用法（在 pick_cube_sim 目录下）:
  conda activate hilserl_sim
  python eval_all_checkpoints.py --run_dir pick_cube_sim_20260501_170623 --n_trajs 20

说明:
  - buffer/ 下是 replay 数据，不是模型；本脚本只遍历 checkpoint_*。
  - 若子进程秒退且 JAX 报 nvidia.cuda_nvcc / __file__ 为 None：脚本会自动为子进程设置
    CUDA_ROOT=.../site-packages/nvidia/cuda_nvcc（若未手动设置 CUDA_ROOT）。
  - 若报 Checkpoint structure file / orbax 相关错误：请与本仓库 requirements.txt 对齐，例如:
      pip install 'flax==0.10.2' 'orbax-checkpoint==0.10.3'
    训练机与本机 flax/orbax 版本不一致会导致无法加载 checkpoint。
  - 可选: --python /path/to/envs/hilserl_sim/bin/python 显式指定解释器。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from datetime import datetime, timezone


def detect_nvidia_cuda_nvcc_root(py_exe: str) -> str | None:
    """JAX 会 import nvidia.cuda_nvcc，其 __file__ 为 None，需在子进程里设置 CUDA_ROOT。"""
    if os.environ.get("CUDA_ROOT"):
        return None
    code = (
        "import os, site\n"
        "for sp in site.getsitepackages():\n"
        " p = os.path.join(sp, 'nvidia', 'cuda_nvcc')\n"
        " if os.path.isfile(os.path.join(p, 'bin', 'ptxas')):\n"
        "  print(p); break\n"
    )
    r = subprocess.run(
        [py_exe, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return r.stdout.strip().splitlines()[-1]


def build_child_env(base: dict[str, str], py_exe: str) -> dict[str, str]:
    env = dict(base)
    cuda_root = detect_nvidia_cuda_nvcc_root(py_exe)
    if cuda_root:
        env.setdefault("CUDA_ROOT", cuda_root)
    return env


def discover_checkpoint_steps(run_dir: str) -> list[int]:
    steps: list[int] = []
    pattern = os.path.join(os.path.abspath(run_dir), "checkpoint_*")
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        base = os.path.basename(path)
        m = re.match(r"checkpoint_(\d+)$", base)
        if m:
            steps.append(int(m.group(1)))
    steps.sort()
    return steps


def parse_eval_stdout(text: str) -> tuple[float | None, float | None]:
    """从 train_rlpd 评估分支的 print 中解析 success rate 与 average time。"""
    sr, at = None, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("success rate:"):
            try:
                sr = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        if line.startswith("average time:"):
            try:
                at = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return sr, at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        default="pick_cube_sim_20260501_170623",
        help="含 checkpoint_* 子目录的 run 路径（相对当前工作目录或绝对路径）",
    )
    ap.add_argument("--n_trajs", type=int, default=20, help="每个 checkpoint 评估的回合数")
    ap.add_argument(
        "--only_steps",
        default="",
        help="只评估这些训练步（逗号分隔），如 5000,10000；留空则评估目录下全部 checkpoint",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="输出 CSV 路径；默认写到 run_dir/eval_success_rates_<timestamp>.csv",
    )
    ap.add_argument(
        "--python",
        default="",
        help="用于运行 train_rlpd 的解释器；默认与当前脚本相同 (sys.executable)，"
        "务必使用含 JAX 的环境，例如 .../envs/hilserl_sim/bin/python",
    )
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    py_exe = args.python.strip() or sys.executable

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"run_dir 不存在: {run_dir}", file=sys.stderr)
        sys.exit(1)

    steps = discover_checkpoint_steps(run_dir)
    if not steps:
        print(f"未在 {run_dir} 下找到 checkpoint_* 目录", file=sys.stderr)
        sys.exit(1)
    if args.only_steps.strip():
        want = {int(x.strip()) for x in args.only_steps.split(",") if x.strip()}
        steps = [s for s in steps if s in want]
        missing = want - set(steps)
        if missing:
            print(f"警告: 以下步数在 run 目录中不存在: {sorted(missing)}", file=sys.stderr)
        if not steps:
            print("过滤后没有可评估的 checkpoint", file=sys.stderr)
            sys.exit(1)

    out_csv = args.out_csv or os.path.join(
        run_dir,
        f"eval_success_rates_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
    )

    train_rlpd = os.path.join(script_dir, "..", "..", "train_rlpd.py")
    train_rlpd = os.path.normpath(train_rlpd)

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env = build_child_env(env, py_exe)

    rows: list[dict] = []
    for step in steps:
        cmd = [
            py_exe,
            train_rlpd,
            "--exp_name=pick_cube_sim",
            "--actor",
            f"--checkpoint_path={run_dir}",
            f"--eval_checkpoint_step={step}",
            f"--eval_n_trajs={args.n_trajs}",
            "--debug",
        ]
        print(f"\n=== checkpoint step {step} ({args.n_trajs} episodes) ===", flush=True)
        p = subprocess.run(
            cmd,
            cwd=script_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        out = p.stdout + "\n" + p.stderr
        if p.returncode != 0:
            print(out[-4000:], file=sys.stderr)
            rows.append(
                {
                    "checkpoint_step": step,
                    "success_rate": "",
                    "avg_episode_time_s": "",
                    "n_trajs": args.n_trajs,
                    "status": f"error_exit_{p.returncode}",
                }
            )
            continue
        sr, at = parse_eval_stdout(out)
        rows.append(
            {
                "checkpoint_step": step,
                "success_rate": sr if sr is not None else "",
                "avg_episode_time_s": at if at is not None else "",
                "n_trajs": args.n_trajs,
                "status": "ok" if sr is not None else "parse_failed",
            }
        )
        print(f"success_rate={sr}  avg_time={at}", flush=True)

    fieldnames = [
        "checkpoint_step",
        "success_rate",
        "avg_episode_time_s",
        "n_trajs",
        "status",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n已写入: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
