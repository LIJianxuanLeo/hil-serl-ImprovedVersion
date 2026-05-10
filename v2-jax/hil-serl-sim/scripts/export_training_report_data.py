#!/usr/bin/env python3
"""
从一次 RLPD 训练 run 目录导出写报告用的结构化数据（CSV / JSON / 简短中文摘要）。

说明：若训练时 WANDB_MODE=offline/disabled 或 WANDB_DISABLED=1，WandBLogger 为 no-op，
**不会**留下 critic/actor loss、episode return 等时间序列；此类曲线无法事后从磁盘恢复。
本脚本会导出：checkpoint 时间线、（若有）本地 buffer 落盘快照、超参数、磁盘占用等。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "examples"))

try:
    from experiments.mappings import CONFIG_MAPPING  # noqa: E402
except Exception:  # noqa: BLE001
    CONFIG_MAPPING = {}

# 当本机未安装 jax/gymnasium 等、无法 import 任务 config 时，用于报告导出的只读副本（须与
# examples/experiments/<exp>/config.py + DefaultTrainingConfig 保持同步）。
STATIC_REPORT_HPARAMS: dict[str, dict] = {
    "pick_cube_sim": {
        "_note": "静态回退：与 DefaultTrainingConfig + pick_cube_sim.TrainConfig 一致时可用",
        "agent": "drq",
        "max_traj_length": 100,
        "batch_size": 256,
        "cta_ratio": 2,
        "discount": 0.97,
        "max_steps": 1_000_000,
        "replay_buffer_capacity": 200_000,
        "random_steps": 0,
        "training_starts": 100,
        "steps_per_update": 50,
        "log_period": 10,
        "eval_period": 2000,
        "encoder_type": "resnet-pretrained",
        "checkpoint_period": 5000,
        "buffer_period": 1000,
        "image_keys": ["wrist_1", "wrist_2"],
        "classifier_keys": ["wrist_1", "wrist_2"],
        "proprio_keys": [
            "tcp_pose",
            "tcp_vel",
            "tcp_force",
            "tcp_torque",
            "gripper_pose",
        ],
        "setup_mode": "single-arm-learned-gripper",
    },
}


def _checkpoint_step(name: str) -> int | None:
    m = re.match(r"^checkpoint_(\d+)$", name)
    return int(m.group(1)) if m else None


def _class_public_attrs(cls) -> dict:
    out = {}
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        for k, v in base.__dict__.items():
            if k.startswith("_"):
                continue
            if callable(v) and not isinstance(v, (staticmethod, classmethod, property)):
                continue
            if isinstance(v, (staticmethod, classmethod)):
                continue
            if isinstance(v, property):
                continue
            out[k] = v
    return out


def collect_hyperparameters(exp_name: str) -> dict:
    if exp_name in CONFIG_MAPPING:
        cfg_cls = CONFIG_MAPPING[exp_name]
        raw = _class_public_attrs(cfg_cls)
    elif exp_name in STATIC_REPORT_HPARAMS:
        raw = dict(STATIC_REPORT_HPARAMS[exp_name])
    else:
        raise SystemExit(
            f"无法解析超参数：{exp_name!r} 不在 CONFIG_MAPPING 中且无静态表。"
            "请在已安装依赖的环境中运行，或扩展 STATIC_REPORT_HPARAMS。"
        )
    serializable = {}
    for k, v in sorted(raw.items()):
        if callable(v) and not isinstance(v, type):
            continue
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, OverflowError):
            serializable[k] = repr(v)
    return serializable


def export_checkpoints(run_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(run_dir.glob("checkpoint_*")):
        if not p.is_dir():
            continue
        step = _checkpoint_step(p.name)
        if step is None:
            continue
        st = p.stat()
        rows.append(
            {
                "step": step,
                "dirname": p.name,
                "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "size_bytes": sum(f.stat().st_size for f in p.rglob("*") if f.is_file()),
            }
        )
    rows.sort(key=lambda r: r["step"])
    return rows


def export_buffer_snapshots(run_dir: Path, sub: str) -> list[dict]:
    d = run_dir / sub
    if not d.is_dir():
        return []
    rows = []
    for p in sorted(d.glob("transitions_*.pkl")):
        m = re.search(r"transitions_(\d+)\.pkl$", p.name)
        if not m:
            continue
        st = p.stat()
        rows.append(
            {
                "actor_step": int(m.group(1)),
                "file": str(p.relative_to(run_dir)),
                "size_bytes": st.st_size,
                "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_report_zh(
    out_dir: Path,
    exp_name: str,
    run_dir: Path,
    ckpt_rows: list[dict],
    hyper: dict,
    buffer_rows: list[dict],
    demo_buffer_rows: list[dict],
) -> None:
    lines = [
        "# 训练过程数据摘要（自动生成）",
        "",
        f"- **任务 / 实验名**: `{exp_name}`",
        f"- **Run 目录**: `{run_dir}`",
        f"- **导出时间**: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. 指标与曲线说明",
        "",
        "- 若训练时关闭了 WandB（`WANDB_DISABLED` / `WANDB_MODE=offline` 等），** learner 侧的 loss、Q 值、actor loss 等不会落盘**，本导出**不包含**这类时间序列。",
        "- 仍可用的材料：**checkpoint 保存步数时间线**、**超参数**、（若 actor 曾落盘）**replay / demo_buffer 周期性快照文件列表**。",
        "",
        "## 2. Checkpoint 时间线",
        "",
    ]
    if not ckpt_rows:
        lines.append("- （未发现 `checkpoint_*` 子目录）")
    else:
        first, last = ckpt_rows[0]["step"], ckpt_rows[-1]["step"]
        total_bytes = sum(r["size_bytes"] for r in ckpt_rows)
        lines.extend(
            [
                f"- **数量**: {len(ckpt_rows)} 个",
                f"- **步数范围**: {first} → {last}",
                f"- **checkpoint 占用（约）**: {total_bytes / (1024 * 1024):.1f} MiB（各目录文件合计）",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. 超参数（节选）",
            "",
            "完整键值见 `hyperparameters.json`。常见字段：",
            "",
        ]
    )
    for key in (
        "max_steps",
        "batch_size",
        "training_starts",
        "checkpoint_period",
        "log_period",
        "steps_per_update",
        "cta_ratio",
        "discount",
        "replay_buffer_capacity",
        "encoder_type",
        "setup_mode",
    ):
        if key in hyper:
            lines.append(f"- `{key}`: `{hyper[key]}`")
    lines.extend(
        [
            "",
            "## 4. Actor 侧 buffer 落盘（若存在）",
            "",
            f"- `buffer/transitions_*.pkl` 快照条数: **{len(buffer_rows)}**",
            f"- `demo_buffer/transitions_*.pkl` 快照条数: **{len(demo_buffer_rows)}**",
            "",
            "明细见 `buffer_snapshots.csv` / `demo_buffer_snapshots.csv`。",
            "",
            "## 5. 报告撰写建议",
            "",
            "- 将 **最终评估**（成功率、轨迹数、随机性说明）单独写在「实验结果」一节；若曾跑 eval，可把命令与数字附在 `eval_results.json`（可手写编辑）。",
            "- 用 `checkpoints.csv` 做「训练进行了多少步、多久存一次盘」的说明；用超参数表做「实验设置」。",
            "",
        ]
    )
    (out_dir / "report_summary_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="训练 run 根目录（内含 checkpoint_*，可选 buffer/、demo_buffer/）",
    )
    ap.add_argument("--exp-name", default="pick_cube_sim", help="CONFIG_MAPPING 中的实验名")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认：run-dir 下的 report_export_<时间戳>）",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir 不存在: {run_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (run_dir / f"report_export_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_rows = export_checkpoints(run_dir)
    buf_rows = export_buffer_snapshots(run_dir, "buffer")
    demo_buf_rows = export_buffer_snapshots(run_dir, "demo_buffer")

    hyper = collect_hyperparameters(args.exp_name)

    write_csv(
        out_dir / "checkpoints.csv",
        ["step", "dirname", "mtime_iso", "size_bytes"],
        ckpt_rows,
    )
    write_csv(
        out_dir / "buffer_snapshots.csv",
        ["actor_step", "file", "size_bytes", "mtime_iso"],
        buf_rows,
    )
    write_csv(
        out_dir / "demo_buffer_snapshots.csv",
        ["actor_step", "file", "size_bytes", "mtime_iso"],
        demo_buf_rows,
    )

    manifest = {
        "exp_name": args.exp_name,
        "run_dir": str(run_dir),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_count": len(ckpt_rows),
        "first_step": ckpt_rows[0]["step"] if ckpt_rows else None,
        "last_step": ckpt_rows[-1]["step"] if ckpt_rows else None,
        "total_checkpoint_bytes": sum(r["size_bytes"] for r in ckpt_rows),
        "buffer_snapshot_files": len(buf_rows),
        "demo_buffer_snapshot_files": len(demo_buf_rows),
        "wandb_note": "若训练使用 WandB no-op，则无 loss/return 历史可导出，仅结构数据。",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "hyperparameters.json").write_text(
        json.dumps(hyper, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    eval_template = {
        "eval_checkpoint_step": None,
        "eval_n_trajs": None,
        "success_rate": None,
        "average_episode_time_success_seconds": None,
        "command_example": (
            "bash run_actor.sh --checkpoint_path=<run_name> "
            "--eval_checkpoint_step=<step> --eval_n_trajs=20"
        ),
        "notes_zh": "请填入你实际跑评估时的步数、轨迹数与成功率；策略默认可仍为随机采样。",
    }
    (out_dir / "eval_results.template.json").write_text(
        json.dumps(eval_template, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_report_zh(
        out_dir,
        args.exp_name,
        run_dir,
        ckpt_rows,
        hyper,
        buf_rows,
        demo_buf_rows,
    )

    print(f"[export_training_report_data] 已写入: {out_dir}")
    print("  - checkpoints.csv, run_manifest.json, hyperparameters.json")
    print("  - buffer_snapshots.csv, demo_buffer_snapshots.csv")
    print("  - report_summary_zh.md, eval_results.template.json")


if __name__ == "__main__":
    main()
