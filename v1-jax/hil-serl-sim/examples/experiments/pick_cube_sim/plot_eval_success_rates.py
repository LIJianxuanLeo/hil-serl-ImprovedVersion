#!/usr/bin/env python3
"""从 eval_success_rates_*.csv 画成功率随 checkpoint_step 曲线。

默认输出同名 .svg（纯标准库，无需 matplotlib）。若已安装 matplotlib，可用 --format png。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys


def _write_svg(
    out_path: str,
    steps: list[int],
    rates: list[float],
    n_trajs: str,
    title_extra: str = "",
) -> None:
    W, H = 800, 450
    margin_l, margin_r, margin_t, margin_b = 72, 40, 48, 56
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{W // 2}" y="28" text-anchor="middle" font-size="16" '
        f'font-family="system-ui,sans-serif" fill="#111">'
        f'{esc("Pick-cube sim — eval success rate")}{esc(title_extra)}</text>',
    ]

    if len(steps) == 0:
        parts.append(
            f'<text x="{W // 2}" y="{H // 2}" text-anchor="middle" font-size="13" '
            f'font-family="system-ui,sans-serif" fill="#b45309">'
            f'{esc("无有效数据：CSV 中需 status=ok 且 success_rate 有数值。")}</text>'
            f'<text x="{W // 2}" y="{H // 2 + 22}" text-anchor="middle" font-size="12" '
            f'font-family="system-ui,sans-serif" fill="#64748b">'
            f'{esc("常见原因：评估未在含 JAX 的 conda 环境中运行（error_exit_1）。")}</text>'
        )
        parts.append("</svg>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        return

    xmin, xmax = min(steps), max(steps)
    ymin, ymax = 0.0, 1.0
    if xmin == xmax:
        xmin -= 5000
        xmax += 5000

    def x_px(s: int) -> float:
        return margin_l + (s - xmin) / (xmax - xmin) * plot_w

    def y_px(r: float) -> float:
        return margin_t + (ymax - r) / (ymax - ymin) * plot_h

    # grid
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = y_px(g)
        parts.append(
            f'<line x1="{margin_l}" y1="{gy}" x2="{W - margin_r}" y2="{gy}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_l - 8}" y="{gy + 4}" text-anchor="end" font-size="11" '
            f'font-family="system-ui,sans-serif" fill="#64748b">{g:.2f}</text>'
        )

    # axes
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{W - margin_r}" '
        f'y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" '
        f'y2="{margin_t + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>'
    )

    # polyline
    pts = " ".join(f"{x_px(s):.1f},{y_px(r):.1f}" for s, r in zip(steps, rates))
    parts.append(
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{pts}"/>'
    )
    for s, r in zip(steps, rates):
        cx, cy = x_px(s), y_px(r)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#2563eb" stroke="#fff" stroke-width="1"/>'
        )

    # x labels (sparse)
    nlab = min(12, len(steps))
    idxs = [round(i * (len(steps) - 1) / max(nlab - 1, 1)) for i in range(nlab)] if len(steps) > 1 else [0]
    seen = set()
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        s = steps[i]
        lx = x_px(s)
        parts.append(
            f'<text x="{lx}" y="{H - 18}" text-anchor="middle" font-size="10" '
            f'font-family="system-ui,sans-serif" fill="#334155">{s}</text>'
        )

    ylab = "Success rate" + (f" (n={n_trajs})" if n_trajs else "")
    parts.append(
        f'<text x="18" y="{margin_t + plot_h / 2}" text-anchor="middle" '
        f'font-size="12" font-family="system-ui,sans-serif" fill="#334155" '
        f'transform="rotate(-90 18 {margin_t + plot_h / 2})">{esc(ylab)}</text>'
    )
    parts.append(
        f'<text x="{W // 2}" y="{H - 4}" text-anchor="middle" font-size="12" '
        f'font-family="system-ui,sans-serif" fill="#334155">Training step (checkpoint)</text>'
    )

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="eval_all_checkpoints.py 生成的 CSV")
    ap.add_argument(
        "--out",
        default="",
        help="输出路径；默认与 CSV 同目录、同名 .svg 或 .png",
    )
    ap.add_argument(
        "--format",
        choices=("svg", "png", "auto"),
        default="auto",
        help="auto：优先 png（若已装 matplotlib），否则 svg",
    )
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv_path)
    if not os.path.isfile(csv_path):
        print(f"文件不存在: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    steps: list[int] = []
    rates: list[float] = []
    for r in rows:
        if r.get("status", "").strip() != "ok":
            continue
        try:
            sr = float(r.get("success_rate", "") or "")
            st = int(r["checkpoint_step"])
        except (ValueError, KeyError):
            continue
        steps.append(st)
        rates.append(sr)

    n_trajs = rows[0].get("n_trajs", "").strip() if rows else ""

    fmt = args.format
    if fmt == "auto":
        try:
            import matplotlib  # noqa: F401

            fmt = "png"
        except ImportError:
            fmt = "svg"

    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(csv_path)[0]
        out_path = base + (".png" if fmt == "png" else ".svg")

    title_extra = f" — {n_trajs} ep/chkpt" if n_trajs else ""

    if fmt == "png":
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("未安装 matplotlib，改用 SVG。", file=sys.stderr)
            out_path = os.path.splitext(out_path)[0] + ".svg"
            _write_svg(out_path, steps, rates, n_trajs, title_extra)
            print(f"已保存: {out_path}")
            return

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        if len(steps) == 0:
            ax.text(
                0.5,
                0.5,
                "无有效数据：请检查 CSV 中 status==ok 且 success_rate 有数值。\n"
                "常见原因：评估未在含 JAX 的 conda 环境中运行（error_exit_1）。",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_axis_off()
        else:
            ax.plot(steps, rates, "o-", color="#2563eb", linewidth=2, markersize=6)
            ax.set_xlabel("Training step (checkpoint)")
            ax.set_ylabel(f"Success rate ({n_trajs} eval episodes / checkpoint)")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(steps)
            ax.grid(True, alpha=0.3)
            ax.set_title("Pick-cube sim eval")

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    else:
        _write_svg(out_path, steps, rates, n_trajs, title_extra)

    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
