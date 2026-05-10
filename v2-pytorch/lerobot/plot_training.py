#!/usr/bin/env python3
"""
训练日志可视化工具

使用方法：
    python plot_training.py [output_dir]

示例：
    python plot_training.py outputs/train/franka_sim_sac_touch
    python plot_training.py  # 自动查找最新的训练目录

生成文件：
    {output_dir}/training_logs/training_curves.png
"""

import csv
import json
import os
import sys
from pathlib import Path


def find_latest_output_dir():
    """Find the most recent training output directory."""
    output_base = Path("outputs/train")
    if not output_base.exists():
        return None
    candidates = []
    for d in output_base.iterdir():
        log_dir = d / "training_logs"
        if log_dir.exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def read_csv(filepath):
    """Read CSV file into list of dicts."""
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return list(csv.DictReader(f))


def safe_float(val):
    """Convert to float, return None if empty or invalid."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def plot_training(output_dir):
    """Plot training curves from CSV logs."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.size'] = 10
    except ImportError:
        print("Error: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    log_dir = os.path.join(output_dir, "training_logs")
    training_csv = os.path.join(log_dir, "training_metrics.csv")
    episode_csv = os.path.join(log_dir, "episode_metrics.csv")
    summary_json = os.path.join(log_dir, "training_summary.json")

    # Read data
    training_data = read_csv(training_csv)
    episode_data = read_csv(episode_csv)

    if not training_data and not episode_data:
        print(f"No training data found in {log_dir}")
        sys.exit(1)

    print(f"Training data: {len(training_data)} rows")
    print(f"Episode data: {len(episode_data)} rows")

    # Read summary if available
    if os.path.exists(summary_json):
        with open(summary_json) as f:
            summary = json.load(f)
        print(f"\n--- Training Summary ---")
        for k, v in summary.items():
            if k != "log_files":
                print(f"  {k}: {v}")
        print()

    # Extract training metrics
    t_steps = [safe_float(r.get("optimization_step")) for r in training_data]
    t_loss_critic = [safe_float(r.get("loss_critic")) for r in training_data]
    t_loss_actor = [safe_float(r.get("loss_actor")) for r in training_data]
    t_temperature = [safe_float(r.get("temperature")) for r in training_data]
    t_buffer = [safe_float(r.get("replay_buffer_size")) for r in training_data]

    # Extract episode metrics
    e_steps = [safe_float(r.get("interaction_step")) for r in episode_data]
    e_reward = [safe_float(r.get("episodic_reward")) for r in episode_data]
    e_intervention = [safe_float(r.get("intervention_rate")) for r in episode_data]

    # Determine subplot layout
    has_training = any(v is not None for v in t_loss_critic)
    has_episodes = any(v is not None for v in e_reward)

    n_plots = 0
    if has_training:
        n_plots += 3  # loss, temperature, buffer
    if has_episodes:
        n_plots += 2  # reward, intervention

    if n_plots == 0:
        print("No valid metrics to plot.")
        sys.exit(1)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(f'Training Curves: {os.path.basename(output_dir)}', fontsize=13, fontweight='bold')

    idx = 0

    # 1. Critic & Actor Loss
    if has_training:
        ax = axes[idx]; idx += 1
        valid = [(s, v) for s, v in zip(t_steps, t_loss_critic) if s is not None and v is not None]
        if valid:
            steps, vals = zip(*valid)
            ax.plot(steps, vals, color='#E74C3C', linewidth=0.8, alpha=0.6, label='Critic Loss')
            # Smoothed
            if len(vals) > 20:
                window = max(len(vals) // 50, 5)
                smoothed = _moving_avg(vals, window)
                ax.plot(steps[:len(smoothed)], smoothed, color='#E74C3C', linewidth=2, label=f'Critic (avg {window})')
        valid_a = [(s, v) for s, v in zip(t_steps, t_loss_actor) if s is not None and v is not None]
        if valid_a:
            steps_a, vals_a = zip(*valid_a)
            ax.plot(steps_a, vals_a, color='#3498DB', linewidth=0.8, alpha=0.6, label='Actor Loss')
        ax.set_title('Training Loss')
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 2. Temperature
    if has_training:
        ax = axes[idx]; idx += 1
        valid = [(s, v) for s, v in zip(t_steps, t_temperature) if s is not None and v is not None]
        if valid:
            steps, vals = zip(*valid)
            ax.plot(steps, vals, color='#9B59B6', linewidth=1.5)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial (1.0)')
        ax.set_title('SAC Temperature')
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Temperature (alpha)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 3. Buffer Size
    if has_training:
        ax = axes[idx]; idx += 1
        valid = [(s, v) for s, v in zip(t_steps, t_buffer) if s is not None and v is not None]
        if valid:
            steps, vals = zip(*valid)
            ax.plot(steps, [v/1000 for v in vals], color='#2ECC71', linewidth=1.5)
        ax.set_title('Replay Buffer Size')
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Size (K)')
        ax.grid(True, alpha=0.3)

    # 4. Episodic Reward
    if has_episodes:
        ax = axes[idx]; idx += 1
        valid = [(s, v) for s, v in zip(e_steps, e_reward) if s is not None and v is not None]
        if valid:
            steps, vals = zip(*valid)
            ax.scatter(steps, vals, s=8, alpha=0.3, color='#F39C12', label='Episode Reward')
            # Smoothed
            if len(vals) > 10:
                window = max(len(vals) // 30, 5)
                smoothed = _moving_avg(vals, window)
                ax.plot(steps[:len(smoothed)], smoothed, color='#E67E22', linewidth=2.5, label=f'Avg ({window} ep)')
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max Reward')
        ax.set_title('Episodic Reward')
        ax.set_xlabel('Interaction Step')
        ax.set_ylabel('Reward')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 5. Intervention Rate
    if has_episodes:
        ax = axes[idx]; idx += 1
        valid = [(s, v) for s, v in zip(e_steps, e_intervention) if s is not None and v is not None]
        if valid:
            steps, vals = zip(*valid)
            ax.scatter(steps, [v * 100 for v in vals], s=8, alpha=0.3, color='#E74C3C', label='Per Episode')
            if len(vals) > 10:
                window = max(len(vals) // 30, 5)
                smoothed = _moving_avg([v * 100 for v in vals], window)
                ax.plot(steps[:len(smoothed)], smoothed, color='#C0392B', linewidth=2.5, label=f'Avg ({window} ep)')
        ax.set_title('Intervention Rate')
        ax.set_xlabel('Interaction Step')
        ax.set_ylabel('Rate (%)')
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(log_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # Also save PDF
    pdf_path = os.path.join(log_dir, "training_curves.pdf")
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {pdf_path}")


def _moving_avg(data, window):
    """Simple moving average."""
    result = []
    for i in range(len(data) - window + 1):
        avg = sum(data[i:i+window]) / window
        result.append(avg)
    return result


def main():
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = find_latest_output_dir()
        if output_dir is None:
            print("Usage: python plot_training.py <output_dir>")
            print("  e.g.: python plot_training.py outputs/train/franka_sim_sac_touch")
            print("\nNo training output directory found in outputs/train/")
            sys.exit(1)
        print(f"Auto-detected latest training: {output_dir}")

    if not os.path.exists(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    plot_training(str(output_dir))


if __name__ == "__main__":
    main()
