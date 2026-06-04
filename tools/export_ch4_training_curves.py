#!/usr/bin/env python3
"""Export Chapter 4 training-error curves from TensorBoard logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from thesis_plot_style import setup_chinese_matplotlib
except ImportError:  # pragma: no cover
    from tools.thesis_plot_style import setup_chinese_matplotlib

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tensorboard is required: python -m pip install tensorboard") from exc


DEFAULT_RUNS = {
    "Original": "training_code/logs/depth_camera/single_agent_odom/20260504_013822_random_target_original_loss",
    "RTH": "training_code/logs/depth_camera/single_agent_odom/20260504_122031_rth_dp_depth_odom",
    "RTH+DP": "training_code/logs/depth_camera/rth/nominal/target_hover/single_agent_odom/20260530_034815_rth_normal_target_hover",
}

METRICS = {
    "ch4_training_final_error.png": ("goal/final_error", "终点目标误差", "误差 / m"),
    "ch4_training_min_error.png": ("goal/min_error", "最小目标误差", "误差 / m"),
    "ch4_training_hover_position.png": ("hover/position_error", "悬停窗口位置误差", "误差 / m"),
    "ch4_training_hover_velocity.png": ("hover/velocity_error", "悬停窗口速度误差", "速度 / (m/s)"),
}

COLORS = {
    "Original": "#4C78A8",
    "RTH": "#54A24B",
    "RTH+DP": "#F58518",
}


setup_chinese_matplotlib()


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    events = sorted((run_dir / "tb").glob("events.out.tfevents*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file found under {run_dir / 'tb'}")
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = [(int(ev.step), float(ev.value)) for ev in acc.Scalars(tag)]
    return out


def smooth(values: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1 or len(values) <= 2:
        return values
    half = window // 2
    smoothed: list[tuple[int, float]] = []
    ys = [value for _, value in values]
    for idx, (step, _) in enumerate(values):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        smoothed.append((step, sum(ys[start:end]) / (end - start)))
    return smoothed


def write_metric_csv(output_dir: Path, all_scalars: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
    with open(output_dir / "ch4_training_curves.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["run", "tag", "step", "value"])
        writer.writeheader()
        for run, scalars in all_scalars.items():
            for tag, _, _ in METRICS.values():
                for step, value in scalars.get(tag, []):
                    writer.writerow({"run": run, "tag": tag, "step": step, "value": f"{value:.8g}"})


def plot_metric(
    output_path: Path,
    all_scalars: dict[str, dict[str, list[tuple[int, float]]]],
    tag: str,
    title: str,
    ylabel: str,
    smooth_window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.55), dpi=300)
    for run, scalars in all_scalars.items():
        values = smooth(scalars.get(tag, []), smooth_window)
        if not values:
            continue
        xs = [step for step, _ in values]
        ys = [value for _, value in values]
        ax.plot(xs, ys, label=run, color=COLORS.get(run), linewidth=1.55)

    ax.set_title(title)
    ax.set_xlabel("训练步数")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.75)
    ax.legend(ncol=3, loc="upper right", frameon=False, handlelength=2.4)
    ax.margins(x=0)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_runs(items: list[str]) -> dict[str, str]:
    runs = dict(DEFAULT_RUNS)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--run must be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        runs[name] = path
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        default="/home/ember/桌面/thesis/thesis-latex/Figures",
        help="Directory where the four Chapter 4 PNG figures are written.",
    )
    parser.add_argument("--output-dir", default="thesis_outputs/ch4_training_curves")
    parser.add_argument("--smooth-window", type=int, default=25)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override/add a run. Can be passed multiple times.",
    )
    args = parser.parse_args()

    runs = parse_runs(args.run)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scalars = {name: load_scalars(Path(path)) for name, path in runs.items()}
    write_metric_csv(output_dir, all_scalars)

    for filename, (tag, title, ylabel) in METRICS.items():
        plot_metric(figures_dir / filename, all_scalars, tag, title, ylabel, args.smooth_window)

    print(f"Wrote CSV to {output_dir / 'ch4_training_curves.csv'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
