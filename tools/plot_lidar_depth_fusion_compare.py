#!/usr/bin/env python3
"""Plot training curves comparing Depth Camera vs LiDAR+Depth Fusion."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from thesis_plot_style import setup_chinese_matplotlib
except ImportError:  # pragma: no cover
    from tools.thesis_plot_style import setup_chinese_matplotlib

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tensorboard is required: python -m pip install tensorboard") from exc

setup_chinese_matplotlib()

# ── configuration ──────────────────────────────────────────────────────────
DEPTH_RUN = Path(
    "training_code/logs/depth_camera/diffphys/nominal/navigation/"
    "multi_agent_odom/20260530_034859"
)
FUSION_RUN = Path(
    "training_code/logs/lidar_depth_fusion/diffphys/nominal/navigation/"
    "single_agent_odom/20260531_104609"
)

LABEL_DEPTH = "深度相机 (Depth Camera)"
LABEL_FUSION = "激光雷达+深度相机 (LiDAR+Depth)"

COLOR_DEPTH = "#2196F3"  # blue
COLOR_FUSION = "#FF5722"  # deep orange

SMOOTH_WINDOW = 50  # moving-average window for smoother curves


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


def moving_average(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def tail_stats(scalars: list[tuple[int, float]], window: int) -> dict:
    """Compute mean/std over the last `window` data points."""
    vals = [v for _, v in scalars[-window:]] if window > 0 else [v for _, v in scalars]
    if not vals:
        return {"mean": None, "std": None}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std}


def plot_success(
    output_path: Path,
    depth_scalars: dict,
    fusion_scalars: dict,
    smooth: int,
):
    """Plot success/main rate comparison."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)

    for label, scalars, color in [
        (LABEL_DEPTH, depth_scalars, COLOR_DEPTH),
        (LABEL_FUSION, fusion_scalars, COLOR_FUSION),
    ]:
        values = scalars.get("success/main", [])
        if not values:
            continue
        xs = np.array([s for s, _ in values])
        ys = np.array([v for _, v in values])

        # Raw (faint)
        ax.plot(xs, ys, color=color, alpha=0.15, linewidth=0.6)

        # Smoothed
        if len(ys) >= smooth:
            xs_sm = xs[smooth - 1:]
            ys_sm = moving_average(ys, smooth)
            ax.plot(xs_sm, ys_sm, color=color, linewidth=2.0, label=label)

        # Annotate final value
        stats = tail_stats(values, smooth)
        if stats["mean"] is not None:
            final_pct = stats["mean"] * 100
            ax.axhline(y=stats["mean"], color=color, linestyle="--", alpha=0.4, linewidth=0.8)
            ax.text(
                xs[-1] + 200, stats["mean"],
                f"{final_pct:.1f}%",
                color=color, fontsize=9, va="center", fontweight="bold",
            )

    ax.set_xlabel("训练步数 (Training Steps)")
    ax.set_ylabel("成功率 (Success Rate)")
    ax.set_title("训练成功率对比: 深度相机 vs 激光雷达+深度相机融合")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_losses(
    output_path: Path,
    depth_scalars: dict,
    fusion_scalars: dict,
    smooth: int,
):
    """Plot key loss components."""
    loss_tags = [
        ("loss/total", "总损失 (Total)"),
        ("loss/collision", "碰撞损失 (Collision)"),
        ("loss/velocity_tracking", "速度跟踪损失 (Velocity Tracking)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)

    for ax, (tag, title) in zip(axes, loss_tags):
        for label, scalars, color in [
            (LABEL_DEPTH, depth_scalars, COLOR_DEPTH),
            (LABEL_FUSION, fusion_scalars, COLOR_FUSION),
        ]:
            values = scalars.get(tag, [])
            if not values:
                continue
            xs = np.array([s for s, _ in values])
            ys = np.array([v for _, v in values])

            ax.plot(xs, ys, color=color, alpha=0.12, linewidth=0.5)
            if len(ys) >= smooth:
                xs_sm = xs[smooth - 1:]
                ys_sm = moving_average(ys, smooth)
                ax.plot(xs_sm, ys_sm, color=color, linewidth=1.8, label=label)

        ax.set_title(title)
        ax.set_xlabel("训练步数")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("训练损失对比: 深度相机 vs 激光雷达+深度相机融合", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_performance(
    output_path: Path,
    depth_scalars: dict,
    fusion_scalars: dict,
    smooth: int,
):
    """Plot performance metrics: avg_speed and action ratio."""
    perf_tags = [
        ("performance/avg_speed", "平均速度 (Avg Speed, m/s)"),
        ("performance/ar", "动作比率 (Action Ratio)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    for ax, (tag, title) in zip(axes, perf_tags):
        for label, scalars, color in [
            (LABEL_DEPTH, depth_scalars, COLOR_DEPTH),
            (LABEL_FUSION, fusion_scalars, COLOR_FUSION),
        ]:
            values = scalars.get(tag, [])
            if not values:
                continue
            xs = np.array([s for s, _ in values])
            ys = np.array([v for _, v in values])

            ax.plot(xs, ys, color=color, alpha=0.12, linewidth=0.5)
            if len(ys) >= smooth:
                xs_sm = xs[smooth - 1:]
                ys_sm = moving_average(ys, smooth)
                ax.plot(xs_sm, ys_sm, color=color, linewidth=1.8, label=label)

        ax.set_title(title)
        ax.set_xlabel("训练步数")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("性能指标对比: 深度相机 vs 激光雷达+深度相机融合", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_endpoint_stats(
    output_path: Path,
    depth_scalars: dict,
    fusion_scalars: dict,
    stats_window: int,
):
    """Bar chart comparing final success rate and key metrics."""
    metrics = [
        ("success/main", "成功率\n(Success Rate)", 100),  # scale to percentage
        ("loss/total", "总损失\n(Total Loss)", 1),
        ("loss/collision", "碰撞损失\n(Collision Loss)", 1),
        ("performance/avg_speed", "平均速度 m/s\n(Avg Speed)", 1),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), dpi=180)
    x = np.arange(2)
    width = 0.5

    for ax, (tag, title, scale) in zip(axes, metrics):
        depth_stats = tail_stats(depth_scalars.get(tag, []), stats_window)
        fusion_stats = tail_stats(fusion_scalars.get(tag, []), stats_window)

        depth_mean = (depth_stats["mean"] or 0) * scale
        fusion_mean = (fusion_stats["mean"] or 0) * scale
        depth_std = (depth_stats["std"] or 0) * scale
        fusion_std = (fusion_stats["std"] or 0) * scale

        bars = ax.bar(
            x,
            [depth_mean, fusion_mean],
            width,
            yerr=[depth_std, fusion_std],
            color=[COLOR_DEPTH, COLOR_FUSION],
            capsize=6,
            edgecolor="white",
            linewidth=0.8,
        )

        # Annotate values on bars
        for bar, val in zip(bars, [depth_mean, fusion_mean]):
            fmt = f"{val:.1f}" if scale == 100 else f"{val:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (depth_std + fusion_std) * 0.05 + 0.02,
                fmt, ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["深度相机", "激光雷达\n+深度相机"], fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("训练终值对比 (最后200步均值 ± 标准差)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="thesis_outputs/lidar_depth_fusion_compare",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTH_WINDOW,
        help="Moving average window size",
    )
    parser.add_argument(
        "--stats_window",
        type=int,
        default=200,
        help="Number of final samples for bar-chart stats",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    depth_scalars = load_scalars(DEPTH_RUN)
    fusion_scalars = load_scalars(FUSION_RUN)

    # Print summary stats
    for label, scalars in [("Depth Camera", depth_scalars), ("LiDAR+Depth Fusion", fusion_scalars)]:
        stats = tail_stats(scalars.get("success/main", []), args.stats_window)
        print(
            f"{label}: success/main final {args.stats_window}-step "
            f"mean={stats['mean']:.4f} ({stats['mean']*100:.1f}%) "
            f"std={stats['std']:.4f}"
        )

    # Plot
    plot_success(figures_dir / "training_lidar_depth_success_compare.png", depth_scalars, fusion_scalars, args.smooth)
    plot_losses(figures_dir / "training_lidar_depth_losses_compare.png", depth_scalars, fusion_scalars, args.smooth)
    plot_performance(figures_dir / "training_lidar_depth_performance_compare.png", depth_scalars, fusion_scalars, args.smooth)
    plot_endpoint_stats(figures_dir / "training_lidar_depth_endpoint_stats.png", depth_scalars, fusion_scalars, args.stats_window)

    # Save CSV and JSON
    all_scalars = {"depth_camera": depth_scalars, "lidar_depth_fusion": fusion_scalars}
    import csv
    with open(output_dir / "training_compare_scalars.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["run", "tag", "step", "value"])
        writer.writeheader()
        for run, scalars in all_scalars.items():
            for tag, values in scalars.items():
                for step, value in values:
                    writer.writerow({"run": run, "tag": tag, "step": step, "value": f"{value:.8g}"})

    summary = {}
    for run, scalars in all_scalars.items():
        summary[run] = {
            tag: tail_stats(scalars.get(tag, []), args.stats_window)
            for tag in [
                "success/main", "loss/total", "loss/collision",
                "loss/velocity_tracking", "performance/avg_speed",
                "performance/ar",
            ]
        }
    with open(output_dir / "training_compare_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True, default=str)

    print(f"\nDone. Output: {output_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
