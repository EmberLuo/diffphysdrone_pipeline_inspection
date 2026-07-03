#!/usr/bin/env python3
"""Lidar sensor-noise robustness: clean lidar vs realistic lidar (noise+dropout DR).

Both are lidar-only MULTI runs, 50k steps, same improved config
(vfov [-45,+20], range 6, vbeams 16). The realistic run additionally applies
per-env range noise (std up to 3cm) and per-beam dropout (up to 10%). The depth
baseline is drawn for context.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from thesis_plot_style import setup_chinese_matplotlib
except ImportError:
    from tools.thesis_plot_style import setup_chinese_matplotlib

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

setup_chinese_matplotlib()

LOG = Path("training_code/logs")
CLEAN = LOG / "lidar_navrl/diffphys/nominal/navigation/multi_agent_odom/20260704_000743_lidaronly_multi_lidarcfg_v2"
REALISTIC = LOG / "lidar_navrl/diffphys/nominal/navigation/multi_agent_odom/20260704_035128_lidaronly_multi_realistic"
DEPTH = LOG / "depth_camera/diffphys/nominal/navigation/multi_agent_odom/20260530_034859"

C_CLEAN = "#4CAF50"
C_REAL = "#9C27B0"
C_DEPTH = "#2196F3"
SMOOTH = 40


def scalars(run: Path, tag: str = "success/main"):
    ev = sorted((run / "tb").glob("events*"))
    if not ev:
        return []
    a = EventAccumulator(str(ev[-1]), size_guidance={"scalars": 0})
    a.Reload()
    if tag not in a.Tags().get("scalars", []):
        return []
    return [(int(s.step), float(s.value)) for s in a.Scalars(tag)]


def smooth(ys, w):
    return np.convolve(ys, np.ones(w) / w, mode="valid") if len(ys) >= w else np.array(ys)


def tail(series, n=200):
    vals = [v for _, v in series[-n:]]
    return (statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)) if vals else (None, None)


def main():
    out = Path("thesis_outputs/lidar_depth_fusion_compare")
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    series = {
        "depth_baseline": scalars(DEPTH),
        "lidar_clean": scalars(CLEAN),
        "lidar_realistic": scalars(REALISTIC),
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=180)
    for key, label, color in [
        ("depth_baseline", "深度相机基线 (Depth baseline)", C_DEPTH),
        ("lidar_clean", "激光雷达-理想 (clean, no noise)", C_CLEAN),
        ("lidar_realistic", "激光雷达-真实感 (noise+dropout DR)", C_REAL),
    ]:
        s = series[key]
        if not s:
            continue
        xs = np.array([x for x, _ in s])
        ys = np.array([y for _, y in s])
        ax.plot(xs, ys, color=color, alpha=0.12, linewidth=0.6)
        if len(ys) >= SMOOTH:
            ax.plot(xs[SMOOTH - 1:], smooth(ys, SMOOTH), color=color, linewidth=2.0, label=label)
        m, _ = tail(s)
        if m is not None:
            ax.axhline(m, color=color, linestyle="--", alpha=0.35, linewidth=0.8)
            ax.text(xs[-1] + 300, m, f"{m*100:.1f}%", color=color, fontsize=9,
                    va="center", fontweight="bold")

    ax.set_xlabel("训练步数 (Training Steps)")
    ax.set_ylabel("成功率 (Success Rate)")
    ax.set_title("激光雷达对传感器噪声的鲁棒性: 加入真实感噪声几乎无损")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 50000)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    p = figs / "lidar_noise_robustness_compare.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

    with open(out / "lidar_noise_robustness.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["run", "step", "success_main"])
        for key, s in series.items():
            for step, v in s:
                w.writerow([key, step, f"{v:.6g}"])
    print(f"Saved {out/'lidar_noise_robustness.csv'}")

    print("\n=== final success/main (tail-200 mean +/- std) ===")
    for key in ("depth_baseline", "lidar_clean", "lidar_realistic"):
        m, sd = tail(series[key])
        print(f"  {key:16s}: {m*100:5.1f}% +/- {sd*100:.1f}")


if __name__ == "__main__":
    main()
