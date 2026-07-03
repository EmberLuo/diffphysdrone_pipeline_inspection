#!/usr/bin/env python3
"""Four-way multi-agent comparison: depth-only vs lidar-only vs fusion(old cfg) vs fusion(new cfg).

All runs are multi-agent nominal navigation, 50k steps, diffphys.
Fusion-new is stitched from an interrupted run (0..20k) + a warm-start resume
(from its step-20000 checkpoint), offset to be continuous.
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
DEPTH = LOG / "depth_camera/diffphys/nominal/navigation/multi_agent_odom/20260530_034859"
LIDAR = LOG / "lidar_navrl/diffphys/nominal/navigation/multi_agent_odom/20260704_000743_lidaronly_multi_lidarcfg_v2"
FUSION_A = LOG / "lidar_depth_fusion/diffphys/nominal/navigation/multi_agent_odom/20260702_201412_fusion_multi_lidarcfg_v2"
FUSION_B = LOG / "lidar_depth_fusion/diffphys/nominal/navigation/multi_agent_odom/20260703_213931_fusion_multi_lidarcfg_v2_resume"
RESUME_OFFSET = 20000

C_DEPTH = "#2196F3"
C_LIDAR = "#4CAF50"
C_FUSION = "#FF5722"
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


def stitched_fusion(tag="success/main"):
    a = [(s, v) for s, v in scalars(FUSION_A, tag) if s <= RESUME_OFFSET]
    b = [(s + RESUME_OFFSET, v) for s, v in scalars(FUSION_B, tag)]
    return a + b


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
        "depth_only": scalars(DEPTH),
        "lidar_only": scalars(LIDAR),
        "fusion_new": stitched_fusion(),
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=180)
    for key, label, color in [
        ("depth_only", "深度相机 (Depth-only)", C_DEPTH),
        ("lidar_only", "单激光雷达 (LiDAR-only, new cfg)", C_LIDAR),
        ("fusion_new", "激光雷达+深度融合 (Fusion, new cfg)", C_FUSION),
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
    ax.set_title("多机导航成功率: 三种传感器方案 (修正激光雷达配置后)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 50000)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    p = figs / "sensor_modality_success_compare.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

    with open(out / "sensor_modality_success.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["run", "step", "success_main"])
        for key, s in series.items():
            for step, v in s:
                w.writerow([key, step, f"{v:.6g}"])
    print(f"Saved {out/'sensor_modality_success.csv'}")

    print("\n=== final success/main (tail-200 mean +/- std) ===")
    for key in ("depth_only", "lidar_only", "fusion_new"):
        m, sd = tail(series[key])
        print(f"  {key:12s}: {m*100:5.1f}% +/- {sd*100:.1f}")


if __name__ == "__main__":
    main()
