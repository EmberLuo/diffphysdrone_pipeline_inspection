#!/usr/bin/env python3
"""Final comparison: depth-multi vs fusion-multi (old cfg) vs fusion-multi (new lidar cfg).

The new-cfg run has two segments: the original run (interrupted, steps 0..~28k)
and a resume that warm-started from its step-20000 checkpoint. We stitch a clean
continuous curve = original[0:20000] ++ resume[0:] offset to start at step 20000.
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
DEPTH_MULTI = LOG / "depth_camera/diffphys/nominal/navigation/multi_agent_odom/20260530_034859"
FUSION_OLD = LOG / "lidar_depth_fusion/diffphys/nominal/navigation/multi_agent_odom/20260702_180643_fusion_multi_compare"
FUSION_NEW_A = LOG / "lidar_depth_fusion/diffphys/nominal/navigation/multi_agent_odom/20260702_201412_fusion_multi_lidarcfg_v2"
FUSION_NEW_B = LOG / "lidar_depth_fusion/diffphys/nominal/navigation/multi_agent_odom/20260703_213931_fusion_multi_lidarcfg_v2_resume"

RESUME_OFFSET = 20000  # resume warm-started from the step-20000 checkpoint of run A

C_DEPTH = "#2196F3"
C_OLD = "#9E9E9E"
C_NEW = "#FF5722"
SMOOTH = 40


def scalars(run: Path, tag: str):
    ev = sorted((run / "tb").glob("events*"))
    if not ev:
        return []
    a = EventAccumulator(str(ev[-1]), size_guidance={"scalars": 0})
    a.Reload()
    if tag not in a.Tags().get("scalars", []):
        return []
    return [(int(s.step), float(s.value)) for s in a.Scalars(tag)]


def stitched_new(tag: str):
    a = scalars(FUSION_NEW_A, tag)
    b = scalars(FUSION_NEW_B, tag)
    a = [(s, v) for s, v in a if s <= RESUME_OFFSET]
    b = [(s + RESUME_OFFSET, v) for s, v in b]
    return a + b


def smooth(ys, w):
    if len(ys) < w:
        return np.array(ys)
    return np.convolve(ys, np.ones(w) / w, mode="valid")


def tail(series, n=200):
    vals = [v for _, v in series[-n:]]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    out = Path("thesis_outputs/lidar_depth_fusion_compare")
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    series = {
        "depth_multi": scalars(DEPTH_MULTI, "success/main"),
        "fusion_old": scalars(FUSION_OLD, "success/main"),
        "fusion_new": stitched_new("success/main"),
    }

    # ── success curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.4, 5), dpi=180)
    for key, label, color in [
        ("depth_multi", "深度相机 (Depth, multi)", C_DEPTH),
        ("fusion_old", "融合-旧雷达配置 (Fusion, old cfg)", C_OLD),
        ("fusion_new", "融合-新雷达配置 (Fusion, new cfg)", C_NEW),
    ]:
        s = series[key]
        if not s:
            continue
        xs = np.array([x for x, _ in s])
        ys = np.array([y for _, y in s])
        ax.plot(xs, ys, color=color, alpha=0.13, linewidth=0.6)
        if len(ys) >= SMOOTH:
            ax.plot(xs[SMOOTH - 1:], smooth(ys, SMOOTH), color=color, linewidth=2.0, label=label)
        m, _ = tail(s)
        if m is not None and key != "fusion_old":
            ax.axhline(m, color=color, linestyle="--", alpha=0.4, linewidth=0.8)
            ax.text(xs[-1] + 300, m, f"{m*100:.1f}%", color=color, fontsize=9,
                    va="center", fontweight="bold")
    ax.axvline(RESUME_OFFSET, color="k", alpha=0.15, linewidth=1.0, linestyle=":")
    ax.text(RESUME_OFFSET, 0.30, " 续训接点", fontsize=8, color="gray", rotation=90, va="bottom")
    ax.set_xlabel("训练步数 (Training Steps)")
    ax.set_ylabel("成功率 (Success Rate)")
    ax.set_title("多机导航成功率: 修正激光雷达配置后融合明显超越深度相机")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 50000)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    p = figs / "fusion_lidarcfg_success_compare.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

    # ── CSV + summary ────────────────────────────────────────────────
    with open(out / "fusion_lidarcfg_success.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["run", "step", "success_main"])
        for key, s in series.items():
            for step, v in s:
                w.writerow([key, step, f"{v:.6g}"])
    print(f"Saved {out/'fusion_lidarcfg_success.csv'}")

    print("\n=== final success/main (tail-200 mean ± std) ===")
    for key in ("depth_multi", "fusion_old", "fusion_new"):
        m, sd = tail(series[key])
        n = len(series[key])
        laststep = series[key][-1][0] if n else 0
        print(f"  {key:12s}: {m*100:5.1f}% ± {sd*100:.1f}  (n={n}, last_step={laststep})")


if __name__ == "__main__":
    main()
