#!/usr/bin/env python3
"""Export TensorBoard training logs into thesis CSV tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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

RUN_LABELS = {
    "Original": "Original",
    "RTH": "RTH",
    "RTH+DP": "RTH+DP",
}

TAGS = [
    "success/safety",
    "success/goal",
    "success/hover",
    "success/main",
    "goal/final_error",
    "goal/min_error",
    "hover/position_error",
    "hover/velocity_error",
    "safety/min_obstacle_distance",
    "control/saturation_ratio",
]

ABLATION_TABLE_METRICS = [
    ("final_error_m", "Final error (m)", "goal/final_error", "scalar"),
    ("hover_error_m", "Hover error (m)", "hover/position_error", "scalar"),
    ("collision_rate", "Collision rate (%)", "success/safety", "collision_rate"),
    ("min_clearance_m", "Min clearance (m)", "safety/min_obstacle_distance", "scalar"),
]


setup_chinese_matplotlib()


def display_run_name(name: str) -> str:
    return RUN_LABELS.get(name, name)


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


def last_value(scalars: dict[str, list[tuple[int, float]]], tag: str) -> float | None:
    values = scalars.get(tag) or []
    return values[-1][1] if values else None


def tail_values(scalars: dict[str, list[tuple[int, float]]], tag: str, window: int) -> list[float]:
    values = scalars.get(tag) or []
    if window > 0:
        values = values[-window:]
    return [value for _, value in values]


def mean_std(values: list[float]) -> tuple[float, float] | tuple[None, None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def ablation_metric_values(
    scalars: dict[str, list[tuple[int, float]]],
    tag: str,
    kind: str,
    window: int,
) -> list[float]:
    values = tail_values(scalars, tag, window)
    if kind == "collision_rate":
        return [max(0.0, min(1.0, 1.0 - value)) for value in values]
    return values


def format_mean_std(mean: float | None, std: float | None, *, percent: bool = False) -> str:
    if mean is None or std is None:
        return ""
    if percent:
        return f"{mean * 100:.2f}±{std * 100:.2f}"
    return f"{mean:.3f}±{std:.3f}"


def read_args_json(run_dir: Path) -> dict:
    path = run_dir / "args.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_scalars_csv(output_dir: Path, all_scalars):
    with open(output_dir / "training_scalars.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["run", "tag", "step", "value"])
        writer.writeheader()
        for run, scalars in all_scalars.items():
            for tag, values in scalars.items():
                for step, value in values:
                    writer.writerow({"run": run, "tag": tag, "step": step, "value": f"{value:.8g}"})


def write_ablation_csv(output_dir: Path, runs, all_scalars):
    with open(output_dir / "local_avoidance_ablation_metrics.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["run", *TAGS])
        writer.writeheader()
        for run in runs:
            row = {"run": run}
            for tag in TAGS:
                value = last_value(all_scalars[run], tag)
                row[tag] = "" if value is None else f"{value:.8g}"
            writer.writerow(row)


def write_ablation_stats_tables(output_dir: Path, runs, all_scalars, stats_window: int):
    rows = []
    for run in runs:
        row = {"run": run, "stats_window": stats_window}
        for key, _, tag, kind in ABLATION_TABLE_METRICS:
            values = ablation_metric_values(all_scalars[run], tag, kind, stats_window)
            mean, std = mean_std(values)
            row[f"{key}_mean"] = "" if mean is None else f"{mean:.8g}"
            row[f"{key}_std"] = "" if std is None else f"{std:.8g}"
            row[f"{key}_mean_std"] = format_mean_std(mean, std, percent=(kind == "collision_rate"))
        rows.append(row)

    fieldnames = ["run", "stats_window"]
    for key, _, _, _ in ABLATION_TABLE_METRICS:
        fieldnames.extend([f"{key}_mean", f"{key}_std", f"{key}_mean_std"])

    with open(output_dir / "local_avoidance_ablation_stats.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = ["Method", *[label for _, label, _, _ in ABLATION_TABLE_METRICS]]
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [row["run"]]
                + [
                    row[f"{key}_mean_std"] or "--"
                    for key, _, _, _ in ABLATION_TABLE_METRICS
                ]
            )
            + " |"
        )
    with open(output_dir / "local_avoidance_ablation_table.md", "w", encoding="utf-8") as fp:
        fp.write("\n".join(markdown_lines) + "\n")

    latex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        "Method & "
        + " & ".join(label.replace("%", r"\%") for _, label, _, _ in ABLATION_TABLE_METRICS)
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            row[f"{key}_mean_std"].replace("±", r" $\pm$ ") if row[f"{key}_mean_std"] else r"\textemdash"
            for key, _, _, _ in ABLATION_TABLE_METRICS
        ]
        latex_lines.append(row["run"].replace("+", r"$+$") + " & " + " & ".join(cells) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    with open(output_dir / "local_avoidance_ablation_table.tex", "w", encoding="utf-8") as fp:
        fp.write("\n".join(latex_lines) + "\n")


def plot_tag(output_path: Path, all_scalars, tag_groups, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for run, tag, label in tag_groups:
        values = all_scalars.get(run, {}).get(tag, [])
        if not values:
            continue
        xs = [step for step, _ in values]
        ys = [value for _, value in values]
        ax.plot(xs, ys, label=label)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("训练步数")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="thesis_outputs/local_avoidance")
    parser.add_argument("--figures_dir", default="")
    parser.add_argument(
        "--stats_window",
        type=int,
        default=200,
        help="Number of final TensorBoard scalar samples used for mean/std ablation statistics. Use 0 for all samples.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override/add a run. Can be passed multiple times.",
    )
    args = parser.parse_args()

    runs = dict(DEFAULT_RUNS)
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run must be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        runs[name] = path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir) if args.figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_scalars = {name: load_scalars(Path(path)) for name, path in runs.items()}
    write_scalars_csv(output_dir, all_scalars)
    write_ablation_csv(output_dir, list(runs), all_scalars)
    write_ablation_stats_tables(output_dir, list(runs), all_scalars, args.stats_window)

    metrics = {}
    for run_name, run_path in runs.items():
        scalars = all_scalars[run_name]
        run_args = read_args_json(Path(run_path))
        metrics[run_name] = {
            "args": {
                "num_iters": run_args.get("num_iters"),
                "batch_size": run_args.get("batch_size"),
                "timesteps": run_args.get("timesteps"),
                "goal_radius": run_args.get("goal_radius"),
                "hover_phase_ratio": run_args.get("hover_phase_ratio"),
            },
            "final": {tag: last_value(scalars, tag) for tag in TAGS},
        }
    with open(output_dir / "local_avoidance_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)

    plot_tag(
        figures_dir / "training_depth_success.png",
        all_scalars,
        [
            (name, "success/safety", f"{display_run_name(name)}-安全") for name in runs
        ]
        + [(name, "success/goal", f"{display_run_name(name)}-目标") for name in runs]
        + [(name, "success/hover", f"{display_run_name(name)}-悬停") for name in runs],
        "训练成功率曲线",
        "成功率",
    )
    plot_tag(
        figures_dir / "training_depth_errors.png",
        all_scalars,
        [(name, "goal/final_error", f"{display_run_name(name)}-最终目标误差") for name in runs]
        + [(name, "hover/position_error", f"{display_run_name(name)}-悬停位置误差") for name in runs],
        "训练误差曲线",
        "误差 / m",
    )
    plot_tag(
        figures_dir / "training_depth_safety.png",
        all_scalars,
        [(name, "safety/min_obstacle_distance", f"{display_run_name(name)}-最小障碍距离") for name in runs]
        + [(name, "control/saturation_ratio", f"{display_run_name(name)}-控制饱和率") for name in runs],
        "安全距离与控制饱和率",
        "指标值",
    )

    print(f"Wrote {output_dir}")
    print(f"Wrote ablation table to {output_dir / 'local_avoidance_ablation_table.md'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
