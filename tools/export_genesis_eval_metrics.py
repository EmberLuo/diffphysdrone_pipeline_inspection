#!/usr/bin/env python3
"""Aggregate Genesis eval folders into thesis robustness/closed-loop CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: python -m pip install pyyaml") from exc


def _norm3(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _load_goals(config_path: Path, env_name: str) -> dict[str, list[float]]:
    with open(config_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    agents = cfg["task"]["layouts"][env_name]["agents"]
    return {item["name"]: list(item["goal"]) for item in agents}


def _parse_log(log_path: Path) -> dict[str, dict]:
    out = {}
    if not log_path.is_file():
        return out
    with open(log_path, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split(",")
            if len(parts) < 9 or parts[0] != "ours":
                continue
            method, env, speed, name, distance, time_s, _, arrived, collisions = parts[:9]
            out[name] = {
                "method_raw": method,
                "env": env,
                "target_speed": speed,
                "traveled_distance_m": float(distance),
                "traveled_time_s": float(time_s),
                "arrived": arrived.lower() == "true",
                "collided": bool(collisions),
                "collisions": collisions,
            }
    return out


def _trace_metrics(trace_path: Path) -> dict[str, dict]:
    if not trace_path.is_file():
        return {}
    with open(trace_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    records = data.get("records", [])
    by_drone: dict[str, list[dict]] = {}
    for rec in records:
        by_drone.setdefault(rec.get("drone", "drone"), []).append(rec)
    out = {}
    for name, items in by_drone.items():
        depth_min = [float(x["depth_min"]) for x in items if "depth_min" in x]
        throttle = [float(x["throttle_des"]) for x in items if "throttle_des" in x]
        saturation_ratio = ""
        if throttle:
            saturated = [x for x in throttle if x <= 0.02 or x >= 0.65]
            saturation_ratio = len(saturated) / len(throttle)
        out[name] = {
            "min_obstacle_distance_m": min(depth_min) if depth_min else "",
            "control_saturation_ratio": saturation_ratio,
        }
    return out


def _episode_rows(exp_dir: Path, goals: dict[str, list[float]], method: str, disturbance: str, hover_tail_samples: int):
    traj_path = exp_dir / "traj_history.json"
    if not traj_path.is_file():
        return []
    with open(traj_path, "r", encoding="utf-8") as fp:
        traj = json.load(fp)
    log_meta = _parse_log(exp_dir / "log")
    trace_meta = _trace_metrics(exp_dir / "policy_trace.json")

    rows = []
    for name, states in traj.items():
        if not states:
            continue
        goal = goals.get(name)
        if goal is None:
            continue
        positions = [s[:3] for s in states]
        final_goal_error = _norm3(positions[-1], goal)
        tail = positions[-max(1, hover_tail_samples):]
        hover_error = math.sqrt(sum(_norm3(p, goal) ** 2 for p in tail) / len(tail))
        path_length = sum(_norm3(positions[i], positions[i - 1]) for i in range(1, len(positions)))
        meta = log_meta.get(name, {})
        tmeta = trace_meta.get(name, {})
        rows.append(
            {
                "episode_dir": str(exp_dir),
                "method": method,
                "disturbance": disturbance,
                "drone": name,
                "arrived": meta.get("arrived", final_goal_error < 1.5),
                "collided": meta.get("collided", ""),
                "target_speed": meta.get("target_speed", ""),
                "traveled_time_s": meta.get("traveled_time_s", ""),
                "path_length_m": path_length,
                "final_goal_error_m": final_goal_error,
                "hover_error_m": hover_error,
                "min_obstacle_distance_m": tmeta.get("min_obstacle_distance_m", ""),
                "control_saturation_ratio": tmeta.get("control_saturation_ratio", ""),
            }
        )
    return rows


def _mean(values):
    values = [float(v) for v in values if v != ""]
    return "" if not values else sum(values) / len(values)


def _rate(values):
    values = [v for v in values if v != ""]
    if not values:
        return ""
    return sum(1 for v in values if bool(v)) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True, help="Directory containing eval episode folders.")
    parser.add_argument("--config", default="sim2sim/drone_genesis/lidar_depth_fusion/config/nav_eval.yaml")
    parser.add_argument("--env", default="single_nav")
    parser.add_argument("--method", default="method")
    parser.add_argument("--disturbance", default="nominal")
    parser.add_argument("--output_dir", default="thesis_outputs/genesis_eval")
    parser.add_argument("--hover_tail_samples", type=int, default=75)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    goals = _load_goals(Path(args.config), args.env)

    rows = []
    for traj in sorted(input_root.glob("**/traj_history.json")):
        rows.extend(_episode_rows(traj.parent, goals, args.method, args.disturbance, args.hover_tail_samples))

    detail_path = output_dir / "genesis_eval_episodes.csv"
    fields = [
        "episode_dir",
        "method",
        "disturbance",
        "drone",
        "arrived",
        "collided",
        "target_speed",
        "traveled_time_s",
        "path_length_m",
        "final_goal_error_m",
        "hover_error_m",
        "min_obstacle_distance_m",
        "control_saturation_ratio",
    ]
    with open(detail_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "genesis_eval_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "method",
                "disturbance",
                "episode_count",
                "goal_reach_rate",
                "collision_rate",
                "mean_goal_error_m",
                "mean_hover_error_m",
                "mean_min_obstacle_distance_m",
                "mean_control_saturation_ratio",
                "mean_path_length_m",
                "mean_time_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": args.method,
                "disturbance": args.disturbance,
                "episode_count": len(rows),
                "goal_reach_rate": _rate([r["arrived"] for r in rows]),
                "collision_rate": _rate([r["collided"] for r in rows]),
                "mean_goal_error_m": _mean([r["final_goal_error_m"] for r in rows]),
                "mean_hover_error_m": _mean([r["hover_error_m"] for r in rows]),
                "mean_min_obstacle_distance_m": _mean([r["min_obstacle_distance_m"] for r in rows]),
                "mean_control_saturation_ratio": _mean([r["control_saturation_ratio"] for r in rows]),
                "mean_path_length_m": _mean([r["path_length_m"] for r in rows]),
                "mean_time_s": _mean([r["traveled_time_s"] for r in rows]),
            }
        )

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
