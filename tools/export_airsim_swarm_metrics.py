#!/usr/bin/env python3
"""Aggregate AirSim swarm wind experiments into a thesis CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run_dir",
    "seed",
    "target_speed",
    "use_wind",
    "wind_mode",
    "mean_wind_norm",
    "max_wind_norm",
    "arrived_count",
    "success_rate",
    "collision_count",
    "mean_travel_distance",
    "mean_travel_time",
]


def parse_log(log_path: Path) -> list[dict]:
    rows = []
    if not log_path.is_file():
        return rows
    with open(log_path, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 9 or parts[0] != "ours":
                continue
            rows.append(
                {
                    "method": parts[0],
                    "env": parts[1],
                    "target_speed": parts[2],
                    "drone": parts[3],
                    "travel_distance": _float_or_none(parts[4]),
                    "travel_time": _float_or_none(parts[5]),
                    "arrived": parts[7].lower() == "true",
                    "collisions": parts[8],
                    "collided": bool(parts[8]),
                }
            )
    return rows


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def wind_meta(run_dir: Path) -> dict:
    summary = load_json(run_dir / "wind_summary.json")
    if not summary:
        policy = load_json(run_dir / "policy_trace.json")
        summary = policy.get("wind_summary", {}) if isinstance(policy, dict) else {}
        if not summary:
            records = load_jsonl(run_dir / "wind_trace.jsonl")
            norms = [float(rec.get("wind_norm", 0.0)) for rec in records]
            summary = {
                "record_count": len(records),
                "sample_count": sum(1 for rec in records if rec.get("resampled")),
                "effective_wind_ratio": (
                    sum(1 for value in norms if value > 1e-9) / len(norms)
                    if norms else 0.0
                ),
                "mean_wind_norm": sum(norms) / len(norms) if norms else "",
                "max_wind_norm": max(norms) if norms else "",
                "config": {},
            }
    config = summary.get("config", {}) if isinstance(summary, dict) else {}
    return {
        "seed": summary.get("seed", ""),
        "use_wind": config.get("use_wind", ""),
        "wind_mode": config.get("wind_mode", ""),
        "mean_wind_norm": summary.get("mean_wind_norm", ""),
        "max_wind_norm": summary.get("max_wind_norm", ""),
    }


def episode_row(run_dir: Path) -> dict:
    log_rows = parse_log(run_dir / "log")
    wind = wind_meta(run_dir)
    distances = [row["travel_distance"] for row in log_rows if row["travel_distance"] is not None]
    times = [row["travel_time"] for row in log_rows if row["travel_time"] is not None]
    arrived_count = sum(1 for row in log_rows if row["arrived"])
    collision_count = sum(1 for row in log_rows if row["collided"])
    target_speed = log_rows[0]["target_speed"] if log_rows else ""
    return {
        "run_dir": str(run_dir),
        "seed": wind["seed"],
        "target_speed": target_speed,
        "use_wind": wind["use_wind"],
        "wind_mode": wind["wind_mode"],
        "mean_wind_norm": wind["mean_wind_norm"],
        "max_wind_norm": wind["max_wind_norm"],
        "arrived_count": arrived_count,
        "success_rate": arrived_count / len(log_rows) if log_rows else "",
        "collision_count": collision_count,
        "mean_travel_distance": _mean(distances),
        "mean_travel_time": _mean(times),
    }


def discover_runs(root: Path) -> list[Path]:
    candidates = {path.parent for path in root.glob("exps_*/*/log")}
    candidates.update(path.parent for path in root.glob("exps_*/*/wind_summary.json"))
    candidates.update(path.parent for path in root.glob("exps_*/*/wind_trace.jsonl"))
    return sorted(candidates)


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="sim2sim/drone_airsim/swarm")
    parser.add_argument("--output", default="thesis_outputs/airsim_swarm_wind_metrics.csv")
    args = parser.parse_args()

    root = Path(args.root)
    runs = discover_runs(root)
    rows = [episode_row(run) for run in runs]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
