#!/usr/bin/env python3
"""Aggregate AirSim complete-task runs for Chapter 4 closed-loop tables."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


TARGETS = {
    "drone_1": np.array([6.0, 1.2, 0.0], dtype=float),
    "drone_2": np.array([6.0, 0.0, 0.0], dtype=float),
    "drone_3": np.array([6.0, -1.2, 0.0], dtype=float),
    "drone_4": np.array([0.0, 1.2, 0.0], dtype=float),
    "drone_5": np.array([0.0, 0.0, 0.0], dtype=float),
    "drone_6": np.array([0.0, -1.2, 0.0], dtype=float),
}


@dataclass
class AgentLog:
    run_dir: Path
    condition: str
    run_index: int
    drone: str
    travel_distance_m: float
    travel_time_s: float
    arrived: bool
    collisions: str


def _float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_namespace(line: str) -> dict:
    if not line.startswith("Namespace("):
        return {}
    try:
        parsed_call = ast.parse(line, mode="eval").body
        if not isinstance(parsed_call, ast.Call):
            return {}
        parsed = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in parsed_call.keywords
            if keyword.arg is not None
        }
    except Exception:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def parse_log(run_dir: Path, condition: str, run_index: int) -> tuple[dict, list[AgentLog]]:
    log_path = run_dir / "log"
    if not log_path.is_file():
        return {}, []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    args = parse_namespace(lines[0]) if lines else {}
    rows: list[AgentLog] = []
    for line in lines[1:]:
        parts = line.rstrip("\n").split(",")
        if len(parts) < 9 or parts[0] != "ours":
            continue
        rows.append(
            AgentLog(
                run_dir=run_dir,
                condition=condition,
                run_index=run_index,
                drone=parts[3],
                travel_distance_m=_float(parts[4]),
                travel_time_s=_float(parts[5]),
                arrived=parts[7].lower() == "true",
                collisions=parts[8],
            )
        )
    return args, rows


def load_traj(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "traj_history.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    traj = {}
    for drone, samples in raw.items():
        arr = np.asarray(samples, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3 and len(arr):
            traj[drone] = np.column_stack([arr[:, 0], -arr[:, 1], -arr[:, 2]])
    return traj


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wind_enabled(run_dir: Path, args: dict) -> bool:
    if "use_wind" in args:
        return bool(args["use_wind"])
    summary = load_json(run_dir / "wind_summary.json")
    config = summary.get("config", {}) if isinstance(summary, dict) else {}
    return bool(config.get("use_wind", False))


def mean_wind_norm(run_dir: Path) -> float:
    summary = load_json(run_dir / "wind_summary.json")
    if "mean_wind_norm" in summary:
        return _float(str(summary["mean_wind_norm"]))
    norms = [_float(str(row.get("wind_norm", ""))) for row in load_jsonl(run_dir / "wind_trace.jsonl")]
    norms = [v for v in norms if math.isfinite(v)]
    return float(np.mean(norms)) if norms else float("nan")


def target_error(traj: dict[str, np.ndarray], drone: str) -> float:
    if drone not in traj or drone not in TARGETS:
        return float("nan")
    return float(np.linalg.norm(traj[drone][-1] - TARGETS[drone]))


def hover_error(traj: dict[str, np.ndarray], drone: str, tail_count: int = 10) -> float:
    if drone not in traj or drone not in TARGETS:
        return float("nan")
    tail = traj[drone][-min(tail_count, len(traj[drone])) :]
    return float(np.sqrt(np.mean(np.sum((tail - TARGETS[drone]) ** 2, axis=1))))


def min_inter_drone_distance(traj: dict[str, np.ndarray]) -> float:
    drones = sorted(traj)
    best = float("nan")
    if len(drones) < 2:
        return best
    max_len = max(len(traj[d]) for d in drones)
    values = []
    for idx in range(max_len):
        poses = []
        for drone in drones:
            arr = traj[drone]
            poses.append(arr[min(idx, len(arr) - 1)])
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                values.append(float(np.linalg.norm(poses[i] - poses[j])))
    return min(values) if values else best


def min_pipe_clearance(run_dir: Path, traj: dict[str, np.ndarray]) -> float:
    scene = load_json(run_dir / "pipe_scene.json")
    pipes = scene.get("pipes", []) if isinstance(scene, dict) else []
    if not pipes:
        return float("nan")
    values = []
    for arr in traj.values():
        for p in arr:
            for pipe in pipes:
                start = np.asarray(pipe["start"], dtype=float)
                end = np.asarray(pipe["end"], dtype=float)
                radius = float(pipe.get("diameter", 0.0)) * 0.5
                seg = end - start
                denom = float(np.dot(seg, seg))
                if denom <= 1e-9:
                    continue
                u = float(np.clip(np.dot(p - start, seg) / denom, 0.0, 1.0))
                closest = start + u * seg
                values.append(float(np.linalg.norm(p - closest) - radius))
    return min(values) if values else float("nan")


def min_depth_clearance(run_dir: Path) -> float:
    policy = load_json(run_dir / "policy_trace.json")
    records = policy.get("records", []) if isinstance(policy, dict) else []
    depth_values = [_float(str(row.get("depth_min", ""))) for row in records]
    depth_values = [v for v in depth_values if math.isfinite(v)]
    if depth_values:
        return min(depth_values)
    values = []
    for row in load_jsonl(run_dir / "policy_trace.jsonl"):
        v = _float(str(row.get("depth_min", "")))
        if math.isfinite(v):
            values.append(v)
    return min(values) if values else float("nan")


def collision_object_count(collisions: str, run_dir: Path) -> int:
    if not collisions:
        return 0

    known = {f"drone_{idx}" for idx in range(1, 7)}
    scene = load_json(run_dir / "pipe_scene.json")
    for pipe in scene.get("pipes", []) if isinstance(scene, dict) else []:
        spawned = pipe.get("spawned_name")
        if spawned:
            known.add(str(spawned))
    known.update(
        {
            "pipe_swarm_obstacle_cross_y_upper",
            "pipe_swarm_obstacle_cross_y_lower",
            "pipe_swarm_obstacle_cross_y_mid_offset",
            "pipe_swarm_obstacle_rail_x_left",
            "pipe_swarm_obstacle_rail_x_right",
            "pipe_swarm_obstacle_vertical_front_left",
            "pipe_swarm_obstacle_vertical_front_right",
            "pipe_swarm_obstacle_vertical_back_left",
            "pipe_swarm_obstacle_vertical_back_right",
            "pipe_swarm_obstacle_diagonal_a",
            "pipe_swarm_obstacle_diagonal_b",
            "1M_Cube_Chamfer4_9",
            "1M_Cube_Chamfer5",
            "1M_Cube_Chamfer6",
            "1M_Cube_Chamfer22",
        }
    )

    remaining = collisions
    count = 0
    for name in sorted(known, key=len, reverse=True):
        occurrences = len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", remaining))
        if occurrences:
            count += occurrences
            remaining = remaining.replace(name, "")

    # Collision object names are joined with underscores, while AirSim object
    # names also contain underscores. Any residue therefore counts as one
    # additional unknown object rather than being split naively.
    if remaining.strip("_"):
        count += 1
    return count


def discover_runs(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.glob("exps_*/*/log") if (path.parent / "traj_history.json").is_file()})


def select_runs(root: Path, condition: str, want_wind: bool, limit: int, min_duration: float) -> list[Path]:
    selected = []
    for run in discover_runs(root):
        args, logs = parse_log(run, condition, len(selected) + 1)
        if not logs:
            continue
        if wind_enabled(run, args) != want_wind:
            continue
        duration = _float(str(args.get("duration", 30.0)), default=30.0)
        if duration < min_duration:
            continue
        selected.append(run)
    return selected[-limit:]


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = finite(values)
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def fmt(value: float, digits: int = 3) -> str:
    return "" if not math.isfinite(value) else f"{value:.{digits}f}"


def build_rows(root: Path, condition: str, want_wind: bool, limit: int, min_duration: float) -> tuple[list[dict], list[dict]]:
    runs = select_runs(root, condition, want_wind, limit, min_duration)
    episode_rows = []
    agent_rows = []
    for run_index, run in enumerate(runs, start=1):
        args, logs = parse_log(run, condition, run_index)
        traj = load_traj(run)
        agent_errors = []
        hover_errors = []
        safety_agent_count = 0
        safety_events = 0
        for row in logs:
            safety_agent_count += int(bool(row.collisions))
            safety_events += collision_object_count(row.collisions, run)
            goal_err = target_error(traj, row.drone)
            hover_err = hover_error(traj, row.drone)
            agent_errors.append(goal_err)
            hover_errors.append(hover_err)
            agent_rows.append(
                {
                    "condition": condition,
                    "run_index": run_index,
                    "run_dir": str(run),
                    "drone": row.drone,
                    "arrived": int(row.arrived),
                    "travel_time_s": fmt(row.travel_time_s),
                    "path_length_m": fmt(row.travel_distance_m),
                    "target_error_m": fmt(goal_err),
                    "hover_error_m": fmt(hover_err),
                    "collision_objects": row.collisions,
                }
            )
        min_depth = min_depth_clearance(run)
        min_pipe = min_pipe_clearance(run, traj)
        min_swarm = min_inter_drone_distance(traj)
        clearance_candidates = finite([min_depth, min_pipe, min_swarm])
        min_clearance = min(clearance_candidates) if clearance_candidates else float("nan")
        arrived_count = sum(1 for row in logs if row.arrived)
        task_completed = int(arrived_count == len(logs))
        collision_free_success = int(task_completed and safety_events == 0)
        episode_rows.append(
            {
                "condition": condition,
                "run_index": run_index,
                "run_dir": str(run),
                "seed": args.get("seed", ""),
                "use_wind": int(want_wind),
                "mean_wind_norm": fmt(mean_wind_norm(run)),
                "agent_count": len(logs),
                "arrived_count": arrived_count,
                "task_completed": task_completed,
                "collision_free_success": collision_free_success,
                "agent_success_rate": fmt(arrived_count / len(logs) * 100.0 if logs else float("nan"), 1),
                "mean_flight_time_s": fmt(np.mean([row.travel_time_s for row in logs])),
                "mean_path_length_m": fmt(np.mean([row.travel_distance_m for row in logs])),
                "mean_target_error_m": fmt(np.mean(finite(agent_errors))),
                "mean_hover_error_m": fmt(np.mean(finite(hover_errors))),
                "min_clearance_m": fmt(min_clearance),
                "safety_trigger_agent_count": safety_agent_count,
                "safety_trigger_count": safety_events,
            }
        )
    return episode_rows, agent_rows


def summarize(episode_rows: list[dict]) -> list[dict]:
    out = []
    for condition in sorted({row["condition"] for row in episode_rows}):
        rows = [row for row in episode_rows if row["condition"] == condition]
        n = len(rows)
        completion_rate = sum(int(row["task_completed"]) for row in rows) / n * 100.0 if n else float("nan")
        collision_free_rate = sum(int(row["collision_free_success"]) for row in rows) / n * 100.0 if n else float("nan")
        agent_success = mean_std(_float(row["agent_success_rate"]) for row in rows)
        flight_time = mean_std(_float(row["mean_flight_time_s"]) for row in rows)
        path_length = mean_std(_float(row["mean_path_length_m"]) for row in rows)
        target = mean_std(_float(row["mean_target_error_m"]) for row in rows)
        hover = mean_std(_float(row["mean_hover_error_m"]) for row in rows)
        clearance = mean_std(_float(row["min_clearance_m"]) for row in rows)
        safety_agents = mean_std(float(row["safety_trigger_agent_count"]) for row in rows)
        safety = mean_std(float(row["safety_trigger_count"]) for row in rows)
        wind = mean_std(_float(row["mean_wind_norm"]) for row in rows)
        out.append(
            {
                "condition": condition,
                "episode_count": n,
                "completion_rate": fmt(completion_rate, 1),
                "collision_free_rate": fmt(collision_free_rate, 1),
                "agent_success_rate_mean": fmt(agent_success[0], 1),
                "agent_success_rate_std": fmt(agent_success[1], 1),
                "flight_time_s_mean": fmt(flight_time[0]),
                "flight_time_s_std": fmt(flight_time[1]),
                "path_length_m_mean": fmt(path_length[0]),
                "path_length_m_std": fmt(path_length[1]),
                "target_error_m_mean": fmt(target[0]),
                "target_error_m_std": fmt(target[1]),
                "hover_error_m_mean": fmt(hover[0]),
                "hover_error_m_std": fmt(hover[1]),
                "min_clearance_m_mean": fmt(clearance[0]),
                "min_clearance_m_std": fmt(clearance[1]),
                "safety_trigger_agents_mean": fmt(safety_agents[0]),
                "safety_trigger_agents_std": fmt(safety_agents[1]),
                "safety_trigger_mean": fmt(safety[0]),
                "safety_trigger_std": fmt(safety[1]),
                "wind_norm_mean": fmt(wind[0]),
                "wind_norm_std": fmt(wind[1]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--swarm-root", type=Path, default=Path("sim2sim/drone_airsim/swarm"))
    parser.add_argument("--pipe-root", type=Path, default=Path("sim2sim/drone_airsim/pipe_swarm"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=25.0)
    parser.add_argument("--output-dir", type=Path, default=Path("thesis_outputs/airsim_closed_loop"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    output = (repo / args.output_dir).resolve()
    groups = [
        ((repo / args.swarm_root).resolve(), "AirSim无风", False),
        ((repo / args.pipe_root).resolve(), "AirSim管道有风", True),
    ]
    episode_rows: list[dict] = []
    agent_rows: list[dict] = []
    for root, condition, want_wind in groups:
        episodes, agents = build_rows(root, condition, want_wind, args.limit, args.min_duration)
        episode_rows.extend(episodes)
        agent_rows.extend(agents)

    summary_rows = summarize(episode_rows)
    if episode_rows:
        write_csv(output / "airsim_closed_loop_episodes.csv", episode_rows)
    if agent_rows:
        write_csv(output / "airsim_closed_loop_agents.csv", agent_rows)
    if summary_rows:
        write_csv(output / "airsim_closed_loop_summary.csv", summary_rows)

    for row in summary_rows:
        print(row)
    missing = [row for row in summary_rows if int(row["episode_count"]) < args.limit]
    if missing:
        names = ", ".join(f"{row['condition']}={row['episode_count']}/{args.limit}" for row in missing)
        raise SystemExit(f"insufficient runs for requested limit: {names}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
