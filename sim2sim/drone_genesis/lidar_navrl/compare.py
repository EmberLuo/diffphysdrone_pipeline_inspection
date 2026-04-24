import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def _find_log_files(root: Path):
    if (root / "log").is_file():
        return [root / "log"]
    return sorted(root.rglob("log"))


def _parse_log_file(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        return rows

    for ln in lines[1:]:
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 9:
            continue
        _, env_name, target_speed, drone_name, distance, t_sec, _, done, collisions = parts[:9]
        collision_count = 0 if collisions == "" else len([x for x in collisions.split("_") if x])
        rows.append(
            {
                "run": path.parent.name,
                "env": env_name,
                "target_speed": float(target_speed),
                "drone": drone_name,
                "distance": float(distance),
                "time": float(t_sec),
                "done": done.lower() == "true",
                "collision_count": collision_count,
            }
        )
    return rows


def _aggregate(rows):
    per_drone = {}
    for r in rows:
        per_drone.setdefault(r["drone"], []).append(r)

    drone_stats = {}
    for drone, items in sorted(per_drone.items()):
        times = [x["time"] for x in items]
        dones = [x["done"] for x in items]
        collisions = [x["collision_count"] for x in items]
        drone_stats[drone] = {
            "num_runs": len(items),
            "time_mean": mean(times),
            "time_std": pstdev(times) if len(times) > 1 else 0.0,
            "completion_rate": sum(dones) / len(dones),
            "collision_events_total": int(sum(collisions)),
        }

    all_dones = [x["done"] for x in rows]
    all_collisions = [x["collision_count"] for x in rows]
    summary = {
        "num_logs": len({x["run"] for x in rows}),
        "num_rows": len(rows),
        "overall_completion_rate": sum(all_dones) / len(all_dones) if all_dones else 0.0,
        "overall_collision_events_total": int(sum(all_collisions)),
    }
    return {"summary": summary, "per_drone": drone_stats}


def _compare(airsim_stats: dict, genesis_stats: dict, threshold_mae: float, threshold_coll_diff: int):
    drones = sorted(set(airsim_stats["per_drone"].keys()) & set(genesis_stats["per_drone"].keys()))
    per_drone = {}
    maes = []
    for d in drones:
        a = airsim_stats["per_drone"][d]
        g = genesis_stats["per_drone"][d]
        mae = abs(a["time_mean"] - g["time_mean"])
        maes.append(mae)
        per_drone[d] = {
            "airsim_time_mean": a["time_mean"],
            "genesis_time_mean": g["time_mean"],
            "time_mean_abs_error": mae,
            "airsim_completion_rate": a["completion_rate"],
            "genesis_completion_rate": g["completion_rate"],
        }

    airsim_collision_total = airsim_stats["summary"]["overall_collision_events_total"]
    genesis_collision_total = genesis_stats["summary"]["overall_collision_events_total"]
    collision_diff = abs(airsim_collision_total - genesis_collision_total)

    avg_mae = mean(maes) if maes else 0.0
    completion_ok = genesis_stats["summary"]["overall_completion_rate"] >= 1.0
    mae_ok = avg_mae <= threshold_mae
    collision_ok = collision_diff <= threshold_coll_diff

    return {
        "per_drone": per_drone,
        "aggregate": {
            "avg_time_mean_abs_error": avg_mae,
            "airsim_collision_total": airsim_collision_total,
            "genesis_collision_total": genesis_collision_total,
            "collision_total_abs_diff": collision_diff,
            "thresholds": {
                "time_mean_mae_max": threshold_mae,
                "collision_total_abs_diff_max": threshold_coll_diff,
            },
            "pass": bool(completion_ok and mae_ok and collision_ok),
            "checks": {
                "completion_ok": completion_ok,
                "mae_ok": mae_ok,
                "collision_ok": collision_ok,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare AirSim and Genesis lidar_navrl nav logs")
    parser.add_argument("--airsim_root", type=str, required=True)
    parser.add_argument("--genesis_root", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--time_mae_threshold", type=float, default=1.5)
    parser.add_argument("--collision_diff_threshold", type=int, default=2)
    args = parser.parse_args()

    airsim_logs = _find_log_files(Path(args.airsim_root))
    genesis_logs = _find_log_files(Path(args.genesis_root))

    if not airsim_logs:
        raise FileNotFoundError(f"No AirSim logs found under {args.airsim_root}")
    if not genesis_logs:
        raise FileNotFoundError(f"No Genesis logs found under {args.genesis_root}")

    airsim_rows = []
    for p in airsim_logs:
        airsim_rows.extend(_parse_log_file(p))
    genesis_rows = []
    for p in genesis_logs:
        genesis_rows.extend(_parse_log_file(p))

    airsim_stats = _aggregate(airsim_rows)
    genesis_stats = _aggregate(genesis_rows)
    comparison = _compare(
        airsim_stats=airsim_stats,
        genesis_stats=genesis_stats,
        threshold_mae=args.time_mae_threshold,
        threshold_coll_diff=args.collision_diff_threshold,
    )

    result = {
        "airsim": airsim_stats,
        "genesis": genesis_stats,
        "comparison": comparison,
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["drone", "airsim_time_mean", "genesis_time_mean",
                 "time_mean_abs_error", "airsim_completion_rate", "genesis_completion_rate"]
            )
            for drone, row in sorted(comparison["per_drone"].items()):
                writer.writerow(
                    [drone, row["airsim_time_mean"], row["genesis_time_mean"],
                     row["time_mean_abs_error"], row["airsim_completion_rate"], row["genesis_completion_rate"]]
                )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()