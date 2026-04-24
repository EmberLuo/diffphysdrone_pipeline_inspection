import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DRONE_GENESIS_DIR = THIS_DIR.parents[1]
REPO_ROOT = THIS_DIR.parents[2]


def _latest_logs(exps_root: Path):
    if not exps_root.exists():
        return []
    return sorted([p for p in exps_root.iterdir() if p.is_dir()])


def main():
    parser = argparse.ArgumentParser(description="Run Genesis lidar_navrl nav eval multiple times")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--target_speed", type=float, default=0.5)
    parser.add_argument("--resume", type=str,
                        default="training_code/logs/lidar_navrl/single_agent/20260422_125516/checkpoint0004.pth")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "config" / "nav_eval.yaml"))
    parser.add_argument("--output_root", type=str, default=str(THIS_DIR))
    parser.add_argument("--show_viewer", action="store_true", default=False)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--duration_sec", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--clockspeed", type=float, default=0.0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    exps_root = output_root / f"exps_{args.target_speed}"
    exps_root.mkdir(parents=True, exist_ok=True)

    before = set(_latest_logs(exps_root))

    for i in range(args.num_runs):
        seed = args.seed_start + i
        cmd = [
            sys.executable,
            str(THIS_DIR / "eval.py"),
            "--config", args.config,
            "--resume", args.resume,
            "--target_speed", str(args.target_speed),
            "--seed", str(seed),
            "--num_episodes", "1",
            "--output_root", str(output_root),
        ]
        if args.duration_sec is not None:
            cmd.extend(["--duration_sec", str(args.duration_sec)])
        if args.num_steps is not None:
            cmd.extend(["--num_steps", str(args.num_steps)])
        cmd.extend(["--clockspeed", str(args.clockspeed)])
        if args.show_viewer:
            cmd.append("--show_viewer")
        else:
            cmd.append("--no_show_viewer")
        if args.record:
            cmd.append("--record")

        print(f"[{i + 1}/{args.num_runs}] running seed={seed}")
        subprocess.run(cmd, check=True)

    after = set(_latest_logs(exps_root))
    new_dirs = sorted(after - before)
    print("new_logs:")
    for d in new_dirs:
        print(d)


if __name__ == "__main__":
    main()
