from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from training_code.training_tasks.common.train_loop import (
    TrainingTask,
    add_common_args,
    build_standard_env,
    run_training,
)
from training_code.training_tasks.lidar_navrl.model import LidarNavRLModel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = ROOT / "logs" / "lidar_navrl"


def _add_lidar_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lidar_range", type=float, default=4.0)
    parser.add_argument("--lidar_hbeams", type=int, default=120)
    parser.add_argument("--lidar_vbeams", type=int, default=6)
    parser.add_argument("--lidar_vfov_min", type=float, default=-10.0)
    parser.add_argument("--lidar_vfov_max", type=float, default=20.0)


def _build_lidar_env(args: argparse.Namespace, device: torch.device):
    from training_code.training_tasks.lidar_navrl.env import Env

    return build_standard_env(
        Env,
        args,
        device,
        lidar_range=args.lidar_range,
        lidar_hbeams=args.lidar_hbeams,
        lidar_vbeams=args.lidar_vbeams,
        lidar_vfov=(args.lidar_vfov_min, args.lidar_vfov_max),
    )


def _build_lidar_model(args: argparse.Namespace, device: torch.device) -> LidarNavRLModel:
    state_dim = 7 if args.no_odom else 10
    return LidarNavRLModel(state_dim, 6).to(device)


def _make_lidar_observation(env, args: argparse.Namespace, ctl_dt: float) -> torch.Tensor:
    return env.render_lidar()


def build_lidar_task(default_log_root: str | Path = DEFAULT_LOG_ROOT) -> TrainingTask:
    return TrainingTask(
        name="lidar_navrl",
        default_log_root=default_log_root,
        build_env=_build_lidar_env,
        build_model=_build_lidar_model,
        make_observation=_make_lidar_observation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_log_root=DEFAULT_LOG_ROOT)
    _add_lidar_args(parser)
    args = parser.parse_args()
    run_training(args, build_lidar_task(default_log_root=DEFAULT_LOG_ROOT))


if __name__ == "__main__":
    main()
