from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from training_code.training_tasks.common.train_loop import (
    TrainingTask,
    add_common_args,
    build_standard_env,
    run_training,
)


def _build_depth_env(args: argparse.Namespace, device: torch.device):
    from training_code.env_cuda import Env

    return build_standard_env(Env, args, device)


def _build_depth_model(args: argparse.Namespace, device: torch.device):
    from training_code.model import Model

    state_dim = 7 if args.no_odom else 10
    return Model(state_dim, 6).to(device)


def _make_depth_observation(env, args: argparse.Namespace, ctl_dt: float) -> torch.Tensor:
    depth, _ = env.render(ctl_dt)
    x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
    return F.max_pool2d(x[:, None], 4, 4)


def build_depth_task(default_log_root: str = "logs") -> TrainingTask:
    return TrainingTask(
        name="depth_camera",
        default_log_root=default_log_root,
        build_env=_build_depth_env,
        build_model=_build_depth_model,
        make_observation=_make_depth_observation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_log_root="logs")
    args = parser.parse_args()
    run_training(args, build_depth_task(default_log_root="logs"))


if __name__ == "__main__":
    main()
