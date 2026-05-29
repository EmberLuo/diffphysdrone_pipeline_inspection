from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from training_code.training_tasks.common.train_loop import (
    TrainingTask,
    add_common_args,
    build_standard_env,
    run_training,
)
from training_code.training_tasks.lidar_depth_fusion.model import LidarDepthFusionModel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = ROOT / "logs" / "lidar_depth_fusion"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
AUTO_ROBUST_CONFIGS = {
    "--use_rth_normal": CONFIG_DIR / "rth_normal.args",
    "--use_robust_env": CONFIG_DIR / "robust_baseline.args",
    "--use_robust_target_hover": CONFIG_DIR / "robust_target_hover.args",
}


def _add_lidar_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lidar_range", type=float, default=4.0)
    parser.add_argument("--lidar_hbeams", type=int, default=120)
    parser.add_argument("--lidar_vbeams", type=int, default=6)
    parser.add_argument("--lidar_vfov_min", type=float, default=-10.0)
    parser.add_argument("--lidar_vfov_max", type=float, default=20.0)


def _build_fusion_env(args: argparse.Namespace, device: torch.device):
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


def _build_fusion_model(args: argparse.Namespace, device: torch.device) -> LidarDepthFusionModel:
    state_dim = 7 if args.no_odom else 10
    return LidarDepthFusionModel(state_dim, 6).to(device)


def _make_fusion_observation(env, args: argparse.Namespace, ctl_dt: float):
    depth, _ = env.render(ctl_dt)
    depth_obs = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
    depth_obs = F.max_pool2d(depth_obs[:, None], 4, 4)
    lidar_obs = env.render_lidar()
    return depth_obs, lidar_obs


def build_fusion_task(default_log_root: str | Path = DEFAULT_LOG_ROOT) -> TrainingTask:
    return TrainingTask(
        name="lidar_depth_fusion",
        default_log_root=default_log_root,
        build_env=_build_fusion_env,
        build_model=_build_fusion_model,
        make_observation=_make_fusion_observation,
    )


def _read_args_file(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return shlex.split(f.read(), comments=True)


def _select_auto_config(argv: list[str]) -> Path | None:
    options = {arg.split("=", 1)[0] for arg in argv}
    if "--use_rth_normal" in options:
        return AUTO_ROBUST_CONFIGS["--use_rth_normal"]
    if "--use_robust_target_hover" in options:
        return AUTO_ROBUST_CONFIGS["--use_robust_target_hover"]
    if "--use_robust_env" in options:
        return AUTO_ROBUST_CONFIGS["--use_robust_env"]
    return None


def _parse_args(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    auto_config = _select_auto_config(argv)
    if auto_config is not None:
        if not auto_config.exists():
            raise FileNotFoundError(f"Auto robust config does not exist: {auto_config}")
        config_args = parser.parse_args(_read_args_file(auto_config))
        parser.set_defaults(**vars(config_args))

    args = parser.parse_args(argv)
    args.auto_config_path = str(auto_config) if auto_config is not None else None
    if auto_config is not None:
        print(f"Auto robust config: {auto_config}")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_common_args(parser, default_log_root=DEFAULT_LOG_ROOT)
    _add_lidar_args(parser)
    args = _parse_args(parser, sys.argv[1:])
    run_training(args, build_fusion_task(default_log_root=DEFAULT_LOG_ROOT))


if __name__ == "__main__":
    main()
