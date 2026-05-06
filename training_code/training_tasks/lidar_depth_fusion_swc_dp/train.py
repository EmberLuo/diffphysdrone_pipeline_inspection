from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from training_code.training_tasks.common import robust_target_hover
from training_code.training_tasks.common.train_loop import TrainingTask, add_common_args, run_training
from training_code.training_tasks.lidar_depth_fusion.train import (
    _add_lidar_args,
    _build_fusion_env,
    _build_fusion_model,
    _make_fusion_observation,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = ROOT / "logs" / "lidar_depth_fusion_swc_dp"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "config_rth.yaml"


def build_rth_task(default_log_root: str | Path = DEFAULT_LOG_ROOT) -> TrainingTask:
    return TrainingTask(
        name="lidar_depth_fusion_swc_dp",
        default_log_root=default_log_root,
        build_env=_build_fusion_env,
        build_model=_build_fusion_model,
        make_observation=_make_fusion_observation,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config with flat robust-target-hover args.")
    add_common_args(parser, default_log_root=DEFAULT_LOG_ROOT)
    _add_lidar_args(parser)
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre_parser.parse_known_args()

    parser = _build_parser()
    parser.set_defaults(**robust_target_hover.load_yaml_defaults(pre_args.config))
    args = parser.parse_args()
    run_training(args, build_rth_task(default_log_root=DEFAULT_LOG_ROOT))


if __name__ == "__main__":
    main()
