from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from training_code.main_cuda import build_depth_task
from training_code.training_tasks.common.train_loop import add_common_args, run_training


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = ROOT / "logs" / "depth_camera"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
AUTO_ROBUST_CONFIGS = {
    "--use_rth_normal": CONFIG_DIR / "rth_normal.args",
    "--use_robust_env": CONFIG_DIR / "robust_baseline.args",
    "--use_robust_target_hover": CONFIG_DIR / "robust_target_hover.args",
}


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
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_log_root=DEFAULT_LOG_ROOT)
    args = _parse_args(parser, sys.argv[1:])
    run_training(args, build_depth_task(default_log_root=str(DEFAULT_LOG_ROOT)))


if __name__ == "__main__":
    main()
