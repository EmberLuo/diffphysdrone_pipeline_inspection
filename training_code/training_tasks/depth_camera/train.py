from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from training_code.main_cuda import build_depth_task
from training_code.training_tasks.common.train_loop import add_common_args, run_training


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = ROOT / "logs" / "depth_camera"


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_log_root=DEFAULT_LOG_ROOT)
    args = parser.parse_args()
    run_training(args, build_depth_task(default_log_root=str(DEFAULT_LOG_ROOT)))


if __name__ == "__main__":
    main()
