import os
import sys
from pathlib import Path


def _find_repo_root():
    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "training_code" / "model.py").exists():
            return root

    for parent in Path(__file__).resolve().parents:
        if (parent / "training_code" / "model.py").exists():
            return parent

    raise ImportError(
        "Could not find training_code/model.py. Set REPO_ROOT to the "
        "diffphysdrone_pipeline_inspection repository root."
    )


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training_code.model import Model  # noqa: E402,F401

__all__ = ["Model"]
