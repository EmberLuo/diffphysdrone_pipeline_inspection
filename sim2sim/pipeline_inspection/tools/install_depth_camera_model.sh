#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4_Firmware}"
SRC_MODEL="${1:-$PX4_DIR/Tools/sitl_gazebo/models/iris_mid360}"
DST_ROOT="${2:-$REPO_ROOT/sim/models}"
DST_MODEL="$DST_ROOT/iris_mid360_e2e"
SNIPPET="$REPO_ROOT/ros/navigation_bringup/urdf/depth_camera_gazebo_snippet.xacro"

if [[ ! -d "$SRC_MODEL" ]]; then
  echo "Source model not found: $SRC_MODEL" >&2
  exit 1
fi
if [[ ! -f "$SNIPPET" ]]; then
  echo "Depth camera snippet not found: $SNIPPET" >&2
  exit 1
fi

mkdir -p "$DST_ROOT"
rm -rf "$DST_MODEL"
cp -a "$SRC_MODEL" "$DST_MODEL"

python3 - "$DST_MODEL" "$SNIPPET" <<'PY'
import pathlib
import sys

model_dir = pathlib.Path(sys.argv[1])
snippet_path = pathlib.Path(sys.argv[2])
snippet = snippet_path.read_text()

sdf_candidates = [model_dir / "model.sdf"] + sorted(model_dir.glob("*.sdf"))
sdf_path = next((p for p in sdf_candidates if p.exists()), None)
if sdf_path is None:
    raise SystemExit(f"No SDF file found in {model_dir}")

text = sdf_path.read_text()
if "e2e_depth_camera" not in text:
    idx = text.rfind("</model>")
    if idx < 0:
        raise SystemExit(f"{sdf_path} has no </model> tag")
    text = text[:idx] + "\n" + snippet + "\n" + text[idx:]
    sdf_path.write_text(text)

config_path = model_dir / "model.config"
if config_path.exists():
    cfg = config_path.read_text()
    cfg = cfg.replace("<name>iris_mid360</name>", "<name>iris_mid360_e2e</name>")
    config_path.write_text(cfg)

print(f"Prepared {model_dir}")
PY

echo "Add this before launching PX4/Gazebo:"
echo "  export GAZEBO_MODEL_PATH=\"$DST_ROOT:\${GAZEBO_MODEL_PATH:-}\""
echo "Use PX4 model: iris_mid360_e2e"
