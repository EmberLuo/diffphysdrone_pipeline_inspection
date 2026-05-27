#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PX4_DIR="${PX4_DIR:-$HOME/PX4_Firmware}"
PX4_REPO="${PX4_REPO:-https://github.com/PX4/PX4-Autopilot.git}"
PX4_TAG="${PX4_TAG:-v1.11.0}"
MID360_SIM_REPO="${MID360_SIM_REPO:-https://github.com/Tfly6/Mid360_px4_sim_plugin.git}"
MID360_SIM_DIR="${MID360_SIM_DIR:-$HOME/Mid360_px4_sim_plugin}"
MID360_WS="${MID360_WS:-$HOME/mid360_sim_ws}"

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo "[ERR] ROS Noetic not found. Run this inside the pipeline_inspection:noetic container." >&2
  exit 1
fi

if [[ ! -d "$PX4_DIR/.git" ]]; then
  echo "[INFO] Cloning PX4 $PX4_TAG into $PX4_DIR"
  git clone --recursive --branch "$PX4_TAG" --depth 1 "$PX4_REPO" "$PX4_DIR"
else
  echo "[INFO] Reusing PX4 checkout: $PX4_DIR"
  git -C "$PX4_DIR" fetch --depth 1 origin "refs/tags/$PX4_TAG:refs/tags/$PX4_TAG"
  git -C "$PX4_DIR" checkout "$PX4_TAG"
  git -C "$PX4_DIR" submodule update --init --recursive --depth 1
fi

if [[ "${SKIP_PX4_BUILD:-0}" != "1" ]]; then
  echo "[INFO] Building PX4 SITL target px4_sitl_default gazebo"
  make -C "$PX4_DIR" px4_sitl_default gazebo
fi

if [[ "${INSTALL_MID360_MODEL:-1}" == "1" ]]; then
  if [[ ! -d "$MID360_SIM_DIR/.git" ]]; then
    echo "[INFO] Cloning Mid360 PX4 model/plugin into $MID360_SIM_DIR"
    git clone --depth 1 "$MID360_SIM_REPO" "$MID360_SIM_DIR"
  else
    echo "[INFO] Reusing Mid360 model/plugin checkout: $MID360_SIM_DIR"
  fi

  mkdir -p "$REPO_ROOT/sim/models"
  rm -rf "$REPO_ROOT/sim/models/Mid360" "$REPO_ROOT/sim/models/iris_mid360"
  cp -a "$MID360_SIM_DIR/livox_laser_simulation/models/Mid360" "$REPO_ROOT/sim/models/Mid360"
  cp -a "$MID360_SIM_DIR/livox_laser_simulation/models/iris_mid360" "$REPO_ROOT/sim/models/iris_mid360"
  python3 - "$REPO_ROOT/sim/models/Mid360/Mid360.sdf" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = text.replace("<ros_topic>/livox/lidar</ros_topic>", "<ros_topic>/livox/lidar2</ros_topic>")
path.write_text(text)
PY

  if [[ "${SKIP_MID360_PLUGIN_BUILD:-0}" != "1" ]]; then
    echo "[INFO] Building Mid360 Gazebo plugin"
    mkdir -p "$MID360_WS/src"
    ln -sfn "$MID360_SIM_DIR/livox_laser_simulation" "$MID360_WS/src/livox_laser_simulation"
    source /opt/ros/noetic/setup.bash
    (cd "$MID360_WS" && catkin_make -DCMAKE_BUILD_TYPE=Release)
    echo "[OK] Mid360 plugin built: $MID360_WS/devel/lib/liblivox_laser_simulation.so"
  fi
fi

echo "[OK] PX4 checkout ready: $PX4_DIR"
if [[ ! -f "$REPO_ROOT/sim/models/iris_mid360/iris_mid360.sdf" ]]; then
  echo "[WARN] iris_mid360 model was not found under PX4 or repo sim/models." >&2
  echo "[WARN] Point-LIO SITL needs a Mid360 model publishing /livox/lidar2 and /livox/imu." >&2
fi
