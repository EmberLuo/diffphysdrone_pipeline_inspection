#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WS_DIR="${WS_DIR:-$REPO_ROOT/integration_ws}"
PX4_DIR="${PX4_DIR:-$HOME/PX4_Firmware}"
VEHICLE="${VEHICLE:-iris}"
WORLD="${WORLD:-$REPO_ROOT/sim/worlds/powerplant_local.world}"
SDF="${SDF:-$REPO_ROOT/sim/models/iris_mid360/iris_mid360.sdf}"
MAP_PATH="${MAP_PATH:-$REPO_ROOT/assets/maps/powerplant_local.pcd}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/assets/validation/point_lio_smoke}"
MAPPING_SECONDS="${MAPPING_SECONDS:-90}"
RELOCALIZATION_SECONDS="${RELOCALIZATION_SECONDS:-30}"

mkdir -p "$LOG_DIR"

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo "[ERR] ROS Noetic not found. Run inside pipeline_inspection:noetic." >&2
  exit 1
fi

source_compat() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

export REPO_ROOT PX4_DIR
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

if [[ "${BOOTSTRAP_PX4:-1}" == "1" ]]; then
  "$SCRIPT_DIR/bootstrap_px4_noetic.sh"
fi

if [[ ! -f "$SDF" ]]; then
  echo "[ERR] Mid360 SDF not found: $SDF" >&2
  echo "[ERR] Run tools/bootstrap_px4_noetic.sh, or set SDF to an available iris_mid360.sdf." >&2
  exit 1
fi
if [[ -z "${MID360_PLUGIN_DIR:-}" && -f "$HOME/mid360_sim_ws/devel/lib/liblivox_laser_simulation.so" ]]; then
  export MID360_PLUGIN_DIR="$HOME/mid360_sim_ws/devel/lib"
fi

source_compat /opt/ros/noetic/setup.bash
"$SCRIPT_DIR/create_nav_integration_ws.sh" "$WS_DIR"
(cd "$WS_DIR" && catkin_make -DCMAKE_BUILD_TYPE=Release)
source_compat "$WS_DIR/devel/setup.bash"
source_compat "$SCRIPT_DIR/use_env.sh"

rm -f "$REPO_ROOT/third_party/point_lio/PCD"/scans*.pcd

cleanup() {
  if [[ -n "${MAPPING_PID:-}" ]] && kill -0 "$MAPPING_PID" 2>/dev/null; then
    kill -INT "$MAPPING_PID" 2>/dev/null || true
    wait "$MAPPING_PID" 2>/dev/null || true
  fi
  if [[ -n "${RELOC_PID:-}" ]] && kill -0 "$RELOC_PID" 2>/dev/null; then
    kill -INT "$RELOC_PID" 2>/dev/null || true
    wait "$RELOC_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[INFO] Starting Point-LIO mapping smoke"
roslaunch navigation_bringup point_lio_mapping.launch \
  world:="$WORLD" vehicle:="$VEHICLE" sdf:="$SDF" gui:=false rviz:=false \
  >"$LOG_DIR/point_lio_mapping.log" 2>&1 &
MAPPING_PID=$!

sleep 25
rostopic type /livox/lidar | tee "$LOG_DIR/livox_lidar_type.txt"
timeout 30 rostopic echo -n 1 /Odometry > "$LOG_DIR/point_lio_odom_first.yaml"
timeout 30 rostopic echo -n 1 /cloud_registered > "$LOG_DIR/point_lio_cloud_first.yaml"

"$SCRIPT_DIR/offboard_preflight.sh" >"$LOG_DIR/offboard_preflight.log" 2>&1 || true
timeout "$MAPPING_SECONDS" python3 "$SCRIPT_DIR/sitl_square_mission.py" \
  --altitude 1.5 --side 2.0 --hold 8.0 >"$LOG_DIR/sitl_square_mission.log" 2>&1 || true

kill -INT "$MAPPING_PID" 2>/dev/null || true
wait "$MAPPING_PID" 2>/dev/null || true
MAPPING_PID=""
sleep 5

if [[ ! -s "$REPO_ROOT/third_party/point_lio/PCD/scans.pcd" ]]; then
  echo "[ERR] Point-LIO did not write a non-empty PCD map." >&2
  exit 1
fi

rosrun map_tools prepare_pcd_map \
  --input_dir "$REPO_ROOT/third_party/point_lio/PCD" \
  --output "$MAP_PATH" \
  --voxel_leaf 0.2 --sor | tee "$LOG_DIR/prepare_pcd_map.log"

echo "[INFO] Starting saved-map relocalization smoke"
roslaunch navigation_bringup point_lio_relocalization.launch \
  world:="$WORLD" vehicle:="$VEHICLE" sdf:="$SDF" gui:=false rviz:=false map_path:="$MAP_PATH" \
  >"$LOG_DIR/point_lio_relocalization.log" 2>&1 &
RELOC_PID=$!

sleep "$RELOCALIZATION_SECONDS"
timeout 30 rostopic echo -n 1 /Odometry > "$LOG_DIR/relocalization_odom_first.yaml"
timeout 30 rostopic echo -n 1 /cloud_registered > "$LOG_DIR/relocalization_cloud_first.yaml"

echo "[OK] Point-LIO smoke artifacts written to $LOG_DIR"
