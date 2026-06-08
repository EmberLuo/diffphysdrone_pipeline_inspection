#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WS_DIR="${WS_DIR:-$REPO_ROOT/integration_ws}"
PX4_DIR="${PX4_DIR:-$HOME/PX4_Firmware}"
VEHICLE="${VEHICLE:-iris}"
SDF="${SDF:-$REPO_ROOT/sim/models/iris_mid360/iris_mid360.sdf}"
WORLD="${WORLD:-$REPO_ROOT/sim/worlds/pipe_factory_local.world}"
MAP_PATH="${MAP_PATH:-$REPO_ROOT/assets/maps/pipe_factory_local.pcd}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/assets/validation/pipe_factory_dense_mapping}"
VIDEO_PATH="${VIDEO_PATH:-$LOG_DIR/gazebo_rviz_dense_pipe_mapping.mp4}"
MISSION_TIMEOUT="${MISSION_TIMEOUT:-900}"
RECORD_SIZE="${RECORD_SIZE:-1920x1080}"
DISPLAY="${DISPLAY:-:1}"

mkdir -p "$LOG_DIR"

DISPLAY_NUM="${DISPLAY#:}"
DISPLAY_NUM="${DISPLAY_NUM%%.*}"
DISPLAY_SOCKET="/tmp/.X11-unix/X${DISPLAY_NUM}"
if [[ ! -S "$DISPLAY_SOCKET" ]]; then
  if ! command -v Xvfb >/dev/null 2>&1; then
    echo "[ERR] DISPLAY=$DISPLAY is not available and Xvfb is not installed." >&2
    exit 1
  fi
  echo "[INFO] Starting Xvfb on $DISPLAY"
  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
  Xvfb "$DISPLAY" -screen 0 "${RECORD_SIZE}x24" -ac +extension GLX +render -noreset \
    >"$LOG_DIR/xvfb.log" 2>&1 &
  XVFB_PID=$!
  sleep 2
fi

source_compat() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

append_px4_ros_paths() {
  if [[ ! -d "$PX4_DIR" ]]; then
    return
  fi
  case ":${ROS_PACKAGE_PATH:-}:" in
    *":$PX4_DIR:"*) ;;
    *) export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH:-}:$PX4_DIR" ;;
  esac
  if [[ -d "$PX4_DIR/Tools/sitl_gazebo" ]]; then
    case ":${ROS_PACKAGE_PATH:-}:" in
      *":$PX4_DIR/Tools/sitl_gazebo:"*) ;;
      *) export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH:-}:$PX4_DIR/Tools/sitl_gazebo" ;;
    esac
  fi
}

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo "[ERR] ROS Noetic not found. Run inside pipeline_inspection:noetic." >&2
  exit 1
fi

export REPO_ROOT PX4_DIR DISPLAY
export DISABLE_ROS1_EOL_WARNINGS="${DISABLE_ROS1_EOL_WARNINGS:-1}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

source_compat /opt/ros/noetic/setup.bash
if [[ ! -f "$WS_DIR/devel/setup.bash" ]]; then
  "$SCRIPT_DIR/create_nav_integration_ws.sh" "$WS_DIR"
  (cd "$WS_DIR" && catkin_make -DCMAKE_BUILD_TYPE=Release)
fi
source_compat "$WS_DIR/devel/setup.bash"
source_compat "$SCRIPT_DIR/use_env.sh"
source_compat "$WS_DIR/devel/setup.bash"
append_px4_ros_paths

if ! rospack find navigation_bringup >/dev/null 2>&1; then
  "$SCRIPT_DIR/create_nav_integration_ws.sh" "$WS_DIR"
  (cd "$WS_DIR" && catkin_make -DCMAKE_BUILD_TYPE=Release)
  source_compat "$WS_DIR/devel/setup.bash"
  source_compat "$SCRIPT_DIR/use_env.sh"
  source_compat "$WS_DIR/devel/setup.bash"
  append_px4_ros_paths
fi

rm -f "$REPO_ROOT/third_party/point_lio/PCD"/scans*.pcd

cleanup() {
  for pid in "${MISSION_PID:-}" "${MARKER_PID:-}" "${RECORD_PID:-}" "${LAUNCH_PID:-}" "${XVFB_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

echo "[INFO] Starting dense pipe factory mapping with Gazebo and RViz"
roslaunch navigation_bringup pipe_factory_mapping.launch gui:=true rviz:=true \
  world:="$WORLD" vehicle:="$VEHICLE" sdf:="$SDF" \
  >"$LOG_DIR/pipe_factory_mapping.launch.log" 2>&1 &
LAUNCH_PID=$!

sleep 35

"$SCRIPT_DIR/layout_gazebo_rviz_windows.py" \
  --display "$DISPLAY" --width "${RECORD_SIZE%x*}" --height "${RECORD_SIZE#*x}" \
  >"$LOG_DIR/layout_gazebo_rviz_windows.log" 2>&1 || true

python3 -u "$SCRIPT_DIR/gazebo_drone_visibility_marker.py" \
  >"$LOG_DIR/gazebo_drone_visibility_marker.log" 2>&1 &
MARKER_PID=$!
sleep 3

echo "[INFO] Recording $DISPLAY to $VIDEO_PATH"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -video_size "$RECORD_SIZE" -framerate 30 -f x11grab -i "$DISPLAY+0,0" \
    -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p "$VIDEO_PATH" \
    >"$LOG_DIR/ffmpeg.log" 2>&1 &
  RECORD_PID=$!
else
  echo "[WARN] ffmpeg is not available in this environment; run a host-side recorder or install ffmpeg." \
    | tee "$LOG_DIR/ffmpeg.log"
fi

"$SCRIPT_DIR/offboard_preflight.sh" >"$LOG_DIR/offboard_preflight.log" 2>&1 || true
timeout "$MISSION_TIMEOUT" python3 -u "$SCRIPT_DIR/sitl_pipe_factory_coverage_mission.py" \
  --land >"$LOG_DIR/sitl_pipe_factory_coverage_mission.log" 2>&1 &
MISSION_PID=$!
wait "$MISSION_PID" || true
MISSION_PID=""

sleep 8
if [[ -n "${RECORD_PID:-}" ]]; then
  kill -INT "$RECORD_PID" 2>/dev/null || true
  wait "$RECORD_PID" 2>/dev/null || true
  RECORD_PID=""
fi

kill -INT "$LAUNCH_PID" 2>/dev/null || true
wait "$LAUNCH_PID" 2>/dev/null || true
LAUNCH_PID=""
sleep 5

if [[ -s "$REPO_ROOT/third_party/point_lio/PCD/scans.pcd" ]]; then
  rosrun map_tools prepare_pcd_map \
    --input_dir "$REPO_ROOT/third_party/point_lio/PCD" \
    --output "$MAP_PATH" \
    --voxel_leaf 0.2 --sor | tee "$LOG_DIR/prepare_pcd_map.log"
else
  echo "[WARN] Point-LIO did not write scans.pcd; leaving existing map untouched." | tee "$LOG_DIR/prepare_pcd_map.log"
fi

echo "[OK] Dense mapping video: $VIDEO_PATH"
