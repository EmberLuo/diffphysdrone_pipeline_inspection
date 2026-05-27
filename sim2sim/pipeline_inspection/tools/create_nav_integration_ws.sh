#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WS_DIR="${1:-$REPO_ROOT/integration_ws}"
SRC_DIR="$WS_DIR/src"

mkdir -p "$SRC_DIR"

link_pkg() {
  local src="$1"
  local name
  name="$(basename "$src")"
  if [[ ! -e "$src/package.xml" ]]; then
    echo "[WARN] skip missing package: $src" >&2
    return
  fi
  ln -sfn "$src" "$SRC_DIR/$name"
  echo "[OK] linked $name"
}

link_pkg "$REPO_ROOT/ros/livox_ros_driver2"
link_pkg "$REPO_ROOT/third_party/point_lio"
link_pkg "$REPO_ROOT/third_party/fast_lio"
link_pkg "$REPO_ROOT/ros/camera_pose_node"

# Fast-Drone planner stack. Do not link FUEL packages here.
link_pkg "$REPO_ROOT/third_party/fast_drone_250/src/utils/quadrotor_msgs"
link_pkg "$REPO_ROOT/third_party/fast_drone_250/src/planner/traj_utils"
link_pkg "$REPO_ROOT/third_party/fast_drone_250/src/planner/plan_env"
link_pkg "$REPO_ROOT/third_party/fast_drone_250/src/planner/path_searching"
link_pkg "$REPO_ROOT/third_party/fast_drone_250/src/planner/bspline_opt"
link_pkg "$REPO_ROOT/third_party/fast_drone_250/src/planner/plan_manage"

# Project ROS packages.
link_pkg "$REPO_ROOT/ros/map_tools"
link_pkg "$REPO_ROOT/ros/pcd_localization"
link_pkg "$REPO_ROOT/ros/global_astar_planner"
link_pkg "$REPO_ROOT/ros/global_path_target_bridge"
link_pkg "$REPO_ROOT/ros/bspline_target_bridge"
link_pkg "$REPO_ROOT/ros/e2e_px4_controller"
link_pkg "$REPO_ROOT/ros/localization_mode_manager"
link_pkg "$REPO_ROOT/ros/navigation_bringup"

# Baseline/control utilities needed by comparison flows.
link_pkg "$REPO_ROOT/baselines/controller_utils/controller_msgs"
link_pkg "$REPO_ROOT/baselines/controller_utils/math_utils"
link_pkg "$REPO_ROOT/baselines/se3_controller"

# Optional external localization package. Set FAST_LIO_LOCALIZATION_DIR if it
# lives elsewhere.
LOCALIZATION_DIR="${FAST_LIO_LOCALIZATION_DIR:-$REPO_ROOT/third_party/fast_lio_localization}"
if [[ -e "$LOCALIZATION_DIR/package.xml" ]]; then
  link_pkg "$LOCALIZATION_DIR"
else
  echo "[WARN] external FAST-LIO localization package not linked: $LOCALIZATION_DIR" >&2
fi

echo
echo "Workspace created at: $WS_DIR"
echo "Build with:"
echo "  cd \"$WS_DIR\""
echo "  catkin_make -DCMAKE_BUILD_TYPE=Release"
echo "  source devel/setup.bash"
