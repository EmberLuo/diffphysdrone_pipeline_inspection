#!/usr/bin/env bash
# NOTE:
# This file is intended to be sourced from an interactive shell.
# Do not change the caller shell options (e.g. set -e / pipefail),
# otherwise the terminal may exit unexpectedly when any later command fails.

source /opt/ros/noetic/setup.bash

# Local, per-user dependencies (no ~/.bashrc auto-load).
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$HOME/.local/lib:${LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$HOME/.local:${CMAKE_PREFIX_PATH:-}"
if [[ -z "${GEOGRAPHICLIB_DATA:-}" || ! -f "$GEOGRAPHICLIB_DATA/geoids/egm96-5.pgm" ]]; then
  if [[ -f "$HOME/.local/share/GeographicLib/geoids/egm96-5.pgm" ]]; then
    export GEOGRAPHICLIB_DATA="$HOME/.local/share/GeographicLib"
  elif [[ -f /usr/share/GeographicLib/geoids/egm96-5.pgm ]]; then
    export GEOGRAPHICLIB_DATA="/usr/share/GeographicLib"
  else
    export GEOGRAPHICLIB_DATA="$HOME/.local/share/GeographicLib"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# PX4 SITL/Gazebo environment (optional, if PX4_Firmware exists).
PX4_DIR="${PX4_DIR:-$HOME/PX4_Firmware}"
PX4_BUILD_DIR="${PX4_BUILD_DIR:-$PX4_DIR/build/px4_sitl_default}"
if [[ -f "$PX4_DIR/Tools/setup_gazebo.bash" && -d "$PX4_BUILD_DIR" ]]; then
  source "$PX4_DIR/Tools/setup_gazebo.bash" "$PX4_DIR" "$PX4_BUILD_DIR" >/dev/null
fi

# Mid360 Gazebo plugin path (prefer repo-local to avoid cross-repo coupling).
USER_MID360_PLUGIN_DIR="${MID360_PLUGIN_DIR:-}"
DEFAULT_MID360_PLUGIN_DIR="$REPO_ROOT/third_party/gazebo_plugins"
LEGACY_MID360_PLUGIN_DIR="$HOME/GitHub/slam_and_nav/catkin_ws/devel/lib"
ACTIVE_MID360_PLUGIN_DIR=""

add_gazebo_plugin_path() {
  local p="$1"
  if [[ -d "$p" ]]; then
    case ":${GAZEBO_PLUGIN_PATH:-}:" in
      *":$p:"*) ;;
      *) export GAZEBO_PLUGIN_PATH="$p:${GAZEBO_PLUGIN_PATH:-}" ;;
    esac
  fi
}

resolve_mid360_plugin_dir() {
  local candidates=()
  local p

  if [[ -n "$USER_MID360_PLUGIN_DIR" ]]; then
    candidates+=("$USER_MID360_PLUGIN_DIR")
  fi
  candidates+=("$DEFAULT_MID360_PLUGIN_DIR" "$LEGACY_MID360_PLUGIN_DIR")

  for p in "${candidates[@]}"; do
    if [[ -f "$p/liblivox_laser_simulation.so" ]]; then
      ACTIVE_MID360_PLUGIN_DIR="$p"
      break
    fi
  done

  if [[ -z "$ACTIVE_MID360_PLUGIN_DIR" ]]; then
    echo "[WARN] Mid360 plugin not found (liblivox_laser_simulation.so)" >&2
    return
  fi

  if [[ -n "$USER_MID360_PLUGIN_DIR" && "$ACTIVE_MID360_PLUGIN_DIR" != "$USER_MID360_PLUGIN_DIR" ]]; then
    echo "[WARN] MID360_PLUGIN_DIR invalid: $USER_MID360_PLUGIN_DIR, fallback to $ACTIVE_MID360_PLUGIN_DIR" >&2
  fi
  if [[ "$ACTIVE_MID360_PLUGIN_DIR" == "$LEGACY_MID360_PLUGIN_DIR" ]]; then
    echo "[WARN] using legacy Mid360 plugin from $LEGACY_MID360_PLUGIN_DIR" >&2
  fi
}

resolve_mid360_plugin_dir
if [[ -n "$ACTIVE_MID360_PLUGIN_DIR" ]]; then
  add_gazebo_plugin_path "$ACTIVE_MID360_PLUGIN_DIR"
  export MID360_PLUGIN_DIR="$ACTIVE_MID360_PLUGIN_DIR"
fi

REPO_MODEL_DIR="$REPO_ROOT/sim/models"
if [[ -d "$REPO_MODEL_DIR" ]]; then
  case ":${GAZEBO_MODEL_PATH:-}:" in
    *":$REPO_MODEL_DIR:"*) ;;
    *) export GAZEBO_MODEL_PATH="$REPO_MODEL_DIR:${GAZEBO_MODEL_PATH:-}" ;;
  esac
fi

sanitize_colon_path_dirs() {
  local raw="$1"
  local out=""
  local p
  IFS=':' read -r -a _parts <<< "$raw"
  for p in "${_parts[@]}"; do
    [[ -z "$p" ]] && continue
    [[ -d "$p" ]] || continue
    case ":$out:" in
      *":$p:"*) ;;
      *) out="${out:+$out:}$p" ;;
    esac
  done
  echo "$out"
}

WS_DEVEL="$HOME/catkin_ws/devel_isolated"

source_pkg_setup() {
  local pkg="$1"
  local setup_file="$WS_DEVEL/$pkg/setup.bash"
  if [[ -f "$setup_file" ]]; then
    CATKIN_SETUP_UTIL_ARGS=--extend source "$setup_file"
  else
    echo "[WARN] missing: $setup_file" >&2
  fi
}

# Runtime chain used by README flow.
source_pkg_setup fast_lio
source_pkg_setup camera_pose_node
source_pkg_setup se3_controller
source_pkg_setup livox_ros_driver2
source_pkg_setup libmavconn
source_pkg_setup uuid_msgs
source_pkg_setup geographic_msgs
source_pkg_setup mavros_msgs
source_pkg_setup mavros
source_pkg_setup lkh_tsp_solver
source_pkg_setup waypoint_generator
source_pkg_setup exploration_manager

# Optional clean integration workspace for Fast-Drone/E2E navigation.
INTEGRATION_WS_SETUP="$REPO_ROOT/integration_ws/devel/setup.bash"
if [[ -f "$INTEGRATION_WS_SETUP" ]]; then
  CATKIN_SETUP_UTIL_ARGS=--extend source "$INTEGRATION_WS_SETUP"
fi

# Keep PX4 ROS package paths after catkin setup scripts modify ROS_PACKAGE_PATH.
if [[ -d "$PX4_DIR" ]]; then
  case ":${ROS_PACKAGE_PATH:-}:" in
    *":$PX4_DIR:"*) ;;
    *) export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH:-}:$PX4_DIR" ;;
  esac
  case ":${ROS_PACKAGE_PATH:-}:" in
    *":$PX4_DIR/Tools/sitl_gazebo:"*) ;;
    *) export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH:-}:$PX4_DIR/Tools/sitl_gazebo" ;;
  esac
fi

# Finalize Gazebo plugin path AFTER all setup.bash files to avoid stale overrides.
if [[ -n "$ACTIVE_MID360_PLUGIN_DIR" ]]; then
  add_gazebo_plugin_path "$ACTIVE_MID360_PLUGIN_DIR"
fi
export GAZEBO_PLUGIN_PATH="$(sanitize_colon_path_dirs "${GAZEBO_PLUGIN_PATH:-}")"

export REPO_ROOT

echo "[OK] Environment loaded for pipeline_inspection"
