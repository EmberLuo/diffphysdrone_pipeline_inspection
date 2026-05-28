#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! source "${SCRIPT_DIR}/use_env.sh" >/dev/null 2>&1; then
  echo "[WARN] Environment helper returned non-zero; continuing with current shell environment."
fi

if ! timeout 3s rostopic list >/dev/null 2>&1; then
  echo "[ERR] ROS master is not reachable. Start PX4 + MAVROS first."
  exit 1
fi

param_exists() {
  local name="$1"
  rosrun mavros mavparam get "${name}" >/dev/null 2>&1
}

set_param() {
  local name="$1"
  local value="$2"
  local required="${3:-optional}"
  if ! param_exists "${name}"; then
    if [[ "${required}" == "required" ]]; then
      echo "[ERR] Required PX4 param is not available: ${name}" >&2
      exit 1
    fi
    echo "[SKIP] ${name} is not available on this PX4 build."
    return 0
  fi
  echo "[SET] ${name}=${value}"
  if ! rosrun mavros mavparam set "${name}" "${value}" >/dev/null; then
    if [[ "${required}" == "required" ]]; then
      echo "[ERR] Failed to set required PX4 param: ${name}" >&2
      exit 1
    fi
    echo "[WARN] Failed to set optional PX4 param: ${name}; continuing."
    return 0
  fi
  rosrun mavros mavparam get "${name}" || true
}

echo "[INFO] Applying OFFBOARD-without-RC safety params..."
set_param COM_RC_IN_MODE 4 required
set_param COM_RCL_EXCEPT 4
set_param NAV_RCL_ACT 0
set_param COM_OBL_RC_ACT 0
set_param COM_ARM_WO_GPS 1

echo "[INFO] Current MAVROS state:"
rostopic echo -n 1 /mavros/state

echo "[OK] Preflight parameters applied."
echo "[NEXT] Keep setpoint stream running, then switch OFFBOARD and arm."
