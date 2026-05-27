#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/use_env.sh" >/dev/null

if ! timeout 3s rostopic list >/dev/null 2>&1; then
  echo "[ERR] ROS master is not reachable. Start PX4 + MAVROS first."
  exit 1
fi

set_param() {
  local name="$1"
  local value="$2"
  echo "[SET] ${name}=${value}"
  rosrun mavros mavparam set "${name}" "${value}" >/dev/null
  rosrun mavros mavparam get "${name}"
}

echo "[INFO] Applying OFFBOARD-without-RC safety params..."
set_param COM_RC_IN_MODE 4
set_param COM_RCL_EXCEPT 4
set_param NAV_RCL_ACT 0
set_param COM_OBL_RC_ACT 0
set_param COM_ARM_WO_GPS 1

echo "[INFO] Current MAVROS state:"
rostopic echo -n 1 /mavros/state

echo "[OK] Preflight parameters applied."
echo "[NEXT] Keep setpoint stream running, then switch OFFBOARD and arm."
