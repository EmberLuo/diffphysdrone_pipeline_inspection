# ROS Packages

This directory contains the project-owned ROS packages for the saved-map
navigation chain:

```text
point_lio -> /Odometry, /cloud_registered, PCD scans
pcd_localization -> /Odometry, /cloud_registered
global_astar_planner -> /global_path
global_path_target_bridge -> /e2e/local_target
e2e_px4_controller -> /mavros/setpoint_raw/attitude
navigation_bringup -> launch entrypoints
```

`bspline_target_bridge` remains for the EGO/B-spline baseline. `livox_ros_driver2`
is a minimal message compatibility package for this integration workspace. The
FAST-LIO package is still linkable as a legacy fallback, while Point-LIO is the
default mapping backend for thesis-aligned experiments.

Build from the project root:

```bash
cd /path/to/pipeline_inspection
export REPO_ROOT=$(pwd)
./tools/create_nav_integration_ws.sh
cd integration_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

Do not link `third_party/fuel/fuel_planner` into the default workspace together
with the Fast-Drone planner stack, because those upstream trees contain
overlapping package names.

Mapping writes raw Point-LIO PCD scans under `third_party/point_lio/PCD/`. Prepare
the runtime map with:

```bash
rosrun map_tools prepare_pcd_map \
  --input_dir ${REPO_ROOT}/third_party/point_lio/PCD \
  --output ${REPO_ROOT}/assets/maps/powerplant_local.pcd \
  --voxel_leaf 0.2 --sor
```

Default mapping launch:

```bash
roslaunch navigation_bringup point_lio_mapping.launch gui:=false rviz:=false
```

Default navigation launch:

```bash
roslaunch navigation_bringup global_astar_navigation.launch
```
