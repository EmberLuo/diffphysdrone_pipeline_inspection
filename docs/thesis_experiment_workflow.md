# Thesis experiment export workflow

This repository already contains the algorithm/runtime pieces for most Chapter 7
experiments. The missing layer is repeatable experiment recording and export.

## 0. Build the ROS workspace

```bash
cd /home/ember/GitHub/diffphysdrone_pipeline_inspection/sim2sim/pipeline_inspection
export REPO_ROOT=$(pwd)
./tools/create_nav_integration_ws.sh
cd integration_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
source ../tools/use_env.sh
```

## 1. Point-LIO mapping metrics, table 7-3

Run mapping:

```bash
cd /home/ember/GitHub/diffphysdrone_pipeline_inspection/sim2sim/pipeline_inspection
source integration_ws/devel/setup.bash
source tools/use_env.sh
roslaunch navigation_bringup pipe_factory_mapping.launch rviz:=true gui:=false
```

In another terminal, record metrics:

```bash
rosrun navigation_bringup point_lio_metrics_recorder.py \
  _odom_topic:=/Odometry \
  _gt_odom_topic:=/mavros/local_position/odom \
  _pcd_dir:=$REPO_ROOT/third_party/point_lio/PCD \
  _prepared_map_path:=$REPO_ROOT/assets/maps/pipe_factory_local.pcd \
  _output_dir:=$REPO_ROOT/assets/validation/thesis_pointlio
```

Fly the square mission:

```bash
python3 tools/sitl_square_mission.py --altitude 1.5 --side 2.0 --hold 8.0
```

Stop mapping cleanly, then prepare the map:

```bash
rosrun map_tools prepare_pcd_map \
  --input_dir $REPO_ROOT/third_party/point_lio/PCD \
  --output $REPO_ROOT/assets/maps/pipe_factory_local.pcd \
  --voxel_leaf 0.2 --sor | tee $REPO_ROOT/assets/validation/thesis_pointlio/prepare_pcd_map.log
```

Saved outputs:

- `assets/validation/thesis_pointlio/point_lio_samples.csv`
- `assets/validation/thesis_pointlio/point_lio_summary.csv`
- `third_party/point_lio/PCD/scans*.pcd`
- `assets/maps/pipe_factory_local.pcd`

Use RViz screenshots or trajectory plotting from `point_lio_samples.csv` for the
Point-LIO map and trajectory figures.

## 2. GNSS mode switching, table 7-4

The code already has a fault injector and metrics recorder.

```bash
roslaunch navigation_bringup gnss_mode_test.launch \
  output_dir:=$REPO_ROOT/assets/validation/thesis_gnss_modes \
  normal_duration:=20 degraded_duration:=20 \
  normal_noise_std:=0.05 degraded_noise_std:=0.5 \
  jump_magnitude:=1.5 jump_period:=5.0
```

Saved outputs:

- `assets/validation/thesis_gnss_modes/gnss_mode_samples.csv`
- `assets/validation/thesis_gnss_modes/gnss_mode_summary.csv`

These fill the GNSS normal/degraded/lost table.

## 3. Global planning metrics, table 7-5

Start saved-map navigation:

```bash
roslaunch navigation_bringup pipe_factory_navigation.launch target_speed:=1.0
```

Record planning metrics:

```bash
rosrun navigation_bringup global_planning_metrics_recorder.py \
  _map_path:=$REPO_ROOT/assets/maps/pipe_factory_local.pcd \
  _resolution:=0.25 \
  _inflation_radius:=0.45 \
  _output_dir:=$REPO_ROOT/assets/validation/thesis_global_planning/res_025_infl_045
```

For a parameter scan, relaunch navigation with the exposed planner args:

```bash
roslaunch navigation_bringup pipe_factory_navigation.launch \
  planner_resolution:=0.25 \
  planner_obstacles_inflation:=0.45 \
  target_speed:=1.0
```

Publish a goal from RViz, or from a terminal:

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
"header:
  frame_id: 'world'
pose:
  position: {x: 8.0, y: 3.0, z: 1.8}
  orientation: {w: 1.0}"
```

Saved outputs:

- `global_planning_samples.csv`
- `global_planning_summary.csv`

Repeat the launch with different `planner_resolution` and
`planner_obstacles_inflation` values. Save each run under a different output
directory.

## 4. Local avoidance training curves and ablation, table 7-6

Existing TensorBoard logs can be exported directly:

```bash
cd /home/ember/GitHub/diffphysdrone_pipeline_inspection
python tools/export_thesis_training_metrics.py \
  --output_dir thesis_outputs/local_avoidance \
  --figures_dir /home/ember/桌面/thesis/thesis-latex/Figures
```

Saved outputs:

- `thesis_outputs/local_avoidance/training_scalars.csv`
- `thesis_outputs/local_avoidance/local_avoidance_ablation_metrics.csv`
- `thesis_outputs/local_avoidance/local_avoidance_metrics.json`
- regenerated `training_depth_success.png`, `training_depth_errors.png`,
  `training_depth_safety.png`

If you run additional ablations, pass them with:

```bash
python tools/export_thesis_training_metrics.py \
  --run "CNN-only=/path/to/run" \
  --run "No dynamic safety=/path/to/run"
```

## 5. Robustness experiments, table 7-7

Run the same policy under low/mid/high disturbance settings and save each batch:

```bash
python sim2sim/drone_genesis/lidar_depth_fusion/batch_eval.py \
  --resume training_code/logs/depth_camera/single_agent_odom/20260504_122043_dob_hover_dp_depth_odom/checkpoint0004.pth \
  --num_runs 10 --seed_start 0 --target_speed 0.5 \
  --output_root thesis_outputs/robustness/dob_high
```

The current Genesis eval writes per-episode `log`, `traj_history.json`, optional
`policy_trace.json`, and sensor videos. A remaining useful improvement is a
small aggregator that converts those eval folders into the 9-row robustness
table. This has been added:

```bash
python tools/export_genesis_eval_metrics.py \
  --input_root thesis_outputs/robustness/dob_high/exps_0.5 \
  --method "dynamic safety+DOB" \
  --disturbance high \
  --output_dir thesis_outputs/robustness/dob_high_summary
```

Saved outputs:

- `genesis_eval_episodes.csv`
- `genesis_eval_summary.csv`

Run this once per method/disturbance folder, then combine the summary rows into
table 7-7.

## 6. Complete closed-loop simulation, table 7-8

Start the full chain:

```bash
roslaunch navigation_bringup pipe_factory_navigation.launch target_speed:=1.0
```

Record closed-loop metrics:

```bash
rosrun navigation_bringup closed_loop_metrics_recorder.py \
  _map_path:=$REPO_ROOT/assets/maps/pipe_factory_local.pcd \
  _goal_radius:=0.5 \
  _output_dir:=$REPO_ROOT/assets/validation/thesis_closed_loop/run01
```

Publish the task goal and let the mission run. Saved outputs:

- `closed_loop_samples.csv`
- `closed_loop_events.csv`
- `closed_loop_summary.csv`

Use `closed_loop_samples.csv` for the real task trajectory figure and
`closed_loop_events.csv` for the state/timing figure.
