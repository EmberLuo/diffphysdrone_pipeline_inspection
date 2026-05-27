# lidar_depth_fusion Genesis Evaluation

Sim2sim evaluation for `LidarDepthFusionModel` checkpoints using Genesis physics,
Genesis depth cameras, and WARP-based lidar raytracing.

## Setup

Requires the `diffphysdrone` conda environment with `genesis`, `warp`, and `torch`
installed.

## Single episode

```bash
python eval.py \
  --resume training_code/logs/lidar_depth_fusion/single_agent/<run>/checkpoint0004.pth \
  --target_speed 0.5 \
  --no_show_viewer
```

If `--resume` is just a checkpoint filename, `eval.py` also searches the latest
run under `training_code/logs/lidar_depth_fusion/single_agent`.

Key flags:

- `--no_odom` / `--odom`: toggle no-odometry mode (default follows config)
- `--env`: layout name from config (default: `single_nav`)
- `--smoothness`: yaw-smoothing factor (default: 0.5)
- `--gru_warmup_steps`: GRU zero-input warmup steps (default: 10)
- `--reach_threshold`: arrival distance threshold (default: 1.5m)
- `--record`: save Genesis scene video
- `--trace_policy`: dump per-step policy trace JSON

## Batch evaluation

```bash
python batch_eval.py \
  --num_runs 10 \
  --resume training_code/logs/lidar_depth_fusion/single_agent/<run>/checkpoint0004.pth
```

## Config

See `config/nav_eval.yaml` for simulation, lidar sensor, depth camera, obstacles,
controller, and termination parameters.

The depth camera is configured as raw `48x64` with `pool=4`, producing `12x16`
observations to match `training_code/training_tasks/lidar_depth_fusion`.

## Architecture

- **env.py**: `NavEnv` — Genesis scene, depth cameras, WARP lidar mesh, and `LidarSensor`
- **eval.py**: evaluation loop with `LidarDepthFusionModel`, GRU warmup, and velocity-tracking control
- **batch_eval.py**: multi-run wrapper calling `eval.py` with different seeds
