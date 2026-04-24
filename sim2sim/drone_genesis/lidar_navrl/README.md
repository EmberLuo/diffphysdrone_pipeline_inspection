# lidar_navrl Genesis Evaluation

Sim2sim evaluation for the lidar_navrl checkpoint using Genesis physics and WARP-based lidar raytracing.

## Setup

Requires the `diffphysdrone` conda environment with `genesis`, `warp`, and `torch` installed.

## Single episode

```bash
python eval.py --resume <checkpoint_path> --target_speed 0.5 --no_odom --no_show_viewer
```

Key flags:

- `--no_odom` / `--odom`: toggle no-odometry mode (default: no_odom, matching training)
- `--env`: layout name from config (default: `single_nav`)
- `--smoothness`: yaw-smoothing factor (default: 0.5)
- `--gru_warmup_steps`: GRU zero-input warmup steps (default: 10)
- `--reach_threshold`: arrival distance threshold (default: 1.5m)
- `--record`: save Genesis scene video
- `--trace_policy`: dump per-step policy trace JSON

## Batch evaluation

```bash
python batch_eval.py --num_runs 10 --target_speed 0.5 --resume <checkpoint_path>
```

## Compare with AirSim logs

```bash
python compare.py --airsim_root <airsim_exps_dir> --genesis_root <genesis_exps_dir>
```

Outputs JSON with per-drone time MAE, completion rates, and collision totals, plus a pass/fail check against configurable thresholds.

## Config

See `config/nav_eval.yaml` for simulation, lidar sensor, obstacles, controller, and termination parameters.

## Architecture

- **env.py**: `NavEnv` — Genesis scene + WARP lidar mesh + LidarSensor
- **eval.py**: evaluation loop with LidarNavRLModel, GRU warmup, velocity-tracking control
- **batch_eval.py**: multi-run wrapper calling eval.py with different seeds
- **compare.py**: log parser comparing AirSim vs Genesis results
