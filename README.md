# Drone Navigation Training Code

This repository contains GPU-accelerated training and sim2sim evaluation code
for agile quadrotor navigation. It keeps a legacy depth-camera training task and
adds a NavRL-style LiDAR navigation task on top of the same differentiable
quadrotor dynamics.

The current codebase is organized around task-specific training entrypoints
with a shared training loop, CUDA sensor/dynamics kernels, and Genesis-based
validation utilities.

## Features

- Differentiable CUDA quadrotor dynamics and obstacle rendering.
- Legacy depth-camera policy training.
- NavRL-style LiDAR policy training with scan observation `[B, 1, 36, 4]`.
- Shared training loop for depth and LiDAR tasks.
- Single-agent and multi-agent training modes.
- TensorBoard logging and checkpoint saving.
- Genesis sim2sim evaluation utilities under `sim2sim/drone_genesis`.

## Repository Layout

```text
training_code/
  main_cuda.py                          # legacy depth-camera entrypoint
  env_cuda.py                           # CUDA-backed training environment
  model.py                              # depth-camera policy network
  src/                                  # CUDA/C++ extension
  training_tasks/
    common/train_loop.py                # shared argparse/training/loss/logging loop
    depth_camera/train.py               # depth-camera task entrypoint
    lidar_navrl/
      train.py                          # LiDAR task entrypoint
      env.py                            # LiDAR observation wrapper
      model.py                          # LiDAR policy network

sim2sim/drone_genesis/                  # Genesis validation and utilities
configs/                                # legacy config files
```

## Environment

The current setup has been used with:

- Python 3.11
- PyTorch 2.2.2
- CUDA 11.8
- `genesis-world==0.4.3` for Genesis sim2sim evaluation

Example environment activation:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffphysdrone
```

Install Genesis dependencies if you want to run sim2sim validation:

```bash
python -m pip install genesis-world==0.4.3
python -m pip install pyyaml
```

## Build CUDA Extension

From the repository root:

```bash
cd training_code/src
python setup.py build_ext --inplace
cd ../..
```

The training code imports the extension as `quadsim_cuda`.

## Training

### LiDAR Navigation

Single-agent LiDAR training:

```bash
python training_code/training_tasks/lidar_navrl/train.py \
  --no_odom \
  $(cat training_code/training_tasks/lidar_navrl/configs/single_agent.args)
```

Multi-agent LiDAR training:

```bash
python training_code/training_tasks/lidar_navrl/train.py \
  --no_odom \
  $(cat training_code/training_tasks/lidar_navrl/configs/multi_agent.args)
```

The LiDAR task uses these default sensor settings:

```text
range: 4.0 m
horizontal beams: 36
vertical beams: 4
vertical FoV: [-10, 20] degrees
observation shape: [B, 1, 36, 4]
```

Logs and checkpoints are written under:

```text
training_code/logs/lidar_navrl/<single_agent|multi_agent>/<timestamp>/
```

### Depth-Camera Navigation

Single-agent depth-camera training:

```bash
python training_code/training_tasks/depth_camera/train.py \
  --no_odom \
  $(cat training_code/training_tasks/depth_camera/configs/single_agent.args)
```

Multi-agent depth-camera training:

```bash
python training_code/training_tasks/depth_camera/train.py \
  --no_odom \
  $(cat training_code/training_tasks/depth_camera/configs/multi_agent.args)
```

Logs and checkpoints are written under:

```text
training_code/logs/depth_camera/<single_agent|multi_agent>/<timestamp>/
```

The legacy depth-camera entrypoint is still available:

```bash
python training_code/main_cuda.py --single --no_odom
```

## TensorBoard

View LiDAR training logs:

```bash
tensorboard --logdir training_code/logs/lidar_navrl --host 0.0.0.0 --port 6006
```

View depth-camera training logs:

```bash
tensorboard --logdir training_code/logs/depth_camera --host 0.0.0.0 --port 6006
```

Open:

```text
http://localhost:6006
```

## Genesis Sim2Sim

Genesis validation code lives in `sim2sim/drone_genesis`.

See:

```text
sim2sim/drone_genesis/README.md
```

Example nav evaluation command from the repository root:

```bash
python sim2sim/drone_genesis/nav/eval.py \
  --resume training_code/logs/depth_camera/single_agent/<run>/checkpoint0004.pth \
  --target_speed 2.5
```

## Notes

- `--single` enables single-agent training; without it, the environment uses
  multi-agent groups.
- `--no_odom` removes local velocity from the policy state.
- Use `--log_root`, `--experiment_name`, `--run_name`, and `--save_every` to
  control output locations and checkpoint cadence.
- Training checkpoints, TensorBoard events, compiled extensions, and evaluation
  outputs are ignored by `.gitignore`.

## Acknowledgement

This project was originally inspired by the
[DiffPhysDrone project](https://henryhuyu.github.io/DiffPhysDrone_Web/). The
current repository has been substantially reorganized and extended with
task-specific training code, a NavRL-style LiDAR task, shared training
infrastructure, and Genesis sim2sim utilities.
