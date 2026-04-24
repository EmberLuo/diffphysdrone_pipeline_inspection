# Drone Genesis Validation Layout (Depth Camera)

This folder contains depth-camera validation tasks that use `genesis-world` to run
quadrotor scenes with a viewer, `gs.sensors.DepthCamera` for depth observations,
and the DiffPhys policy (`Model(10,6)` from repo root) with GRU inference.

It converts `a_pred/v_pred` outputs to direct **4-motor RPM** commands via a
PX4-style controller, reads drone `mass/kf/km/thrust2weight` from URDF, and
matches training-time inference settings by default (15Hz control with dt jitter
and depth noise).

## Environment

Use the existing `diffphysdrone` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffphysdrone
python -m pip install genesis-world==0.4.3
python -m pip install pyyaml
```

The current `diffphysdrone` environment has been verified with
`genesis-world 0.4.3`, imported as `genesis`.

## Run (Nav / Corridor)

From repo root:

```bash
cd /home/ember/GitHub/DiffPhysDrone
python sim2sim/drone_genesis/depth_camera/nav/eval.py --resume checkpoint0004.pth --target_speed 2.5
```

Run longer by duration (recommended):

```bash
python sim2sim/drone_genesis/depth_camera/nav/eval.py --resume checkpoint0004.pth --target_speed 2.5 --duration_sec 30
```

Headless smoke test:

```bash
python sim2sim/drone_genesis/depth_camera/nav/eval.py --resume checkpoint0004.pth --num_steps 1 --no_show_viewer
```

Record video:

```bash
python sim2sim/drone_genesis/depth_camera/nav/eval.py --resume checkpoint0004.pth --record
```

## Run (Swarm Swap)

From repo root:

```bash
cd /home/ember/GitHub/DiffPhysDrone
python sim2sim/drone_genesis/depth_camera/swarm/eval.py --resume checkpoint0004.pth --target_speed 2.5
```

Headless smoke test:

```bash
python sim2sim/drone_genesis/depth_camera/swarm/eval.py --resume checkpoint0004.pth --target_speed 2.5 --num_steps 5 --no_show_viewer
```

Depth debug visualization (OpenCV window):

```bash
python sim2sim/drone_genesis/depth_camera/swarm/eval.py --resume checkpoint0004.pth --target_speed 2.5 --show_depth
```

Run 10 episodes:

```bash
python sim2sim/drone_genesis/depth_camera/swarm/batch_eval.py --num_runs 10 --target_speed 2.5 --resume checkpoint0004.pth
```

## Layout

### Live code

```text
sim2sim/drone_genesis/depth_camera/nav/eval.py
sim2sim/drone_genesis/depth_camera/nav/config/nav_eval.yaml
sim2sim/drone_genesis/depth_camera/nav/env.py
sim2sim/drone_genesis/depth_camera/swarm/eval.py
sim2sim/drone_genesis/depth_camera/swarm/batch_eval.py
sim2sim/drone_genesis/depth_camera/swarm/compare.py
sim2sim/drone_genesis/depth_camera/swarm/env.py
sim2sim/drone_genesis/depth_camera/swarm/config/swarm_eval.yaml
sim2sim/drone_genesis/utils/controller.py
sim2sim/drone_genesis/utils/mixer.py
```

### Archived historical runs

```text
sim2sim/drone_genesis/depth_camera/archive/nav/exps_*/   — past nav evaluation episodes
sim2sim/drone_genesis/depth_camera/archive/swarm/exps_*/ — past swarm evaluation episodes
sim2sim/drone_genesis/depth_camera/archive/swarm/control_diag/ — controller diagnostic outputs
```
