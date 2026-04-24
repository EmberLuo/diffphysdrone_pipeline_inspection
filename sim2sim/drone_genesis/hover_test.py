"""Hover test for PX4StyleRPMController + Genesis physics.

Spawns 1 drone at (0, 0, 1), commands roll=0 pitch=0 yaw=0 throttle=hover_throttle,
and tracks position/attitude for `duration_sec` seconds.

Pass criteria:
  - height stays within [0.7, 1.3] m (±0.3m from target)
  - roll/pitch stay within ±10°
  - xy drift stays within ±0.5m

Usage:
  python sim2sim/drone_genesis/hover_test.py
  python sim2sim/drone_genesis/hover_test.py --duration_sec 10 --show_viewer
  python sim2sim/drone_genesis/hover_test.py --no_show_viewer
"""

import argparse
import json
import math
import sys
from pathlib import Path

import genesis as gs
import numpy as np
import torch
import yaml
from genesis.utils.geom import quat_to_xyz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2sim.drone_genesis.utils.controller import PX4StyleRPMController
from sim2sim.drone_genesis.depth_camera.swarm.env import _read_vehicle_params_from_urdf, ASSET_DIR

CONFIG_PATH = str(Path(__file__).resolve().parent / "depth_camera" / "swarm" / "config" / "swarm_eval.yaml")
DRONE_URDF = str(ASSET_DIR / "drone_ex1" / "drone_ex1.urdf")


def main():
    parser = argparse.ArgumentParser(description="Hover test for Genesis controller")
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--duration_sec", type=float, default=5.0)
    parser.add_argument("--target_height", type=float, default=1.0)
    parser.add_argument("--show_viewer", action="store_true", default=False)
    parser.add_argument("--no_show_viewer", dest="show_viewer", action="store_false")
    parser.set_defaults(show_viewer=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["sim"]
    scene_cfg = cfg["scene"]
    dt = float(sim_cfg.get("dt", 0.01))
    substeps = int(sim_cfg.get("substeps", 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vehicle_params = _read_vehicle_params_from_urdf(DRONE_URDF)
    controller = PX4StyleRPMController(
        cfg=cfg["controller"], num_envs=1, device=device, vehicle_params=vehicle_params,
    )

    gs.init(logging_level="error")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(0.0, 3.0, 3.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=45,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=False,
        ),
        show_viewer=args.show_viewer,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    drone = scene.add_entity(morph=gs.morphs.Drone(file=DRONE_URDF))
    scene.build(n_envs=1)
    env0 = torch.tensor([0], device=device, dtype=gs.tc_int)

    # Place drone at target height, zero velocity.
    drone.set_pos(
        torch.tensor([[0.0, 0.0, args.target_height]], device=device, dtype=gs.tc_float),
        zero_velocity=True, envs_idx=env0,
    )
    drone.zero_all_dofs_velocity(envs_idx=env0)

    # Let one physics tick settle, then clear any induced velocity.
    scene.step()
    drone.zero_all_dofs_velocity(envs_idx=env0)

    hover_rpm = controller.hover_rpm
    max_steps = int(args.duration_sec / dt)

    pos_log = []
    euler_log = []

    roll_cmd = torch.zeros(1, device=device, dtype=torch.float32)
    pitch_cmd = torch.zeros(1, device=device, dtype=torch.float32)
    yaw_cmd = torch.zeros(1, device=device, dtype=torch.float32)
    throttle_cmd = torch.full((1,), controller.hover_throttle, device=device, dtype=torch.float32)

    controller.reset()

    print(f"[hover_test] dt={dt} duration={args.duration_sec}s max_steps={max_steps}")
    print(f"[hover_test] hover_rpm={hover_rpm:.1f} hover_throttle={controller.hover_throttle:.3f}")
    print(f"[hover_test] mass={vehicle_params['mass']} kf={vehicle_params['kf']} twr={vehicle_params['thrust2weight']}")

    for step_idx in range(max_steps):
        sim_time = step_idx * dt

        base_quat = drone.get_quat()[0].to(device=device, dtype=torch.float32)
        base_ang_vel = drone.get_ang()[0].to(device=device, dtype=torch.float32)

        rpm_cmd = controller.compute_rpm_from_rpy_throttle(
            roll_des=roll_cmd,
            pitch_des=pitch_cmd,
            yaw_des=yaw_cmd,
            throttle_des=throttle_cmd,
            base_quat=base_quat[None],
            base_ang_vel=base_ang_vel[None],
            dt=dt,
        )

        drone.set_propellels_rpm(rpm_cmd)
        scene.step()

        pos = drone.get_pos()[0].cpu().numpy()
        quat = drone.get_quat()[0].cpu().to(torch.float32)
        euler_rad = quat_to_xyz(quat[None])[0].cpu().numpy()
        euler_deg = euler_rad * 180.0 / math.pi

        pos_log.append(pos.tolist())
        euler_log.append(euler_deg.tolist())

        if step_idx % 50 == 0:
            print(
                f"  t={sim_time:.2f}s  pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
                f"rpy=[{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}]°"
            )

    # Evaluate pass/fail.
    pos_arr = np.array(pos_log)
    euler_arr = np.array(euler_log)

    height_min, height_max = pos_arr[:, 2].min(), pos_arr[:, 2].max()
    height_mean = pos_arr[:, 2].mean()
    xy_drift = np.max(np.abs(pos_arr[:, :2]))
    roll_max = np.max(np.abs(euler_arr[:, 0]))
    pitch_max = np.max(np.abs(euler_arr[:, 1]))

    h_min_allowed = args.target_height - 0.3
    h_max_allowed = args.target_height + 0.3
    h_pass = abs(height_mean - args.target_height) < 0.3 and height_min > h_min_allowed and height_max < h_max_allowed
    xy_pass = xy_drift < 0.5
    att_pass = roll_max < 10.0 and pitch_max < 10.0

    print()
    print("--- Hover test results ---")
    print(f"  Height:  mean={height_mean:.3f}m  min={height_min:.3f}m  max={height_max:.3f}m  {'PASS' if h_pass else 'FAIL'}")
    print(f"  XY drift: max={xy_drift:.4f}m  {'PASS' if xy_pass else 'FAIL'}")
    print(f"  Roll max: {roll_max:.2f}°  Pitch max: {pitch_max:.2f}°  {'PASS' if att_pass else 'FAIL'}")
    overall = h_pass and xy_pass and att_pass
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")

    out_dir = Path(__file__).resolve().parent / "hover_test_out"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "hover_log.json", "w") as f:
        json.dump({
            "duration_sec": args.duration_sec,
            "dt": dt,
            "target_height": args.target_height,
            "hover_rpm": hover_rpm,
            "hover_throttle": controller.hover_throttle,
            "vehicle_params": vehicle_params,
            "height_mean": float(height_mean),
            "height_min": float(height_min),
            "height_max": float(height_max),
            "xy_drift_max": float(xy_drift),
            "roll_max_deg": float(roll_max),
            "pitch_max_deg": float(pitch_max),
            "pass_height": bool(h_pass),
            "pass_xy": bool(xy_pass),
            "pass_attitude": bool(att_pass),
            "pass_overall": bool(overall),
            "pos_log": pos_log,
            "euler_log": euler_log,
        }, f)
    print(f"  Log saved to {out_dir / 'hover_log.json'}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
