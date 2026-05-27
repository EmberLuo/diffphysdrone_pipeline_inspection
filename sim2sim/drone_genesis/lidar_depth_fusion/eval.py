import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import genesis as gs
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from genesis.utils.geom import quat_to_R

THIS_DIR = Path(__file__).resolve().parent
DRONE_GENESIS_DIR = THIS_DIR.parents[0]
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DRONE_GENESIS_DIR) not in sys.path:
    sys.path.insert(0, str(DRONE_GENESIS_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sim2sim.drone_genesis.lidar_depth_fusion.env import NavEnv
from training_code.training_tasks.lidar_depth_fusion.model import LidarDepthFusionModel as Model
from sim2sim.drone_genesis.utils.controller import PX4StyleRPMController


class VideoRecorder:
    def __init__(self, output: Path, width: int, height: int, fps: int = 15, pix_fmt: str = "y8") -> None:
        self.output = str(output)
        cmd = [
            "/usr/bin/ffmpeg",
            "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", pix_fmt,
            "-r", f"{fps}", "-i", "-",
            "-an", "-loglevel", "error", "-pix_fmt", "yuv420p",
            self.output,
        ]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def add_image(self, image: np.ndarray):
        if self.p.stdin is not None:
            self.p.stdin.write(image.tobytes())

    def close(self):
        if self.p.stdin is not None:
            self.p.stdin.close()
        self.p.wait()


def make_body_frame(rotmat: torch.Tensor) -> torch.Tensor:
    fwd = rotmat[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    up[:, 2] = 1.0
    fwd[:, 2] = 0.0
    fwd = F.normalize(fwd, p=2, dim=-1)
    left = torch.cross(up, fwd, dim=-1)
    return torch.stack([fwd, left, up], dim=-1)


def _resolve_resume_path(resume_arg: str) -> Path:
    p = Path(resume_arg)
    if p.is_file():
        return p.resolve()

    cand = [
        REPO_ROOT / resume_arg,
    ]
    for fusion_log_root in [
        REPO_ROOT / "training_code" / "logs" / "lidar_depth_fusion" / "single_agent_no_odom",
        REPO_ROOT / "training_code" / "logs" / "lidar_depth_fusion" / "single_agent",
    ]:
        if fusion_log_root.is_dir():
            for run_dir in sorted((p for p in fusion_log_root.iterdir() if p.is_dir()), reverse=True):
                cand.append(run_dir / resume_arg)

    for c in cand:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(f"Could not find checkpoint: {resume_arg}")


def _build_layout(cfg: dict, env_name: str, device: torch.device):
    layout = cfg["task"]["layouts"][env_name]["agents"]
    names = [item["name"] for item in layout]
    starts = torch.tensor([item["start"] for item in layout], device=device, dtype=torch.float32)
    goals = torch.tensor([item["goal"] for item in layout], device=device, dtype=torch.float32)
    yaw = torch.atan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
    return names, starts, goals, yaw


def _sample_start_noise(starts: torch.Tensor, cfg: dict):
    ncfg = cfg["task"].get("start_noise", {})
    nxy = float(ncfg.get("xy", 0.1))
    nz = float(ncfg.get("z", 0.25))
    noisy = starts.clone()
    noisy[:, 0] += (torch.rand_like(noisy[:, 0]) * 2.0 - 1.0) * nxy
    noisy[:, 1] += (torch.rand_like(noisy[:, 1]) * 2.0 - 1.0) * nxy
    noisy[:, 2] += (torch.rand_like(noisy[:, 2]) * 2.0 - 1.0) * nz
    return noisy


def parse_args():
    parser = argparse.ArgumentParser(description="Genesis lidar_depth_fusion nav eval")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "config" / "nav_eval.yaml"))
    parser.add_argument("--resume", type=str, default="checkpoint0004.pth")
    parser.add_argument("--env", type=str, default="single_nav")
    parser.add_argument("--target_speed", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--smoothness", type=float, default=0.5)
    parser.add_argument("--clockspeed", type=float, default=0.25)
    parser.add_argument("--no_odom", dest="no_odom", action="store_true")
    parser.add_argument("--odom", dest="no_odom", action="store_false")
    # None means "follow config policy.no_odom", otherwise CLI takes precedence.
    parser.set_defaults(no_odom=None)
    parser.add_argument("--ctl_error_std", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--duration_sec", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--trace_policy", action="store_true", default=False)
    parser.add_argument("--trace_stride", type=int, default=1)
    parser.add_argument("--arrived_hold_speed", type=float, default=0.5)
    parser.add_argument("--arrived_hold_kv", type=float, default=1.0)
    parser.add_argument("--gru_warmup_steps", type=int, default=10)
    parser.add_argument("--reach_threshold", type=float, default=1.5)
    parser.add_argument("--output_root", type=str, default=str(THIS_DIR))
    parser.add_argument("--draw_lidar_points", action="store_true", default=False)
    parser.add_argument("--lidar_draw_stride", type=int, default=1)
    parser.add_argument("--lidar_draw_radius", type=float, default=0.015)
    parser.add_argument("--lidar_draw_max_points", type=int, default=720)

    parser.add_argument("--show_viewer", dest="show_viewer", action="store_true")
    parser.add_argument("--no_show_viewer", dest="show_viewer", action="store_false")
    parser.set_defaults(show_viewer=True)
    return parser.parse_args()


def _resolve_policy_no_odom(args, cfg: dict) -> bool:
    cfg_no_odom = bool(cfg.get("policy", {}).get("no_odom", False))
    if args.no_odom is None:
        return cfg_no_odom
    return bool(args.no_odom)


@torch.no_grad()
def run_episode(
    env: NavEnv,
    controller: PX4StyleRPMController,
    model,
    cfg: dict,
    args,
    episode_idx: int,
    resume_path: Path,
):
    episode_seed = args.seed + episode_idx
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.output_root) / f"exps_{args.target_speed}" / f"{timestamp}_ep{episode_idx:02d}"
    log_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(__file__, log_dir / "eval.py")
    trace_jsonl_file = None
    trace_jsonl_fp = None
    if args.trace_policy:
        trace_jsonl_file = log_dir / "policy_trace.jsonl"
        trace_jsonl_fp = open(trace_jsonl_file, "w", encoding="utf-8")

    names, starts, goals, yaw = _build_layout(cfg, args.env, env.device)
    starts = _sample_start_noise(starts, cfg)
    env.reset_episode(starts, yaw, goals)

    policy_no_odom = _resolve_policy_no_odom(args, cfg)
    margin = float(args.margin)
    margin_tensor = torch.full((env.num_agents, 1), margin, device=env.device, dtype=torch.float32)

    lidar_h = int(cfg["lidar_sensor"].get("horizontal_line_num", 120))
    lidar_w = int(cfg["lidar_sensor"].get("vertical_line_num", 6))
    depth_cfg = cfg["depth_camera"]
    depth_h = int(depth_cfg.get("height", 48))
    depth_w = int(depth_cfg.get("width", 64))
    depth_pool = int(depth_cfg.get("pool", 4))
    sensor_fps = int(cfg.get("policy_eval", {}).get("control_hz", 15))
    if depth_h % depth_pool != 0 or depth_w % depth_pool != 0:
        raise ValueError(f"depth size ({depth_h}, {depth_w}) must be divisible by pool={depth_pool}")
    depth_obs_h = depth_h // depth_pool
    depth_obs_w = depth_w // depth_pool
    lidar_recorder = VideoRecorder(log_dir / "lidar.mp4", width=lidar_h, height=lidar_w * env.num_agents, fps=sensor_fps)
    depth_recorder = VideoRecorder(log_dir / "depth.mp4", width=depth_w, height=depth_h * env.num_agents, fps=sensor_fps)

    if args.record:
        env.start_recording()

    h = None
    state_dim = 7 if policy_no_odom else 10
    warmup_steps = max(0, int(args.gru_warmup_steps))
    if warmup_steps > 0:
        for _ in range(warmup_steps):
            _, _, h = model(
                (
                    torch.zeros(env.num_agents, 1, depth_obs_h, depth_obs_w, device=env.device, dtype=torch.float32),
                    torch.zeros(env.num_agents, 1, lidar_h, lidar_w, device=env.device, dtype=torch.float32),
                ),
                torch.zeros(env.num_agents, state_dim, device=env.device, dtype=torch.float32),
                h,
            )

    p_target = goals.clone()
    arrived_flag = torch.zeros((env.num_agents,), device=env.device, dtype=torch.bool)
    crashed_flag = torch.zeros((env.num_agents,), device=env.device, dtype=torch.bool)
    traveled_distance = torch.zeros((env.num_agents,), device=env.device, dtype=torch.float32)
    traveled_time = torch.zeros((env.num_agents,), device=env.device, dtype=torch.float32)
    has_collided = [set() for _ in range(env.num_agents)]

    traj_history = {name: [] for name in names}
    policy_trace = []

    last_pos = env.base_pos.clone()
    forward_vec = quat_to_R(env.base_quat)[:, :, 0].clone()

    ctl_error_std = (
        float(args.ctl_error_std)
        if args.ctl_error_std is not None
        else float(cfg.get("policy_eval", {}).get("ctl_error_std", 0.17))
    )
    ctl_error = torch.randn((env.num_agents, 3), device=env.device) * ctl_error_std

    control_hz = float(cfg.get("policy_eval", {}).get("control_hz", 15.0))
    control_dt = 1.0 / control_hz

    if args.duration_sec is not None:
        end_time = float(args.duration_sec)
    else:
        end_time = float(cfg["task"].get("max_duration_sec", 30.0))

    max_steps = int(cfg["sim"].get("max_steps", 3000))
    if args.num_steps is not None:
        max_steps = int(args.num_steps)

    t_begin_real = time.perf_counter()

    hover = torch.full((env.num_agents, 4), controller.hover_rpm, device=env.device, dtype=torch.float32)
    rpm_cmd = hover.clone()
    roll_cmd = torch.zeros((env.num_agents,), device=env.device, dtype=torch.float32)
    pitch_cmd = torch.zeros((env.num_agents,), device=env.device, dtype=torch.float32)
    yaw_cmd = quat_to_R(env.base_quat)[:, :, 0]
    yaw_cmd = torch.atan2(yaw_cmd[:, 1], yaw_cmd[:, 0])
    throttle_cmd = torch.full((env.num_agents,), controller.hover_throttle, device=env.device, dtype=torch.float32)
    control_elapsed = control_dt
    policy_updates = 0
    first_policy_sim_time = None
    last_policy_sim_time = None
    sim_time_final = 0.0
    all_finished_announced = False

    for step_idx in range(max_steps):
        sim_time = step_idx * env.dt
        sim_time_final = sim_time
        if sim_time >= end_time:
            break

        for i, name in enumerate(names):
            p = env.base_pos[i]
            q = env.base_quat[i]
            traj_history[name].append(
                [float(p[0].item()), float(p[1].item()), float(p[2].item()),
                 float(q[0].item()), float(q[1].item()), float(q[2].item()), float(q[3].item())]
            )

        delta = torch.norm(env.base_pos - last_pos, dim=-1)
        active = ~(arrived_flag | crashed_flag)
        traveled_distance[active] += delta[active]
        traveled_time[active] = sim_time
        last_pos = env.base_pos.clone()

        if control_elapsed >= control_dt - 1e-9:
            lidar_obs = env.get_lidar()  # (num_agents, 1, 120, 6), matching training lidar range-difference input
            depth = env.get_depth().to(device=env.device, dtype=torch.float32)
            depth_obs = 3.0 / depth.clamp(0.3, 24.0) - 0.6
            depth_obs = F.max_pool2d(depth_obs[:, None], depth_pool, depth_pool)
            if (
                args.draw_lidar_points
                and args.show_viewer
                and (policy_updates % max(1, int(args.lidar_draw_stride)) == 0)
            ):
                env.scene.clear_debug_objects()
                lidar_points = env.get_lidar_debug_points(max_points=int(args.lidar_draw_max_points))
                if lidar_points.numel() > 0:
                    env.scene.draw_debug_spheres(
                        poss=lidar_points,
                        radius=max(0.0, float(args.lidar_draw_radius)),
                        color=(0.1, 0.9, 0.3, 0.7),
                    )
            scan_viz = np.uint8(np.clip(lidar_obs.detach().cpu().numpy() / 4.0 * 255.0, 0, 255))
            lidar_recorder.add_image(scan_viz.reshape(-1, lidar_h))
            depth_viz = np.uint8(np.clip(depth.detach().cpu().numpy() / 24.0 * 255.0, 0, 255))
            depth_recorder.add_image(depth_viz.reshape(-1, depth_w))

            rotmat = quat_to_R(env.base_quat)
            env_R = rotmat.clone()
            R = make_body_frame(rotmat)

            target_v = p_target - env.base_pos
            target_v_norm = torch.norm(target_v, 2, -1, keepdim=True).clamp_min(1e-6)
            target_v = target_v / target_v_norm * target_v_norm.clamp_max(args.target_speed)

            local_v = torch.squeeze(env.base_lin_vel[:, None] @ R, 1)
            target_v_local = torch.squeeze(target_v[:, None] @ R, 1)
            up_vec = env_R[:, 2]

            state_items = [target_v_local, up_vec, margin_tensor]
            if not policy_no_odom:
                state_items.insert(0, local_v)
            state = torch.cat(state_items, -1)

            action, _, h = model((depth_obs, lidar_obs), state, h)
            v_setpoint, v_est = (R @ action.reshape(env.num_agents, 3, -1)).unbind(-1)

            a_cmd = v_setpoint - v_est + ctl_error
            a_setpoint = a_cmd.clone()
            a_setpoint[:, 2] += controller.g

            throttle_acc = torch.norm(a_setpoint, dim=-1).clamp_min(1e-4)
            up_cmd = a_setpoint / throttle_acc[:, None]
            throttle_acc = throttle_acc + local_v[:, 2] * local_v[:, 2].abs() * 0.01

            forward_vec = env_R[:, :, 0] * args.smoothness + p_target - env.base_pos
            den = -up_cmd[:, 2]
            den = torch.where(den.abs() < 1e-3, torch.full_like(den, -1e-3), den)
            forward_vec[:, 2] = (forward_vec[:, 0] * up_cmd[:, 0] + forward_vec[:, 1] * up_cmd[:, 1]) / den
            forward_vec = F.normalize(forward_vec, p=2, dim=-1)
            left_vec = torch.cross(up_cmd, forward_vec, dim=-1)

            roll_des = torch.atan2(left_vec[:, 2], up_cmd[:, 2])
            pitch_des = torch.asin(torch.clamp(-forward_vec[:, 2], -0.999999, 0.999999))
            yaw_des = torch.atan2(forward_vec[:, 1], forward_vec[:, 0])
            throttle_des = throttle_acc / 9.8 * controller.hover_throttle

            roll_cmd = roll_des
            pitch_cmd = pitch_des
            yaw_cmd = yaw_des
            throttle_cmd = throttle_des

            if args.trace_policy and (policy_updates % max(1, args.trace_stride) == 0):
                depth_cpu = depth.detach().cpu()
                depth_obs_cpu = depth_obs.detach().cpu()
                lidar_cpu = lidar_obs.detach().cpu()
                state_cpu = state.detach().cpu()
                action_cpu = action.detach().cpu()
                v_set_cpu = v_setpoint.detach().cpu()
                v_est_cpu = v_est.detach().cpu()
                a_set_cpu = a_setpoint.detach().cpu()
                rpm_cpu = rpm_cmd.detach().cpu()
                local_v_cpu = local_v.detach().cpu()
                target_v_local_cpu = target_v_local.detach().cpu()
                up_vec_cpu = up_vec.detach().cpu()
                target_v_cpu = target_v.detach().cpu()
                base_lin_vel_cpu = env.base_lin_vel.detach().cpu()
                base_pos_cpu = env.base_pos.detach().cpu()
                p_target_cpu = p_target.detach().cpu()
                roll_des_cpu = roll_des.detach().cpu()
                pitch_des_cpu = pitch_des.detach().cpu()
                yaw_des_cpu = yaw_des.detach().cpu()
                throttle_des_cpu = throttle_des.detach().cpu()
                throttle_acc_cpu = throttle_acc.detach().cpu()
                wall_time = float(time.perf_counter() - t_begin_real)
                for i, name in enumerate(names):
                    depth_flat = depth_cpu[i].flatten()
                    rec = {
                        "policy_step": int(policy_updates),
                        "sim_time": float(sim_time),
                        "wall_time": wall_time,
                        "drone": name,
                        "depth_mean": float(depth_cpu[i].mean().item()),
                        "depth_p50": float(torch.quantile(depth_flat, 0.5).item()),
                        "depth_min": float(depth_flat.min().item()),
                        "depth_p10": float(torch.quantile(depth_flat, 0.1).item()),
                        "depth_p90": float(torch.quantile(depth_flat, 0.9).item()),
                        "depth_max": float(depth_flat.max().item()),
                        "depth_obs_mean": float(depth_obs_cpu[i].mean().item()),
                        "lidar_mean": float(lidar_cpu[i].mean().item()),
                        "lidar_max": float(lidar_cpu[i].max().item()),
                        "lidar_min": float(lidar_cpu[i].min().item()),
                        "state": state_cpu[i].tolist(),
                        "local_v": local_v_cpu[i].tolist(),
                        "target_v_local": target_v_local_cpu[i].tolist(),
                        "up_vec": up_vec_cpu[i].tolist(),
                        "target_v_world": target_v_cpu[i].tolist(),
                        "base_lin_vel_world": base_lin_vel_cpu[i].tolist(),
                        "base_pos_world": base_pos_cpu[i].tolist(),
                        "target_pos_world": p_target_cpu[i].tolist(),
                        "roll_des": float(roll_des_cpu[i].item()),
                        "pitch_des": float(pitch_des_cpu[i].item()),
                        "yaw_des": float(yaw_des_cpu[i].item()),
                        "throttle_des": float(throttle_des_cpu[i].item()),
                        "throttle_acc": float(throttle_acc_cpu[i].item()),
                        "action": action_cpu[i].tolist(),
                        "v_setpoint": v_set_cpu[i].tolist(),
                        "v_est": v_est_cpu[i].tolist(),
                        "a_setpoint": a_set_cpu[i].tolist(),
                        "rpm": rpm_cpu[i].tolist(),
                    }
                    policy_trace.append(rec)
                    if trace_jsonl_fp is not None:
                        trace_jsonl_fp.write(json.dumps(rec) + "\n")
                if trace_jsonl_fp is not None:
                    trace_jsonl_fp.flush()

            control_elapsed -= control_dt
            if control_elapsed < 0:
                control_elapsed = 0.0
            policy_updates += 1
            if first_policy_sim_time is None:
                first_policy_sim_time = sim_time
            last_policy_sim_time = sim_time

        arrived_active = arrived_flag & (~crashed_flag)
        if torch.any(arrived_active):
            rotmat_hold = quat_to_R(env.base_quat)
            env_R_hold = rotmat_hold.clone()
            p_err = p_target - env.base_pos
            p_norm = torch.norm(p_err, dim=-1, keepdim=True).clamp_min(1e-6)
            v_des = p_err / p_norm * p_norm.clamp_max(args.arrived_hold_speed)
            a_hold = (v_des - env.base_lin_vel) * args.arrived_hold_kv
            a_set_hold = a_hold.clone()
            a_set_hold[:, 2] += controller.g
            thr_hold = torch.norm(a_set_hold, dim=-1).clamp_min(1e-4)
            up_hold = a_set_hold / thr_hold[:, None]

            fwd_hold = env_R_hold[:, :, 0] * args.smoothness + p_err
            den = -up_hold[:, 2]
            den = torch.where(den.abs() < 1e-3, torch.full_like(den, -1e-3), den)
            fwd_hold[:, 2] = (fwd_hold[:, 0] * up_hold[:, 0] + fwd_hold[:, 1] * up_hold[:, 1]) / den
            fwd_hold = F.normalize(fwd_hold, p=2, dim=-1)
            left_hold = torch.cross(up_hold, fwd_hold, dim=-1)

            roll_hold = torch.atan2(left_hold[:, 2], up_hold[:, 2])
            pitch_hold = torch.asin(torch.clamp(-fwd_hold[:, 2], -0.999999, 0.999999))
            yaw_hold = torch.atan2(fwd_hold[:, 1], fwd_hold[:, 0])
            throttle_hold = thr_hold / 9.8 * controller.hover_throttle

            roll_cmd = torch.where(arrived_active, roll_hold, roll_cmd)
            pitch_cmd = torch.where(arrived_active, pitch_hold, pitch_cmd)
            yaw_cmd = torch.where(arrived_active, yaw_hold, yaw_cmd)
            throttle_cmd = torch.where(arrived_active, throttle_hold, throttle_cmd)

        rpm_cmd = controller.compute_rpm_from_rpy_throttle(
            roll_des=roll_cmd, pitch_des=pitch_cmd, yaw_des=yaw_cmd, throttle_des=throttle_cmd,
            base_quat=env.base_quat, base_ang_vel=env.base_ang_vel, dt=env.dt,
        )
        inactive = crashed_flag
        rpm_cmd[inactive] = hover[inactive]

        info = env.step(rpm_cmd, record_frame=args.record)
        control_elapsed += env.dt

        dist_to_target = torch.norm(p_target - env.base_pos, dim=-1)
        newly_reached = (~arrived_flag) & (dist_to_target < env.reach_threshold)
        for i in torch.nonzero(newly_reached, as_tuple=False).flatten().tolist():
            arrived_flag[i] = True
            traveled_time[i] = sim_time
            print(f"{names[i]} arrived in {sim_time}s!")

        newly_crash = (~crashed_flag) & info["crash"]
        for i in torch.nonzero(newly_crash, as_tuple=False).flatten().tolist():
            crashed_flag[i] = True
            reasons = []
            if bool(info.get("crash_height", torch.zeros_like(info["crash"]))[i].item()):
                reasons.append("height")
            if bool(info.get("crash_attitude", torch.zeros_like(info["crash"]))[i].item()):
                reasons.append("attitude")
            if bool(info.get("crash_bounds", torch.zeros_like(info["crash"]))[i].item()):
                reasons.append("bounds")
            rstr = ",".join(reasons) if reasons else "unknown"
            print(f"{names[i]} crashed at {sim_time}s ({rstr})")

        for i, j in info["collision_pairs"]:
            n_i, n_j = names[i], names[j]
            if n_j not in has_collided[i]:
                print(f"{n_i} collide with {n_j}!")
            has_collided[i].add(n_j)
            has_collided[j].add(n_i)

        if torch.all(arrived_flag):
            end_time = min(end_time, sim_time + 0.5)
        elif torch.all(arrived_flag | crashed_flag):
            if not all_finished_announced:
                print(f"all agents finished (arrived or crashed) at {sim_time}s, ending episode early.")
                all_finished_announced = True
            end_time = min(end_time, sim_time + 0.5)

        if args.clockspeed > 0 and sim_time > 0:
            desired_real_elapsed = sim_time / args.clockspeed
            real_elapsed = time.perf_counter() - t_begin_real
            sleep_s = desired_real_elapsed - real_elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    if args.record:
        env.stop_recording(str(log_dir / f"{timestamp}.mp4"), fps=int(cfg["sim"].get("max_visualize_fps", 60)))
    lidar_recorder.close()
    depth_recorder.close()
    if trace_jsonl_fp is not None:
        trace_jsonl_fp.close()

    with open(log_dir / "traj_history.json", "w", encoding="utf-8") as f:
        json.dump(traj_history, f)

    trace_file = None
    if args.trace_policy:
        trace_file = log_dir / "policy_trace.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "framework": "genesis",
                    "sensor": "lidar_depth_fusion",
                    "target_speed": args.target_speed,
                    "seed": int(episode_seed),
                    "control_hz": control_hz,
                    "no_odom": bool(policy_no_odom),
                    "records": policy_trace,
                },
                f,
            )

    with open(log_dir / "log", "w", encoding="utf-8") as f:
        f.write(f"{args}\n")
        for i, name in enumerate(names):
            collisions = "_".join(sorted(has_collided[i]))
            f.write(
                f"ours,{args.env},{args.target_speed},{name},{traveled_distance[i].item():.2f},{traveled_time[i].item():.2f},0,{bool(arrived_flag[i].item())},{collisions}\n"
            )

    if policy_updates >= 2 and first_policy_sim_time is not None and last_policy_sim_time is not None:
        hz_denom = max(1e-6, last_policy_sim_time - first_policy_sim_time)
        effective_policy_hz = float((policy_updates - 1) / hz_denom)
    else:
        effective_policy_hz = 0.0

    result = {
        "episode_idx": episode_idx,
        "seed": episode_seed,
        "log_dir": str(log_dir.resolve()),
        "resume": str(resume_path),
        "target_speed": args.target_speed,
        "policy_updates": policy_updates,
        "sim_time_sec": float(sim_time_final),
        "effective_policy_hz": effective_policy_hz,
        "completed": bool(torch.all(arrived_flag).item()),
        "arrived_count": int(arrived_flag.sum().item()),
        "crashed_count": int(crashed_flag.sum().item()),
    }
    if trace_file is not None:
        result["trace_file"] = str(trace_file.resolve())
    if trace_jsonl_file is not None:
        result["trace_jsonl_file"] = str(trace_jsonl_file.resolve())
    print(json.dumps(result, indent=2))
    return result


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gs.init(logging_level="error")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for lidar_depth_fusion Genesis evaluation.")

    resume_path = _resolve_resume_path(args.resume)

    policy_no_odom = _resolve_policy_no_odom(args, cfg)
    model = Model(7 if policy_no_odom else 10, 6).to(device)
    state_dict = torch.load(str(resume_path), map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
    model.eval()

    env = NavEnv(cfg=cfg, env_name=args.env, show_viewer=args.show_viewer, device=str(device))
    env.reach_threshold = float(args.reach_threshold)
    controller = PX4StyleRPMController(
        cfg=cfg["controller"],
        num_envs=env.num_agents,
        device=device,
        vehicle_params=env.vehicle_params,
    )

    all_results = []
    for episode_idx in range(args.num_episodes):
        result = run_episode(
            env=env, controller=controller, model=model, cfg=cfg, args=args,
            episode_idx=episode_idx, resume_path=resume_path,
        )
        all_results.append(result)

    print(json.dumps({"num_episodes": args.num_episodes, "results": all_results}, indent=2))


if __name__ == "__main__":
    main()
