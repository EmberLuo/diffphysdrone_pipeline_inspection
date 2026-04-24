from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import argparse
import json
import math
from pathlib import Path
from random import normalvariate
from typing import Any, Callable

from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass(frozen=True)
class TrainingTask:
    name: str
    default_log_root: str | Path
    build_env: Callable[[argparse.Namespace, torch.device], Any]
    build_model: Callable[[argparse.Namespace, torch.device], torch.nn.Module]
    make_observation: Callable[[Any, argparse.Namespace, float], torch.Tensor]


def add_common_args(parser: argparse.ArgumentParser, default_log_root: str | Path) -> None:
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_iters", type=int, default=50000)
    parser.add_argument("--coef_v", type=float, default=1.0, help="smooth l1 of norm(v_set - v_real)")
    parser.add_argument("--coef_speed", type=float, default=0.0, help="legacy")
    parser.add_argument("--coef_v_pred", type=float, default=2.0, help="mse loss for velocity estimation (no odom)")
    parser.add_argument("--coef_collide", type=float, default=2.0, help="softplus loss for collision")
    parser.add_argument("--coef_obj_avoidance", type=float, default=1.5, help="quadratic clearance loss")
    parser.add_argument("--coef_d_acc", type=float, default=0.01, help="control acceleration regularization")
    parser.add_argument("--coef_d_jerk", type=float, default=0.001, help="control jerk regularization")
    parser.add_argument("--coef_d_snap", type=float, default=0.0, help="legacy")
    parser.add_argument("--coef_ground_affinity", type=float, default=0.0, help="legacy")
    parser.add_argument("--coef_bias", type=float, default=0.0, help="legacy")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_decay", type=float, default=0.4)
    parser.add_argument("--speed_mtp", type=float, default=1.0)
    parser.add_argument("--fov_x_half_tan", type=float, default=0.53)
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--cam_angle", type=int, default=10)
    parser.add_argument("--single", default=False, action="store_true")
    parser.add_argument("--gate", default=False, action="store_true")
    parser.add_argument("--ground_voxels", default=False, action="store_true")
    parser.add_argument("--scaffold", default=False, action="store_true")
    parser.add_argument("--random_rotation", default=False, action="store_true")
    parser.add_argument("--yaw_drift", default=False, action="store_true")
    parser.add_argument("--no_odom", default=False, action="store_true")
    parser.add_argument("--log_root", default=str(default_log_root))
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--save_every", type=int, default=10000)


def build_standard_env(env_cls: type, args: argparse.Namespace, device: torch.device, **extra_kwargs: Any) -> Any:
    return env_cls(
        args.batch_size,
        64,
        48,
        args.grad_decay,
        device,
        fov_x_half_tan=args.fov_x_half_tan,
        single=args.single,
        gate=args.gate,
        ground_voxels=args.ground_voxels,
        scaffold=args.scaffold,
        speed_mtp=args.speed_mtp,
        random_rotation=args.random_rotation,
        cam_angle=args.cam_angle,
        **extra_kwargs,
    )


def barrier(x: torch.Tensor, v_to_pt: torch.Tensor) -> torch.Tensor:
    return (v_to_pt * (1 - x).relu().pow(2)).mean()


def is_save_iter(i: int) -> bool:
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


def _smooth_dict(queue: dict[str, list[float]], values: dict[str, torch.Tensor]) -> None:
    for key, value in values.items():
        queue[key].append(float(value))


def _make_run_dir(args: argparse.Namespace, task: TrainingTask) -> Path:
    if not args.log_root:
        args.log_root = str(task.default_log_root)

    experiment_name = args.experiment_name or ("single_agent" if args.single else "multi_agent")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = Path(args.log_root) / experiment_name / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _local_frame(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    fwd = env.R[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    fwd[:, 2] = 0
    up[:, 2] = 1
    fwd = F.normalize(fwd, 2, -1)
    R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)
    local_v = torch.squeeze(env.v[:, None] @ R, 1)
    return R, local_v


def _state_from_env(env: Any, args: argparse.Namespace, target_v_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R, local_v = _local_frame(env)
    target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
    target_v_unit = target_v_raw / target_v_norm
    target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
    state = [
        torch.squeeze(target_v[:, None] @ R, 1),
        env.R[:, 2],
        env.margin[:, None],
    ]
    if not args.no_odom:
        state.insert(0, local_v)
    return torch.cat(state, -1), target_v, R


def _log_figures(
    writer: SummaryWriter,
    step: int,
    sample_idx: int,
    p_history: torch.Tensor,
    v_history: torch.Tensor,
    act_buffer: torch.Tensor,
) -> None:
    fig_p, ax = plt.subplots()
    p_plot = p_history[:, sample_idx].cpu()
    ax.plot(p_plot[:, 0], label="x")
    ax.plot(p_plot[:, 1], label="y")
    ax.plot(p_plot[:, 2], label="z")
    ax.legend()

    fig_v, ax = plt.subplots()
    v_plot = v_history[:, sample_idx].cpu()
    ax.plot(v_plot[:, 0], label="x")
    ax.plot(v_plot[:, 1], label="y")
    ax.plot(v_plot[:, 2], label="z")
    ax.legend()

    fig_a, ax = plt.subplots()
    act_plot = act_buffer[:, sample_idx].cpu()
    ax.plot(act_plot[:, 0], label="x")
    ax.plot(act_plot[:, 1], label="y")
    ax.plot(act_plot[:, 2], label="z")
    ax.legend()

    writer.add_figure("p_history", fig_p, step)
    writer.add_figure("v_history", fig_v, step)
    writer.add_figure("a_reals", fig_a, step)
    plt.close(fig_p)
    plt.close(fig_v)
    plt.close(fig_a)


def run_training(args: argparse.Namespace, task: TrainingTask) -> Path:
    if args.save_every <= 0:
        raise ValueError(f"--save_every must be positive, got {args.save_every}")

    run_dir = _make_run_dir(args, task)
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    print(args)
    print(f"Task: {task.name}")
    print(f"Run dir: {run_dir.resolve()}")

    device = torch.device("cuda")
    env = task.build_env(args, device)
    model = task.build_model(args, device)

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
        if missing_keys:
            print("missing_keys:", missing_keys)
        if unexpected_keys:
            print("unexpected_keys:", unexpected_keys)

    optim = AdamW(model.parameters(), args.lr)
    sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)
    scalar_queue: dict[str, list[float]] = defaultdict(list)

    ctl_dt = 1 / 15
    B = args.batch_size
    sample_idx = min(4, B - 1)
    pbar = tqdm(range(args.num_iters), ncols=80)
    try:
        for i in pbar:
            env.reset()
            model.reset()
            p_history = []
            v_history = []
            target_v_history = []
            vec_to_pt_history = []
            v_preds = []
            h = None

            act_lag = 1
            act_buffer = [env.act] * (act_lag + 1)
            target_v_raw = env.p_target - env.p
            if args.yaw_drift:
                drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
                zeros = torch.zeros_like(drift_av)
                ones = torch.ones_like(drift_av)
                R_drift = torch.stack(
                    [
                        torch.cos(drift_av),
                        -torch.sin(drift_av),
                        zeros,
                        torch.sin(drift_av),
                        torch.cos(drift_av),
                        zeros,
                        zeros,
                        zeros,
                        ones,
                    ],
                    -1,
                ).reshape(B, 3, 3)

            for t in range(args.timesteps):
                ctl_dt = normalvariate(1 / 15, 0.1 / 15)
                obs = task.make_observation(env, args, ctl_dt)
                p_history.append(env.p)
                vec_to_pt_history.append(env.find_vec_to_nearest_pt())

                if args.yaw_drift:
                    target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
                else:
                    target_v_raw = env.p_target - env.p.detach()
                env.run(act_buffer[t], ctl_dt, target_v_raw)

                state, target_v, R = _state_from_env(env, args, target_v_raw)
                act, _, h = model(obs, state, h)

                a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
                v_preds.append(v_pred)
                act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
                act_buffer.append(act)

                v_history.append(env.v)
                target_v_history.append(target_v)

            p_history_t = torch.stack(p_history)
            loss_ground_affinity = p_history_t[..., 2].relu().pow(2).mean()
            act_buffer_t = torch.stack(act_buffer)

            v_history_t = torch.stack(v_history)
            v_history_cum = v_history_t.cumsum(0)
            avg_window = min(30, max(1, v_history_t.shape[0] - 1))
            v_history_avg = (v_history_cum[avg_window:] - v_history_cum[:-avg_window]) / avg_window
            target_v_history_t = torch.stack(target_v_history)
            delta_v = torch.norm(v_history_avg - target_v_history_t[1 : 1 + v_history_avg.shape[0]], 2, -1)
            loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

            v_preds_t = torch.stack(v_preds)
            loss_v_pred = F.mse_loss(v_preds_t, v_history_t.detach())

            target_v_history_norm = torch.norm(target_v_history_t, 2, -1)
            target_v_history_normalized = target_v_history_t / target_v_history_norm[..., None]
            fwd_v = torch.sum(v_history_t * target_v_history_normalized, -1)
            loss_bias = F.mse_loss(v_history_t, fwd_v[..., None] * target_v_history_normalized) * 3

            jerk_history = act_buffer_t.diff(1, 0).mul(15)
            snap_history = F.normalize(act_buffer_t - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
            loss_d_acc = act_buffer_t.pow(2).sum(-1).mean()
            loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
            loss_d_snap = snap_history.pow(2).sum(-1).mean()

            vec_to_pt_history_t = torch.stack(vec_to_pt_history)
            distance = torch.norm(vec_to_pt_history_t, 2, -1) - env.margin
            with torch.no_grad():
                v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
            loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
            loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

            speed_history = v_history_t.norm(2, -1)
            loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

            loss = (
                args.coef_v * loss_v
                + args.coef_obj_avoidance * loss_obj_avoidance
                + args.coef_bias * loss_bias
                + args.coef_d_acc * loss_d_acc
                + args.coef_d_jerk * loss_d_jerk
                + args.coef_d_snap * loss_d_snap
                + args.coef_speed * loss_speed
                + args.coef_v_pred * loss_v_pred
                + args.coef_collide * loss_collide
                + args.coef_ground_affinity
                + loss_ground_affinity
            )

            if torch.isnan(loss):
                print("loss is nan, exiting...")
                raise SystemExit(1)

            pbar.set_description_str(f"loss: {loss:.3f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()

            with torch.no_grad():
                avg_speed = speed_history.mean(0)
                success = torch.all(distance.flatten(0, 1) > 0, 0)
                success_rate = success.sum() / B
                _smooth_dict(
                    scalar_queue,
                    {
                        "loss": loss,
                        "loss_v": loss_v,
                        "loss_v_pred": loss_v_pred,
                        "loss_obj_avoidance": loss_obj_avoidance,
                        "loss_d_acc": loss_d_acc,
                        "loss_d_jerk": loss_d_jerk,
                        "loss_d_snap": loss_d_snap,
                        "loss_bias": loss_bias,
                        "loss_speed": loss_speed,
                        "loss_collide": loss_collide,
                        "loss_ground_affinity": loss_ground_affinity,
                        "success": success_rate,
                        "max_speed": speed_history.max(0).values.mean(),
                        "avg_speed": avg_speed.mean(),
                        "ar": (success * avg_speed).mean(),
                    },
                )

                if is_save_iter(i):
                    _log_figures(writer, i + 1, sample_idx, p_history_t, v_history_t, act_buffer_t)

                if (i + 1) % args.save_every == 0:
                    checkpoint_path = run_dir / f"checkpoint{((i + 1) // args.save_every - 1):04d}.pth"
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path.resolve()}")

                if (i + 1) % 25 == 0:
                    for key, values in scalar_queue.items():
                        writer.add_scalar(key, sum(values) / len(values), i + 1)
                    scalar_queue.clear()
    finally:
        writer.close()

    return run_dir
