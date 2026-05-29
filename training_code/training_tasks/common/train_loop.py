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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training_code.training_tasks.common import robust_target_hover


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
    robust_target_hover.add_robust_target_hover_args(parser)


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


def is_save_iter(i: int) -> bool:
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


def _smooth_dict(queue: dict[str, list[float]], values: dict[str, torch.Tensor]) -> None:
    for key, value in values.items():
        queue[key].append(float(value))


def _mean_metric_history(history: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not history:
        return {}
    result = {}
    keys = sorted({key for metrics in history for key in metrics})
    for key in keys:
        values = [metrics[key].float().mean() for metrics in history if key in metrics]
        if values:
            result[key] = torch.stack(values).mean()
    return result


def _make_run_dir(args: argparse.Namespace, task: TrainingTask) -> Path:
    if not args.log_root:
        args.log_root = str(task.default_log_root)

    log_root = _classified_log_root(Path(args.log_root), task)
    experiment_name = args.experiment_name or default_experiment_name(args, task)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = log_root / experiment_name / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=False)
    args.log_sensor = task.name
    args.log_algorithm = _training_algorithm(args)
    args.log_environment = _training_environment(args)
    args.log_task = _training_task(args)
    args.resolved_log_root = str(log_root)
    args.resolved_experiment_name = experiment_name
    args.resolved_run_dir = str(run_dir)
    return run_dir


def _classified_log_root(log_root: Path, task: TrainingTask) -> Path:
    if log_root.name == task.name:
        return log_root
    return log_root / task.name


def default_experiment_name(args: argparse.Namespace, task: TrainingTask) -> str:
    algorithm = _training_algorithm(args)
    environment = _training_environment(args)
    training_task = _training_task(args)
    agent_mode = "single_agent" if args.single else "multi_agent"
    odom_mode = "no_odom" if args.no_odom else "odom"
    return str(Path(algorithm) / environment / training_task / f"{agent_mode}_{odom_mode}")


def _training_algorithm(args: argparse.Namespace) -> str:
    if robust_target_hover.is_enabled(args):
        return "rth"
    return "diffphys"


def _training_environment(args: argparse.Namespace) -> str:
    if bool(getattr(args, "use_wind", False)):
        if bool(getattr(args, "use_wind_curriculum", False)):
            return "strong_wind_curriculum"
        return "strong_wind"
    if _has_non_wind_robust_perturbation(args):
        return "robust_env"
    return "nominal"


def _has_non_wind_robust_perturbation(args: argparse.Namespace) -> bool:
    return any(
        bool(getattr(args, name, False))
        for name in (
            "use_localization_noise",
            "randomize_start_target_z",
        )
    )


def _training_task(args: argparse.Namespace) -> str:
    if robust_target_hover.is_enabled(args):
        return "target_hover"
    if robust_target_hover.is_environment_enabled(args):
        return "random_target"
    return "navigation"


def _target_hover_diagnostics(
    args: argparse.Namespace,
    env: Any,
    p_history: torch.Tensor,
    v_history: torch.Tensor,
    safety_success: torch.Tensor,
) -> dict[str, torch.Tensor]:
    goal_radius = float(args.goal_radius)
    target_goal = env.p_target.detach().to(device=p_history.device, dtype=p_history.dtype)
    goal_error = torch.norm(p_history - target_goal[None], p=2, dim=-1)
    final_goal_error = goal_error[-1]
    min_goal_error = goal_error.amin(dim=0)

    hover_start = int(p_history.shape[0] * (1.0 - float(args.hover_phase_ratio)))
    hover_start = min(max(0, hover_start), max(0, p_history.shape[0] - 1))
    hover_pos_error = torch.norm(p_history[hover_start:] - target_goal[None], p=2, dim=-1)
    hover_vel_error = torch.norm(v_history[hover_start:], p=2, dim=-1)
    hover_position_error_i = hover_pos_error.mean(0)
    hover_velocity_error_i = hover_vel_error.mean(0)

    goal_success = final_goal_error < goal_radius
    hover_success = (hover_position_error_i < goal_radius) & (hover_velocity_error_i < 0.5)
    return {
        "goal/final_error": final_goal_error.mean(),
        "goal/min_error": min_goal_error.mean(),
        "hover/position_error": hover_pos_error.mean(),
        "hover/velocity_error": hover_vel_error.mean(),
        "success/safety": safety_success.float().mean(),
        "success/goal": goal_success.float().mean(),
        "success/hover": hover_success.float().mean(),
    }



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
    rth_enabled = robust_target_hover.is_enabled(args)
    robust_env_enabled = robust_target_hover.is_environment_enabled(args)
    pbar = tqdm(range(args.num_iters), ncols=80)
    try:
        for i in pbar:
            env.reset()
            if robust_env_enabled:
                robust_target_hover.reset(env, args, iteration=i, num_iters=args.num_iters)
            dob_state = robust_target_hover.init_dob_state(env, args)
            model.reset()
            p_history = []
            v_history = []
            a_history = []
            wind_history = []
            dob_metric_history = []
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
                if robust_env_enabled:
                    robust_target_hover.maybe_update_wind(env, args, t, iteration=i, num_iters=args.num_iters)
                obs = task.make_observation(env, args, ctl_dt)
                p_history.append(env.p)
                vec_to_pt_history.append(env.find_vec_to_nearest_pt())

                if args.yaw_drift:
                    target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
                else:
                    target_v_raw = env.p_target - env.p.detach()
                v_before = env.v.detach().clone()
                act_applied = act_buffer[t]
                env.run(act_applied, ctl_dt, target_v_raw)
                dob_state = robust_target_hover.update_dob_state(
                    args=args,
                    env=env,
                    dob_state=dob_state,
                    v_before=v_before,
                    v_after=env.v.detach(),
                    act_applied=act_applied.detach(),
                    ctl_dt=ctl_dt,
                )

                if rth_enabled:
                    a_history.append(env.a)
                    wind_history.append(env.v_wind)
                if robust_env_enabled:
                    state, target_v, R = robust_target_hover.state_from_env(env, args)
                else:
                    state, target_v, R = robust_target_hover.state_from_env(env, args, target_v_raw)
                act, _, h = model(obs, state, h)

                a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
                v_preds.append(v_pred)
                act_base = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
                act_cmd, dob_state, dob_metrics = robust_target_hover.apply_dob_hover_compensation(
                    args=args,
                    env=env,
                    act_base=act_base,
                    dob_state=dob_state,
                    ctl_dt=ctl_dt,
                )
                if dob_metrics:
                    dob_metric_history.append(dob_metrics)
                act_buffer.append(act_cmd)

                v_history.append(env.v)
                target_v_history.append(target_v)

            p_history_t = torch.stack(p_history)
            act_buffer_t = torch.stack(act_buffer)
            v_history_t = torch.stack(v_history)
            target_v_history_t = torch.stack(target_v_history)
            v_preds_t = torch.stack(v_preds)
            vec_to_pt_history_t = torch.stack(vec_to_pt_history)

            if rth_enabled:
                rth_loss = robust_target_hover.compute_loss(
                    args,
                    env,
                    p_history=p_history_t,
                    v_history=v_history_t,
                    target_v_history=target_v_history_t,
                    vec_to_pt_history=vec_to_pt_history_t,
                    v_preds=v_preds_t,
                    act_buffer=act_buffer_t,
                    a_history=torch.stack(a_history),
                    wind_history=torch.stack(wind_history),
                )
                loss = rth_loss.loss
                distance = rth_loss.distance
                speed_history = rth_loss.speed_history
                log_metrics = rth_loss.metrics
            else:
                original_loss = robust_target_hover.compute_original_loss(
                    args,
                    env,
                    p_history=p_history_t,
                    v_history=v_history_t,
                    target_v_history=target_v_history_t,
                    vec_to_pt_history=vec_to_pt_history_t,
                    v_preds=v_preds_t,
                    act_buffer=act_buffer_t,
                )
                loss = original_loss.loss_per_trajectory.mean()
                if float(args.coef_ground_affinity) != 0.0:
                    loss = loss + float(args.coef_ground_affinity)
                distance = original_loss.distance
                speed_history = original_loss.speed_history
                log_metrics = {
                    "loss/total": loss.detach(),
                    **{k: v.mean().detach() for k, v in original_loss.metrics.items()},
                }
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
                if not rth_enabled:
                    log_metrics.update(
                        {
                            "performance/avg_speed": avg_speed.mean(),
                            "performance/ar": (success * avg_speed).mean(),
                        }
                    )
                    if robust_env_enabled:
                        log_metrics.update(
                            _target_hover_diagnostics(args, env, p_history_t, v_history_t, success)
                        )
                    log_metrics["success/main"] = success_rate
                log_metrics.update(_mean_metric_history(dob_metric_history))
                _smooth_dict(scalar_queue, log_metrics)

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
