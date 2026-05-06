"""Robust environment perturbations and target-hover training losses."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class OriginalLossResult:
    loss_per_trajectory: torch.Tensor
    metrics: dict[str, torch.Tensor]
    distance: torch.Tensor
    speed_history: torch.Tensor


@dataclass
class RobustTargetHoverLoss:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    distance: torch.Tensor
    speed_history: torch.Tensor


def add_robust_target_hover_args(parser: argparse.ArgumentParser) -> None:
    def add_bool(name: str, default: bool = False, help_text: str | None = None) -> None:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
        group.add_argument(f"--no_{name}", dest=name, action="store_false")
        parser.set_defaults(**{name: default})

    add_bool("use_robust_target_hover", False)
    add_bool("use_robust_env", False)

    add_bool("use_wind", False)
    parser.add_argument(
        "--wind_mode",
        default="constant",
        choices=["constant", "gust", "side", "vertical", "mixed", "constant_wind", "gust_wind", "side_wind", "vertical_wind"],
    )
    parser.add_argument("--wind_mean_range", type=float, nargs="+", default=[0.0, 0.0])
    parser.add_argument("--wind_gust_range", type=float, nargs="+", default=[0.0, 0.0])
    parser.add_argument("--wind_vertical_range", type=float, nargs="+", default=[0.0, 0.0])
    parser.add_argument("--wind_side_range", type=float, nargs="+", default=[0.0, 0.0])
    parser.add_argument("--wind_update_interval", type=int, default=8)
    parser.add_argument("--wind_randomize_prob", type=float, default=1.0)

    add_bool("use_localization_noise", False)
    parser.add_argument("--pos_noise_std_range", type=float, nargs="+", default=[0.0, 0.0])
    parser.add_argument("--vel_noise_std_range", type=float, nargs="+", default=[0.0, 0.0])
    parser.add_argument("--sigma_p_range", type=float, nargs="+", default=[0.0, 0.0])

    parser.add_argument("--goal_radius", type=float, default=0.5)

    parser.add_argument("--lambda_original", type=float, default=1.0)

    add_bool("use_dynamic_safety_margin", False)
    parser.add_argument("--lambda_corridor", type=float, default=0.0)
    parser.add_argument("--base_safe_distance", type=float, default=0.35)
    parser.add_argument("--safe_margin_extra", type=float, default=0.0)
    parser.add_argument("--safe_margin_k_pos_uncertainty", type=float, default=1.0)
    parser.add_argument("--safe_margin_k_wind", type=float, default=0.05)
    parser.add_argument("--corridor_softplus_sigma", type=float, default=0.2)

    add_bool("use_hover_loss", False)
    parser.add_argument("--lambda_hover", type=float, default=0.0)
    parser.add_argument("--hover_phase_ratio", type=float, default=0.35)
    parser.add_argument("--lambda_hover_pos", type=float, default=1.0)
    parser.add_argument("--lambda_hover_vel", type=float, default=0.5)
    parser.add_argument("--lambda_hover_acc", type=float, default=0.01)

    add_bool("use_uncertainty_loss", False)
    parser.add_argument("--lambda_uncertainty", type=float, default=0.0)

    add_bool("use_control_margin_loss", False)
    parser.add_argument("--lambda_margin", type=float, default=0.0)
    parser.add_argument("--control_margin_rho", type=float, default=0.8)
    parser.add_argument("--control_margin_eps", type=float, default=0.5)
    parser.add_argument("--control_u_max", type=float, default=15.0)

    add_bool("use_ground_affinity_loss", True)
    parser.add_argument("--lambda_ground_affinity", type=float, default=1.0)

    add_bool("use_cvar_loss", False)
    parser.add_argument("--lambda_cvar", type=float, default=0.0)
    parser.add_argument("--cvar_top_ratio", type=float, default=0.2)

    add_bool("use_dob_hover_observer", False)
    add_bool("use_dob_hover_compensation", False)
    parser.add_argument("--dob_beta", type=float, default=0.08)
    parser.add_argument("--dob_kp", type=float, default=2.0)
    parser.add_argument("--dob_kv", type=float, default=1.2)
    parser.add_argument("--dob_ki", type=float, default=0.0)
    parser.add_argument("--dob_integral_limit", type=float, default=1.0)
    parser.add_argument("--dob_hover_radius", type=float, default=1.5)
    parser.add_argument("--dob_hover_gate_temp", type=float, default=0.3)
    parser.add_argument("--dob_blend", type=float, default=0.3)
    parser.add_argument("--dob_max_comp_acc", type=float, default=4.0)
    parser.add_argument("--dob_max_act_norm", type=float, default=18.0)
    add_bool("dob_detach_observer", True)
    add_bool("dob_log_only", True)


def load_yaml_defaults(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PyYAML is required for --config YAML files.") from exc

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(loaded).__name__}")
    return _flatten_mapping(loaded)


def is_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_robust_target_hover", False))


def is_environment_enabled(args: argparse.Namespace) -> bool:
    return _flag(args, "use_robust_env") or is_enabled(args)


def is_dob_enabled(args: argparse.Namespace) -> bool:
    return _flag(args, "use_dob_hover_observer") or _flag(args, "use_dob_hover_compensation")


def reset(env: Any, args: argparse.Namespace) -> None:
    if not is_environment_enabled(args):
        return

    B = env.batch_size
    device = env.device
    dtype = env.p.dtype

    if _flag(args, "use_localization_noise") or _flag(args, "use_dynamic_safety_margin") or _flag(args, "use_uncertainty_loss"):
        env.localization_sigma = _sample_uniform(args.sigma_p_range, B, 1, device, dtype).squeeze(-1).clamp_min(0.0)
    else:
        env.localization_sigma = torch.zeros(B, device=device, dtype=dtype)

    if _flag(args, "use_localization_noise"):
        env.pos_noise_std = _sample_uniform(args.pos_noise_std_range, B, 1, device, dtype).squeeze(-1).clamp_min(0.0)
        env.vel_noise_std = _sample_uniform(args.vel_noise_std_range, B, 1, device, dtype).squeeze(-1).clamp_min(0.0)
    else:
        env.pos_noise_std = torch.zeros(B, device=device, dtype=dtype)
        env.vel_noise_std = torch.zeros(B, device=device, dtype=dtype)

    if _flag(args, "use_wind"):
        env.v_wind = _sample_wind(args, env)
    else:
        env.v_wind = torch.zeros_like(env.v_wind)


def maybe_update_wind(env: Any, args: argparse.Namespace, step: int) -> None:
    if not is_environment_enabled(args) or not _flag(args, "use_wind"):
        return
    mode = _wind_mode(args)
    interval = max(1, int(getattr(args, "wind_update_interval", 1)))
    if mode in {"gust", "mixed"} and step > 0 and step % interval == 0:
        env.v_wind = _sample_wind(args, env)


def state_from_env(
    env: Any,
    args: argparse.Namespace,
    target_v_raw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_v_raw is None:
        target_v_raw = env.p_target - _observed_position(env, args).detach()
    body_R, local_v = _local_frame(env, _observed_velocity(env, args))
    target_v_norm = torch.norm(target_v_raw, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
    target_v_unit = target_v_raw / target_v_norm
    target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
    state_items = [
        torch.squeeze(target_v[:, None] @ body_R, 1),
        env.R[:, 2],
        env.margin[:, None],
    ]
    if not args.no_odom:
        state_items.insert(0, local_v)
    return torch.cat(state_items, -1), target_v, body_R


def init_dob_state(env: Any, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    B = env.batch_size
    device = env.device
    dtype = env.p.dtype
    zeros_3 = torch.zeros(B, 3, device=device, dtype=dtype)
    zeros_1 = torch.zeros(B, device=device, dtype=dtype)
    return {
        "d_hat": zeros_3.clone(),
        "e_int": zeros_3.clone(),
        "hover_gate": zeros_1.clone(),
        "comp_norm": zeros_1.clone(),
        "d_raw_norm": zeros_1.clone(),
        "d_hat_norm": zeros_1.clone(),
    }


def update_dob_state(
    *,
    args: argparse.Namespace,
    env: Any,
    dob_state: dict[str, torch.Tensor],
    v_before: torch.Tensor,
    v_after: torch.Tensor,
    act_applied: torch.Tensor,
    ctl_dt: float,
) -> dict[str, torch.Tensor]:
    if not is_dob_enabled(args):
        return dob_state

    dt = max(float(ctl_dt), 1e-6)
    a_obs = (v_after - v_before) / dt
    u_applied = act_applied - env.g_std
    d_raw = a_obs - u_applied
    if _flag(args, "dob_detach_observer"):
        d_raw = d_raw.detach()

    beta = min(1.0, max(0.0, float(args.dob_beta)))
    d_hat = (1.0 - beta) * dob_state["d_hat"] + beta * d_raw
    d_hat = clamp_norm(d_hat, float(args.dob_max_comp_acc))
    if _flag(args, "dob_detach_observer"):
        d_hat = d_hat.detach()

    dob_state["d_hat"] = d_hat
    dob_state["d_raw_norm"] = torch.norm(d_raw, p=2, dim=-1).detach()
    dob_state["d_hat_norm"] = torch.norm(d_hat, p=2, dim=-1).detach()
    return dob_state


def apply_dob_hover_compensation(
    *,
    args: argparse.Namespace,
    env: Any,
    act_base: torch.Tensor,
    dob_state: dict[str, torch.Tensor],
    ctl_dt: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not is_dob_enabled(args):
        return act_base, dob_state, {}

    p_goal = env.p_target.detach()
    e_p = env.p.detach() - p_goal
    e_v = env.v.detach()
    dist_to_goal = torch.norm(e_p, p=2, dim=-1)
    gate_temp = max(float(args.dob_hover_gate_temp), 1e-6)
    hover_gate = torch.sigmoid((float(args.dob_hover_radius) - dist_to_goal) / gate_temp)

    compensation_active = _flag(args, "use_dob_hover_compensation") and not _flag(args, "dob_log_only")
    if compensation_active:
        e_int = dob_state["e_int"] + hover_gate[:, None] * e_p * float(ctl_dt)
        e_int = clamp_norm(e_int, float(args.dob_integral_limit))
        d_hat = dob_state["d_hat"]
        u_hold = (
            -float(args.dob_kp) * e_p
            - float(args.dob_kv) * e_v
            - float(args.dob_ki) * e_int
            - d_hat
        )
        u_hold = clamp_norm(u_hold, float(args.dob_max_comp_acc))
        act_hold = env.g_std + u_hold
        blend = min(1.0, max(0.0, float(args.dob_blend))) * hover_gate[:, None]
        act_cmd = (1.0 - blend) * act_base + blend * act_hold
        u_cmd = clamp_norm(act_cmd - env.g_std, float(args.dob_max_act_norm))
        act_cmd = env.g_std + u_cmd
        comp_norm = torch.norm(act_cmd - act_base, p=2, dim=-1)
        enabled_ratio = (hover_gate > 0.5).to(dtype=act_base.dtype)
    else:
        e_int = dob_state["e_int"]
        act_cmd = act_base
        comp_norm = torch.zeros_like(hover_gate)
        enabled_ratio = torch.zeros_like(hover_gate)

    dob_state["e_int"] = e_int
    dob_state["hover_gate"] = hover_gate.detach()
    dob_state["comp_norm"] = comp_norm.detach()

    metrics = {
        "dob/d_hat_norm": dob_state["d_hat_norm"],
        "dob/d_raw_norm": dob_state["d_raw_norm"],
        "dob/hover_gate": hover_gate.detach(),
        "dob/comp_norm": comp_norm.detach(),
        "dob/act_base_norm": torch.norm(act_base - env.g_std, p=2, dim=-1).detach(),
        "dob/act_cmd_norm": torch.norm(act_cmd - env.g_std, p=2, dim=-1).detach(),
        "dob/dist_to_goal": dist_to_goal.detach(),
        "dob/enabled_ratio": enabled_ratio.detach(),
    }
    return act_cmd, dob_state, metrics


def compute_loss(
    args: argparse.Namespace,
    env: Any,
    *,
    p_history: torch.Tensor,
    v_history: torch.Tensor,
    target_v_history: torch.Tensor,
    vec_to_pt_history: torch.Tensor,
    v_preds: torch.Tensor,
    act_buffer: torch.Tensor,
    a_history: torch.Tensor,
    wind_history: torch.Tensor,
) -> RobustTargetHoverLoss:
    original = compute_original_loss(
        args,
        env,
        p_history=p_history,
        v_history=v_history,
        target_v_history=target_v_history,
        vec_to_pt_history=vec_to_pt_history,
        v_preds=v_preds,
        act_buffer=act_buffer,
    )
    original_i = original.loss_per_trajectory
    original_metrics = original.metrics
    distance = original.distance
    speed_history = original.speed_history

    goal_radius = float(args.goal_radius)
    target_goal = env.p_target.detach().to(device=p_history.device, dtype=p_history.dtype)
    target_errors = p_history - target_goal[None]

    hover_start = int(p_history.shape[0] * (1.0 - float(args.hover_phase_ratio)))
    hover_start = min(max(0, hover_start), max(0, p_history.shape[0] - 1))
    hover_p = p_history[hover_start:]
    hover_v = v_history[hover_start:]
    hover_a = a_history[hover_start:]
    hover_pos_error = torch.norm(hover_p - target_goal[None], p=2, dim=-1)
    hover_vel_error = torch.norm(hover_v, p=2, dim=-1)
    hover_acc_error = torch.norm(hover_a, p=2, dim=-1)
    hover_loss_i = (
        float(args.lambda_hover_pos) * hover_pos_error.pow(2)
        + float(args.lambda_hover_vel) * hover_vel_error.pow(2)
        + float(args.lambda_hover_acc) * hover_acc_error.pow(2)
    ).mean(0)
    if not _flag(args, "use_hover_loss"):
        hover_loss_i = torch.zeros_like(original_i)

    raw_obstacle_distance_t = torch.norm(vec_to_pt_history, p=2, dim=-1)
    obstacle_distance = raw_obstacle_distance_t.amin(dim=(0, 1))
    d_obs_t = raw_obstacle_distance_t.amin(dim=1)
    sigma_p = _sigma_for_batch(env, p_history.shape[1], p_history.device, p_history.dtype)
    wind_norm_t = torch.norm(wind_history, p=2, dim=-1)
    d_safe_t = (
        float(args.base_safe_distance)
        + float(args.safe_margin_extra)
        + env.margin[None]
        + float(args.safe_margin_k_pos_uncertainty) * sigma_p[None]
        + float(args.safe_margin_k_wind) * wind_norm_t
    )
    corridor_sigma = max(float(args.corridor_softplus_sigma), 1e-6)
    corridor_violation = d_safe_t - d_obs_t
    corridor_loss_i = F.softplus(corridor_violation / corridor_sigma).pow(2).mean(0)
    if not _flag(args, "use_dynamic_safety_margin"):
        corridor_loss_i = torch.zeros_like(original_i)

    uncertainty_loss_i = sigma_p.pow(2) * v_history.norm(p=2, dim=-1).pow(2).mean(0)
    if not _flag(args, "use_uncertainty_loss"):
        uncertainty_loss_i = torch.zeros_like(original_i)

    policy_act = act_buffer[-v_history.shape[0] :]
    control_norm = torch.norm(policy_act - env.g_std, p=2, dim=-1)
    margin_threshold = float(args.control_margin_rho) * float(args.control_u_max)
    margin_eps = max(float(args.control_margin_eps), 1e-6)
    margin_loss_i = F.softplus((control_norm - margin_threshold) / margin_eps).pow(2).mean(0)
    if not _flag(args, "use_control_margin_loss"):
        margin_loss_i = torch.zeros_like(original_i)

    total_i = (
        float(args.lambda_original) * original_i
        + float(args.lambda_corridor) * corridor_loss_i
        + float(args.lambda_hover) * hover_loss_i
        + float(args.lambda_uncertainty) * uncertainty_loss_i
        + float(args.lambda_margin) * margin_loss_i
    )
    mean_loss = total_i.mean()
    if _flag(args, "use_cvar_loss") and float(args.lambda_cvar) > 0.0:
        top_ratio = min(1.0, max(0.0, float(args.cvar_top_ratio)))
        top_k = max(1, int(math.ceil(total_i.numel() * top_ratio)))
        cvar_loss = torch.topk(total_i, top_k).values.mean()
        final_loss = mean_loss + float(args.lambda_cvar) * cvar_loss
    else:
        cvar_loss = mean_loss * 0.0
        final_loss = mean_loss

    safety_margin_violation_count = (corridor_violation > 0).float().sum()
    safety_success = torch.all(distance.flatten(0, 1) > 0, dim=0)
    goal_error = torch.norm(target_errors, p=2, dim=-1)
    final_goal_error = goal_error[-1]
    min_goal_error = goal_error.amin(dim=0)
    hover_position_error_i = hover_pos_error.mean(0)
    hover_velocity_error_i = hover_vel_error.mean(0)
    goal_success = final_goal_error < goal_radius
    hover_success = (hover_position_error_i < goal_radius) & (hover_velocity_error_i < 0.5)
    target_hover_success = safety_success & goal_success & hover_success
    success = target_hover_success if _flag(args, "use_hover_loss") else safety_success

    metrics = {
        "loss/total": final_loss.detach(),
        "loss/original": original_i.mean().detach(),
        "loss/corridor": corridor_loss_i.mean().detach(),
        "loss/hover": hover_loss_i.mean().detach(),
        "loss/uncertainty": uncertainty_loss_i.mean().detach(),
        "loss/control_margin": margin_loss_i.mean().detach(),
        "loss/cvar": cvar_loss.detach(),
        "safety/min_obstacle_distance": obstacle_distance.mean().detach(),
        "safety/margin_violation_count": safety_margin_violation_count.detach(),
        "hover/position_error": hover_pos_error.mean().detach(),
        "hover/velocity_error": hover_vel_error.mean().detach(),
        "goal/final_error": final_goal_error.mean().detach(),
        "goal/min_error": min_goal_error.mean().detach(),
        "disturbance/wind_norm": wind_norm_t.mean().detach(),
        "localization/sigma": sigma_p.mean().detach(),
        "control/saturation_ratio": (control_norm > margin_threshold).float().mean().detach(),
        "success/safety": safety_success.float().mean().detach(),
        "success/goal": goal_success.float().mean().detach(),
        "success/hover": hover_success.float().mean().detach(),
        "success/main": success.float().mean().detach(),
        "performance/avg_speed": speed_history.mean().detach(),
        "performance/ar": (safety_success * speed_history.mean(0)).mean().detach(),
    }
    metrics.update(original_metrics)
    return RobustTargetHoverLoss(loss=final_loss, metrics=metrics, distance=distance, speed_history=speed_history)


def compute_original_loss(
    args: argparse.Namespace,
    env: Any,
    *,
    p_history: torch.Tensor,
    v_history: torch.Tensor,
    target_v_history: torch.Tensor,
    vec_to_pt_history: torch.Tensor,
    v_preds: torch.Tensor,
    act_buffer: torch.Tensor,
) -> OriginalLossResult:
    loss_ground_affinity_i = p_history[..., 2].relu().pow(2).mean(0)

    if v_history.shape[0] > 1:
        avg_window = min(30, max(1, v_history.shape[0] - 1))
        v_history_cum = v_history.cumsum(0)
        v_history_avg = (v_history_cum[avg_window:] - v_history_cum[:-avg_window]) / avg_window
        target_for_avg = target_v_history[1 : 1 + v_history_avg.shape[0]]
        delta_v = torch.norm(v_history_avg - target_for_avg, p=2, dim=-1)
    else:
        delta_v = torch.norm(v_history - target_v_history, p=2, dim=-1)
    loss_v_i = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v), reduction="none").mean(0)

    loss_v_pred_i = (v_preds - v_history.detach()).pow(2).mean(dim=(0, 2))

    target_v_norm = torch.norm(target_v_history, p=2, dim=-1).clamp_min(1e-6)
    target_v_normalized = target_v_history / target_v_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_normalized, dim=-1)
    loss_bias_i = (v_history - fwd_v[..., None] * target_v_normalized).pow(2).mean(dim=(0, 2)) * 3

    jerk_history = act_buffer.diff(1, 0).mul(15)
    if act_buffer.shape[0] > 2:
        snap_history = F.normalize(act_buffer - env.g_std, dim=-1).diff(1, 0).diff(1, 0).mul(15**2)
        loss_d_snap_i = snap_history.pow(2).sum(-1).mean(0)
    else:
        loss_d_snap_i = torch.zeros_like(loss_v_i)
    loss_d_acc_i = act_buffer.pow(2).sum(-1).mean(0)
    loss_d_jerk_i = jerk_history.pow(2).sum(-1).mean(0)

    distance = torch.norm(vec_to_pt_history, p=2, dim=-1) - env.margin
    if distance.shape[1] > 1:
        with torch.no_grad():
            v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
        loss_obj_avoidance_i = _barrier_per_trajectory(distance[:, 1:], v_to_pt)
        loss_collide_i = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean(dim=(0, 1))
    else:
        loss_obj_avoidance_i = (1.0 - distance).relu().pow(2).mean(dim=(0, 1))
        loss_collide_i = F.softplus(distance.mul(-32)).mean(dim=(0, 1))

    loss_speed_i = F.smooth_l1_loss(fwd_v, target_v_norm, reduction="none").mean(0)

    original_i = (
        float(args.coef_v) * loss_v_i
        + float(args.coef_obj_avoidance) * loss_obj_avoidance_i
        + float(args.coef_bias) * loss_bias_i
        + float(args.coef_d_acc) * loss_d_acc_i
        + float(args.coef_d_jerk) * loss_d_jerk_i
        + float(args.coef_d_snap) * loss_d_snap_i
        + float(args.coef_speed) * loss_speed_i
        + float(args.coef_v_pred) * loss_v_pred_i
        + float(args.coef_collide) * loss_collide_i
    )
    if _flag(args, "use_ground_affinity_loss"):
        # z > 0 is normal flight for these target-hover tasks, not an error state.
        original_i = original_i + float(args.lambda_ground_affinity) * loss_ground_affinity_i

    metrics = {
        "loss/velocity_tracking": loss_v_i.mean().detach(),
        "loss/velocity_prediction": loss_v_pred_i.mean().detach(),
        "loss/object_avoidance": loss_obj_avoidance_i.mean().detach(),
        "loss/control_acc": loss_d_acc_i.mean().detach(),
        "loss/control_jerk": loss_d_jerk_i.mean().detach(),
        "loss/collision": loss_collide_i.mean().detach(),
        "loss/ground_affinity": loss_ground_affinity_i.mean().detach(),
    }
    speed_history = v_history.norm(p=2, dim=-1)
    return OriginalLossResult(
        loss_per_trajectory=original_i,
        metrics=metrics,
        distance=distance,
        speed_history=speed_history,
    )


def _barrier_per_trajectory(x: torch.Tensor, v_to_pt: torch.Tensor) -> torch.Tensor:
    return (v_to_pt * (1 - x).relu().pow(2)).mean(dim=(0, 1))


def clamp_norm(x: torch.Tensor, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.zeros_like(x)
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / norm.clamp_min(eps), max=1.0)
    return x * scale


def _local_frame(env: Any, v_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fwd = env.R[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    fwd[:, 2] = 0
    up[:, 2] = 1
    fwd = F.normalize(fwd, p=2, dim=-1, eps=1e-6)
    body_R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)
    local_v = torch.squeeze(v_obs[:, None] @ body_R, 1)
    return body_R, local_v


def _observed_position(env: Any, args: argparse.Namespace) -> torch.Tensor:
    if not _uses_position_noise(env, args):
        return env.p.detach()
    noise = torch.randn_like(env.p) * env.pos_noise_std[:, None]
    return env.p.detach() + noise


def _observed_velocity(env: Any, args: argparse.Namespace) -> torch.Tensor:
    if not _uses_velocity_noise(env, args):
        return env.v
    noise = torch.randn_like(env.v) * env.vel_noise_std[:, None]
    return env.v + noise


def _uses_position_noise(env: Any, args: argparse.Namespace) -> bool:
    return is_environment_enabled(args) and _flag(args, "use_localization_noise") and getattr(env, "pos_noise_std", None) is not None


def _uses_velocity_noise(env: Any, args: argparse.Namespace) -> bool:
    return is_environment_enabled(args) and _flag(args, "use_localization_noise") and getattr(env, "vel_noise_std", None) is not None


def _sample_wind(args: argparse.Namespace, env: Any) -> torch.Tensor:
    B = env.batch_size
    device = env.device
    dtype = env.v.dtype
    mode = _wind_mode(args)

    wind = _sample_uniform(args.wind_mean_range, B, 3, device, dtype)
    if mode in {"side", "mixed"}:
        wind[:, 1] = wind[:, 1] + _sample_signed_component(args.wind_side_range, B, device, dtype)
    if mode in {"vertical", "mixed"}:
        wind[:, 2] = wind[:, 2] + _sample_signed_component(args.wind_vertical_range, B, device, dtype)
    if mode in {"gust", "mixed"}:
        wind = wind + _sample_uniform(args.wind_gust_range, B, 3, device, dtype)

    prob = min(1.0, max(0.0, float(getattr(args, "wind_randomize_prob", 1.0))))
    if prob < 1.0:
        mask = torch.rand(B, device=device) < prob
        wind = torch.where(mask[:, None], wind, torch.zeros_like(wind))
    return wind


def _sample_signed_component(value: Any, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    flat = _as_float_list(value)
    if len(flat) == 2 and flat[0] >= 0.0:
        lo, hi = flat
        mag = torch.empty(B, device=device, dtype=dtype).uniform_(float(lo), float(hi))
        sign = torch.where(torch.rand(B, device=device) < 0.5, -torch.ones(B, device=device), torch.ones(B, device=device))
        return mag * sign.to(dtype)
    return _sample_uniform(value, B, 1, device, dtype).squeeze(-1)


def _wind_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "wind_mode", "constant")).lower()
    aliases = {
        "constant_wind": "constant",
        "gust_wind": "gust",
        "side_wind": "side",
        "vertical_wind": "vertical",
    }
    return aliases.get(mode, mode)


def _sample_uniform(value: Any, B: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    flat = _as_float_list(value)
    if len(flat) == 0:
        return torch.zeros(B, dim, device=device, dtype=dtype)
    if len(flat) == 1:
        return torch.full((B, dim), float(flat[0]), device=device, dtype=dtype)
    if len(flat) == 2:
        lo, hi = flat
        return torch.empty(B, dim, device=device, dtype=dtype).uniform_(float(lo), float(hi))
    if len(flat) == dim:
        return torch.tensor(flat, device=device, dtype=dtype)[None].expand(B, dim).clone()
    if len(flat) == dim * 2:
        pairs = torch.tensor(flat, device=device, dtype=dtype).reshape(dim, 2)
        lo = pairs[:, 0]
        hi = pairs[:, 1]
        return lo[None] + torch.rand(B, dim, device=device, dtype=dtype) * (hi - lo)[None]
    raise ValueError(f"Cannot interpret range {value!r} for dimension {dim}")


def _sigma_for_batch(env: Any, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = getattr(env, "localization_sigma", None)
    if sigma is None:
        return torch.zeros(B, device=device, dtype=dtype)
    return sigma.to(device=device, dtype=dtype)


def _as_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(v) for v in value.flatten().tolist()]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(part) for part in value.replace(",", " ").split()]
    result: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            result.extend(_as_float_list(item))
        else:
            result.append(float(item))
    return result


def _flatten_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value))
        else:
            flat[key] = value
    return flat


def _flag(args: argparse.Namespace, name: str) -> bool:
    return bool(getattr(args, name, False))
