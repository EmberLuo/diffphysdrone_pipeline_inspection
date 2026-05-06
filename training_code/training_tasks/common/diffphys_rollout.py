from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
from random import normalvariate
from typing import Any

import torch
import torch.nn.functional as F

from training_code.training_tasks.common import robust_target_hover
from training_code.training_tasks.common.ppo_buffer import RolloutStorage
from training_code.training_tasks.common.ppo_model import DepthDiffPPOActorCritic
from training_code.training_tasks.common.train_loop import TrainingTask


@dataclass
class DiffPhysLoss:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class DiffPhysRollout:
    storage: RolloutStorage
    diff_loss: DiffPhysLoss
    last_obs: torch.Tensor
    last_state: torch.Tensor
    last_critic_hx: torch.Tensor | None
    metrics: dict[str, torch.Tensor]



def _make_yaw_drift(batch_size: int, device: torch.device) -> torch.Tensor:
    drift_av = torch.randn(batch_size, device=device) * (5 * math.pi / 180 / 15)
    zeros = torch.zeros_like(drift_av)
    ones = torch.ones_like(drift_av)
    return torch.stack(
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
    ).reshape(batch_size, 3, 3)


def _current_clearance(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    vec_to_pt = env.find_vec_to_nearest_pt()
    clearance = torch.norm(vec_to_pt, p=2, dim=-1) - env.margin
    return clearance.min(dim=0).values, vec_to_pt


def _step_reward(
    args: argparse.Namespace,
    env: Any,
    target_v: torch.Tensor,
    v_pred: torch.Tensor,
    env_action: torch.Tensor,
    prev_env_action: torch.Tensor,
    prev_goal_dist: torch.Tensor,
    episode_done: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    goal_dist = torch.norm(env.p_target - env.p.detach(), p=2, dim=-1)
    progress = (prev_goal_dist - goal_dist).clamp(
        -float(args.progress_clip),
        float(args.progress_clip),
    )

    delta_v = torch.norm(env.v.detach() - target_v.detach(), p=2, dim=-1)
    speed_loss = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v), reduction="none")
    v_pred_loss = (v_pred.detach() - env.v.detach()).pow(2).mean(dim=-1)
    acc_loss = env_action.detach().pow(2).sum(dim=-1)
    jerk_loss = (env_action.detach() - prev_env_action.detach()).pow(2).sum(dim=-1)

    clearance, _ = _current_clearance(env)
    avoidance_loss = (1.0 - clearance).relu().pow(2)
    collision_loss = F.softplus(clearance * -32.0)
    collided = clearance < 0.0
    success = goal_dist < float(args.success_radius)

    active = ~episode_done
    new_success = success & active
    new_collision = collided & active
    done = episode_done | success | collided

    reward = (
        float(args.progress_coef) * progress
        + float(args.coef_alive)
        - float(args.coef_v) * speed_loss
        - float(args.coef_v_pred) * v_pred_loss
        - float(args.coef_obj_avoidance) * avoidance_loss
        - float(args.coef_collide) * collision_loss
        - float(args.coef_d_acc) * acc_loss
        - float(args.coef_d_jerk) * jerk_loss * float(args.jerk_scale)
        + float(args.success_reward) * new_success.float()
        - float(args.collision_penalty) * new_collision.float()
    )
    reward = torch.where(active, reward, torch.zeros_like(reward))

    metrics = {
        "reward": reward.detach().mean(),
        "progress": progress.detach().mean(),
        "speed_loss_step": speed_loss.detach().mean(),
        "v_pred_loss_step": v_pred_loss.detach().mean(),
        "avoidance_loss_step": avoidance_loss.detach().mean(),
        "collision_loss_step": collision_loss.detach().mean(),
        "step_success": new_success.float().mean(),
        "step_collision": new_collision.float().mean(),
        "clearance": clearance.detach().mean(),
        "goal_dist": goal_dist.detach().mean(),
    }
    return reward.detach(), done.detach(), metrics


def compute_diffphys_loss(args: argparse.Namespace, env: Any, history: dict[str, list[torch.Tensor]]) -> DiffPhysLoss:
    p_history = torch.stack(history["p"])
    v_history = torch.stack(history["v"])
    target_v_history = torch.stack(history["target_v"])
    v_preds = torch.stack(history["v_pred"])
    act_buffer = torch.stack(history["act"])
    vec_to_pt_history = torch.stack(history["vec_to_pt"])

    if robust_target_hover.is_enabled(args):
        rth_loss = robust_target_hover.compute_loss(
            args,
            env,
            p_history=p_history,
            v_history=v_history,
            target_v_history=target_v_history,
            vec_to_pt_history=vec_to_pt_history,
            v_preds=v_preds,
            act_buffer=act_buffer,
            a_history=torch.stack(history["a"]),
            wind_history=torch.stack(history["wind"]),
        )
        metrics = _ppo_rth_metrics(rth_loss.metrics, diff_loss_prefix=True)
        return DiffPhysLoss(loss=rth_loss.loss, metrics=metrics)

    original_loss = robust_target_hover.compute_original_loss(
        args,
        env,
        p_history=p_history,
        v_history=v_history,
        target_v_history=target_v_history,
        vec_to_pt_history=vec_to_pt_history,
        v_preds=v_preds,
        act_buffer=act_buffer,
    )
    loss = original_loss.loss_per_trajectory.mean()
    if float(args.coef_ground_affinity) != 0.0:
        loss = loss + float(args.coef_ground_affinity)

    with torch.no_grad():
        distance = original_loss.distance
        speed_history = original_loss.speed_history
        success = torch.all(distance.flatten(0, 1) > 0, dim=0)
        metrics: dict[str, torch.Tensor] = {
            f"diff_loss/{k}": v.mean().detach()
            for k, v in original_loss.metrics.items()
        }
        metrics["diff_loss/loss/total"] = loss.detach()
        metrics["rollout/success/safety"] = success.float().mean()
        metrics["rollout/performance/avg_speed"] = speed_history.mean()
        metrics["rollout/performance/ar"] = (success * speed_history.mean(0)).mean()
    return DiffPhysLoss(loss=loss, metrics=metrics)


def _ppo_rth_metrics(
    rth_metrics: dict[str, torch.Tensor],
    *,
    diff_loss_prefix: bool,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for key, value in rth_metrics.items():
        metrics[f"scenario/{key}"] = value

    metrics["rollout/success/safety"] = rth_metrics["success/safety"]
    metrics["rollout/success/goal"] = rth_metrics["success/goal"]
    metrics["rollout/success/hover"] = rth_metrics["success/hover"]
    metrics["rollout/performance/avg_speed"] = rth_metrics["performance/avg_speed"]
    metrics["rollout/performance/ar"] = rth_metrics["performance/ar"]
    if diff_loss_prefix:
        for key, value in rth_metrics.items():
            if key not in {
                "success/main",
                "performance/avg_speed",
                "performance/ar",
            }:
                metrics[f"diff_loss/{key}"] = value
    return metrics


def collect_diffphys_rollout(
    args: argparse.Namespace,
    task: TrainingTask,
    env: Any,
    model: DepthDiffPPOActorCritic,
    device: torch.device,
    keep_diff_graph: bool,
) -> DiffPhysRollout:
    env.reset()
    rth_enabled = robust_target_hover.is_enabled(args)
    robust_env_enabled = robust_target_hover.is_environment_enabled(args)
    if robust_env_enabled:
        robust_target_hover.reset(env, args)
    model.reset()

    batch_size = int(args.batch_size)
    actor_h: torch.Tensor | None = None
    critic_h: torch.Tensor | None = None
    zero_h = torch.zeros(batch_size, model.hidden_size, device=device)
    episode_done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    act_buffer = [env.act] * (int(args.act_lag) + 1)
    target_v_raw = env.p_target - env.p
    yaw_drift_R = _make_yaw_drift(batch_size, device) if args.yaw_drift else None

    storage_items: dict[str, list[torch.Tensor]] = {
        "obs": [],
        "states": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "dones": [],
        "actor_hxs": [],
        "critic_hxs": [],
    }
    history: dict[str, list[torch.Tensor]] = {
        "p": [],
        "v": [],
        "target_v": [],
        "vec_to_pt": [],
        "v_pred": [],
        "act": list(act_buffer),
    }
    if rth_enabled:
        history["a"] = []
        history["wind"] = []
    metric_sums: dict[str, torch.Tensor] = {}

    with torch.set_grad_enabled(keep_diff_graph):
        for step in range(int(args.timesteps)):
            ctl_dt = normalvariate(float(args.ctl_dt_mean), float(args.ctl_dt_std))
            if robust_env_enabled:
                robust_target_hover.maybe_update_wind(env, args, step)
            obs = task.make_observation(env, args, ctl_dt)
            history["p"].append(env.p)
            history["vec_to_pt"].append(env.find_vec_to_nearest_pt())
            prev_goal_dist = torch.norm(env.p_target - env.p.detach(), p=2, dim=-1)

            if yaw_drift_R is not None:
                target_v_raw = torch.squeeze(target_v_raw[:, None] @ yaw_drift_R, 1)
            else:
                target_v_raw = env.p_target - env.p.detach()

            env.run(act_buffer[len(history["v"])], ctl_dt, target_v_raw)
            if rth_enabled:
                history["a"].append(env.a)
                history["wind"].append(env.v_wind)
            if robust_env_enabled:
                state, target_v, body_R = robust_target_hover.state_from_env(env, args)
            else:
                state, target_v, body_R = robust_target_hover.state_from_env(env, args, target_v_raw)

            actor_h_store = actor_h.detach() if actor_h is not None else zero_h
            critic_h_store = critic_h.detach() if critic_h is not None else zero_h
            action, log_prob, _, actor_h = model.act(obs, state, actor_h)
            with torch.no_grad():
                value, critic_h = model.value(obs.detach(), state.detach(), critic_h)

            a_pred, v_pred, *_ = (body_R @ action.reshape(batch_size, 3, -1)).unbind(-1)
            env_action = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
            prev_env_action = act_buffer[-1]
            act_buffer.append(env_action)
            history["act"].append(env_action)
            history["v"].append(env.v)
            history["target_v"].append(target_v)
            history["v_pred"].append(v_pred)

            reward, done, step_metrics = _step_reward(
                args,
                env,
                target_v,
                v_pred,
                env_action,
                prev_env_action,
                prev_goal_dist,
                episode_done,
            )
            episode_done = done

            storage_items["obs"].append(obs.detach())
            storage_items["states"].append(state.detach())
            storage_items["actions"].append(action.detach())
            storage_items["log_probs"].append(log_prob.detach())
            storage_items["values"].append(value.detach())
            storage_items["rewards"].append(reward)
            storage_items["dones"].append(done)
            storage_items["actor_hxs"].append(actor_h_store.detach())
            storage_items["critic_hxs"].append(critic_h_store.detach())

            for key, value_t in step_metrics.items():
                metric_sums[key] = metric_sums.get(key, torch.zeros_like(value_t)) + value_t

    diff_loss = compute_diffphys_loss(args, env, history)

    with torch.no_grad():
        ctl_dt = float(args.ctl_dt_mean)
        last_obs = task.make_observation(env, args, ctl_dt).detach()
        last_target_v_raw = env.p_target - env.p.detach()
        if robust_env_enabled:
            last_state, _, _ = robust_target_hover.state_from_env(env, args)
        else:
            last_state, _, _ = robust_target_hover.state_from_env(env, args, last_target_v_raw)
        last_state = last_state.detach()

    metrics = {
        f"reward/{key}": value / max(1, int(args.timesteps))
        for key, value in metric_sums.items()
    }
    metrics.update(diff_loss.metrics)
    metrics["rollout/terminal_fraction"] = episode_done.float().mean()

    storage = RolloutStorage(
        obs=torch.stack(storage_items["obs"]),
        states=torch.stack(storage_items["states"]),
        actions=torch.stack(storage_items["actions"]),
        log_probs=torch.stack(storage_items["log_probs"]),
        values=torch.stack(storage_items["values"]),
        rewards=torch.stack(storage_items["rewards"]),
        dones=torch.stack(storage_items["dones"]),
        actor_hxs=torch.stack(storage_items["actor_hxs"]),
        critic_hxs=torch.stack(storage_items["critic_hxs"]),
    )
    return DiffPhysRollout(
        storage=storage,
        diff_loss=diff_loss,
        last_obs=last_obs,
        last_state=last_state,
        last_critic_hx=critic_h.detach() if critic_h is not None else None,
        metrics=metrics,
    )
