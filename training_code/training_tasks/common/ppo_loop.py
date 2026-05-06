from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training_code.training_tasks.common.diffphys_rollout import collect_diffphys_rollout
from training_code.training_tasks.common.ppo_buffer import minibatch_indices, normalize_advantages
from training_code.training_tasks.common.ppo_model import DepthDiffPPOActorCritic
from training_code.training_tasks.common.train_loop import TrainingTask, default_experiment_name


def add_diffppo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diff_aux_coef", type=float, default=0.1)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--init_log_std", type=float, default=-1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ctl_dt_mean", type=float, default=1.0 / 15.0)
    parser.add_argument("--ctl_dt_std", type=float, default=0.1 / 15.0)
    parser.add_argument("--act_lag", type=int, default=1)
    parser.add_argument("--success_reward", type=float, default=20.0)
    parser.add_argument("--success_radius", type=float, default=0.5)
    parser.add_argument("--collision_penalty", type=float, default=10.0)
    parser.add_argument("--progress_coef", type=float, default=2.0)
    parser.add_argument("--progress_clip", type=float, default=0.25)
    parser.add_argument("--coef_alive", type=float, default=0.01)
    parser.add_argument("--jerk_scale", type=float, default=1.0)


def _make_run_dir(args: argparse.Namespace, task: TrainingTask) -> Path:
    if not args.log_root:
        args.log_root = str(task.default_log_root)

    experiment_name = args.experiment_name or default_experiment_name(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = Path(args.log_root) / experiment_name / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _load_resume(
    args: argparse.Namespace,
    model: DepthDiffPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not args.resume:
        return 0

    loaded: Any = torch.load(args.resume, map_location=device)
    if isinstance(loaded, dict) and "actor_state_dict" in loaded:
        model.actor.load_state_dict(loaded["actor_state_dict"], strict=False)
        if "critic_state_dict" in loaded:
            model.critic.load_state_dict(loaded["critic_state_dict"], strict=False)
        if "log_std" in loaded:
            model.log_std.data.copy_(loaded["log_std"].to(device))
        if "optimizer_state_dict" in loaded:
            optimizer.load_state_dict(loaded["optimizer_state_dict"])
        return int(loaded.get("iter", 0))

    missing, unexpected = model.actor.load_state_dict(loaded, strict=False)
    if missing:
        print("missing actor keys:", missing)
    if unexpected:
        print("unexpected actor keys:", unexpected)
    return 0


def _save_checkpoint(
    run_dir: Path,
    checkpoint_idx: int,
    iteration: int,
    args: argparse.Namespace,
    model: DepthDiffPPOActorCritic,
    optimizer: torch.optim.Optimizer,
) -> None:
    actor_path = run_dir / f"checkpoint{checkpoint_idx:04d}.pth"
    state_path = run_dir / f"ppo_state{checkpoint_idx:04d}.pth"
    torch.save(model.actor.state_dict(), actor_path)
    torch.save(
        {
            "actor_state_dict": model.actor.state_dict(),
            "critic_state_dict": model.critic.state_dict(),
            "log_std": model.log_std.detach().cpu(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "iter": int(iteration),
        },
        state_path,
    )
    print(f"Checkpoint saved: {actor_path.resolve()}")
    print(f"PPO state saved: {state_path.resolve()}")


def _diff_aux_grads(
    args: argparse.Namespace,
    model: DepthDiffPPOActorCritic,
    diff_loss: torch.Tensor,
) -> list[torch.Tensor | None] | None:
    if float(args.diff_aux_coef) <= 0.0:
        return None
    if not diff_loss.requires_grad:
        return None

    params = model.actor_parameters
    grads = torch.autograd.grad(
        float(args.diff_aux_coef) * diff_loss,
        params,
        retain_graph=False,
        allow_unused=True,
    )
    return [None if grad is None else grad.detach() for grad in grads]


def _run_ppo_update(
    args: argparse.Namespace,
    model: DepthDiffPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    diff_grads: list[torch.Tensor | None] | None,
) -> dict[str, float]:
    storage = rollout.storage
    num_samples = storage.rewards.numel()
    device = storage.rewards.device

    flat_obs = storage.flatten(storage.obs)
    flat_states = storage.flatten(storage.states)
    flat_actions = storage.flatten(storage.actions)
    flat_old_log_probs = storage.flatten(storage.log_probs)
    flat_old_values = storage.flatten(storage.values)
    flat_returns = storage.flatten(returns)
    flat_advantages = storage.flatten(advantages)
    flat_actor_hxs = storage.flatten(storage.actor_hxs)
    flat_critic_hxs = storage.flatten(storage.critic_hxs)

    stats: dict[str, float] = defaultdict(float)
    update_count = 0
    planned_updates = max(1, int(args.ppo_epochs) * min(int(args.num_minibatches), num_samples))

    for _ in range(int(args.ppo_epochs)):
        for mb_idx in minibatch_indices(num_samples, int(args.num_minibatches), device):
            log_probs, entropy, values = model.evaluate_actions(
                flat_obs[mb_idx],
                flat_states[mb_idx],
                flat_actions[mb_idx],
                flat_actor_hxs[mb_idx],
                flat_critic_hxs[mb_idx],
            )
            old_log_probs = flat_old_log_probs[mb_idx]
            mb_advantages = flat_advantages[mb_idx]
            ratio = torch.exp(log_probs - old_log_probs)

            unclipped = -mb_advantages * ratio
            clipped = -mb_advantages * torch.clamp(
                ratio,
                1.0 - float(args.clip_range),
                1.0 + float(args.clip_range),
            )
            policy_loss = torch.max(unclipped, clipped).mean()

            old_values = flat_old_values[mb_idx]
            mb_returns = flat_returns[mb_idx]
            values_clipped = old_values + (values - old_values).clamp(
                -float(args.clip_range),
                float(args.clip_range),
            )
            value_loss_unclipped = (values - mb_returns).pow(2)
            value_loss_clipped = (values_clipped - mb_returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + float(args.vf_coef) * value_loss - float(args.ent_coef) * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            if diff_grads is not None:
                for param, grad in zip(model.actor_parameters, diff_grads):
                    if grad is None:
                        continue
                    scaled_grad = grad / planned_updates
                    if param.grad is None:
                        param.grad = scaled_grad.clone()
                    else:
                        param.grad.add_(scaled_grad)
            nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - log_probs).mean()
                clip_fraction = ((ratio - 1.0).abs() > float(args.clip_range)).float().mean()
            stats["loss/policy"] += float(policy_loss.detach())
            stats["loss/value"] += float(value_loss.detach())
            stats["loss/entropy"] += float(entropy_loss.detach())
            stats["loss/total"] += float(loss.detach())
            stats["ppo/approx_kl"] += float(approx_kl.detach())
            stats["ppo/clip_fraction"] += float(clip_fraction.detach())
            update_count += 1

    denom = max(1, update_count)
    return {key: value / denom for key, value in stats.items()}


def run_diffppo_training(args: argparse.Namespace, task: TrainingTask) -> Path:
    if args.save_every <= 0:
        raise ValueError(f"--save_every must be positive, got {args.save_every}")
    if args.act_lag < 0:
        raise ValueError(f"--act_lag must be non-negative, got {args.act_lag}")

    run_dir = _make_run_dir(args, task)
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    device = torch.device("cuda")
    env = task.build_env(args, device)
    state_dim = 7 if args.no_odom else 10
    model = DepthDiffPPOActorCritic(state_dim=state_dim, init_log_std=args.init_log_std).to(device)
    optimizer = AdamW(model.parameters(), lr=float(args.lr))
    start_iter = _load_resume(args, model, optimizer, device)

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    print(args)
    print(f"Task: {task.name}_diffppo")
    print(f"Run dir: {run_dir.resolve()}")

    pbar = tqdm(range(start_iter, int(args.num_iters)), ncols=100)
    try:
        for iteration in pbar:
            keep_diff_graph = float(args.diff_aux_coef) > 0.0
            rollout = collect_diffphys_rollout(
                args=args,
                task=task,
                env=env,
                model=model,
                device=device,
                keep_diff_graph=keep_diff_graph,
            )
            if torch.isnan(rollout.diff_loss.loss):
                print("diff loss is nan, exiting...")
                raise SystemExit(1)

            diff_grads = _diff_aux_grads(args, model, rollout.diff_loss.loss)
            with torch.no_grad():
                last_value, _ = model.value(
                    rollout.last_obs,
                    rollout.last_state,
                    rollout.last_critic_hx,
                )
                returns, advantages = rollout.storage.compute_returns_and_advantages(
                    last_value.detach(),
                    gamma=float(args.gamma),
                    gae_lambda=float(args.gae_lambda),
                )
                advantages = normalize_advantages(advantages)

            ppo_stats = _run_ppo_update(
                args=args,
                model=model,
                optimizer=optimizer,
                rollout=rollout,
                returns=returns.detach(),
                advantages=advantages.detach(),
                diff_grads=diff_grads,
            )

            metrics = {key: float(value.detach()) for key, value in rollout.metrics.items()}
            metrics.update(ppo_stats)
            metrics["ppo/action_std"] = float(model.log_std.detach().exp().mean())
            metrics["ppo/diff_aux_coef"] = float(args.diff_aux_coef)
            for key, value in metrics.items():
                writer.add_scalar(key, value, iteration + 1)

            pbar.set_description_str(
                f"ppo {ppo_stats.get('loss/total', 0.0):.3f} "
                f"diff {metrics.get('diff_loss/loss/total', 0.0):.3f}"
            )

            if (iteration + 1) % int(args.save_every) == 0:
                checkpoint_idx = (iteration + 1) // int(args.save_every) - 1
                _save_checkpoint(run_dir, checkpoint_idx, iteration + 1, args, model, optimizer)
    finally:
        writer.close()

    return run_dir
