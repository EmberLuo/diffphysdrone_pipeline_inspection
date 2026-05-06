from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutStorage:
    obs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    actor_hxs: torch.Tensor
    critic_hxs: torch.Tensor

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(self.rewards)
        last_advantage = torch.zeros_like(last_values)

        for t in reversed(range(self.rewards.shape[0])):
            next_values = last_values if t == self.rewards.shape[0] - 1 else self.values[t + 1]
            next_not_done = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * next_values * next_not_done - self.values[t]
            last_advantage = delta + gamma * gae_lambda * next_not_done * last_advantage
            advantages[t] = last_advantage

        returns = advantages + self.values
        return returns, advantages

    def flatten(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(self.rewards.numel(), *values.shape[2:])


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def minibatch_indices(
    num_samples: int,
    num_minibatches: int,
    device: torch.device,
) -> list[torch.Tensor]:
    num_minibatches = max(1, min(int(num_minibatches), int(num_samples)))
    indices = torch.randperm(num_samples, device=device)
    return [chunk for chunk in indices.chunk(num_minibatches) if chunk.numel() > 0]
