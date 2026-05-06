from __future__ import annotations

import math

import torch
from torch import nn

from training_code.model import Model as DepthActor


class DepthCritic(nn.Module):
    hidden_size = 192

    def __init__(self, dim_obs: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, self.hidden_size, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, self.hidden_size)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.stem(obs) + self.v_proj(state))
        hx = self.gru(x, hx)
        value = self.fc(self.act(hx)).squeeze(-1)
        return value, hx


class DepthDiffPPOActorCritic(nn.Module):
    action_dim = 6
    hidden_size = 192

    def __init__(self, state_dim: int, init_log_std: float = -0.5) -> None:
        super().__init__()
        self.actor = DepthActor(state_dim, self.action_dim)
        self.critic = DepthCritic(state_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_log_std)))

    @property
    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.actor.parameters()) + [self.log_std]

    def reset(self) -> None:
        self.actor.reset()

    def _dist_stats(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = log_std.exp().expand_as(mean)
        return log_std, std

    @staticmethod
    def _log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        var_term = ((action - mean) / std).pow(2)
        log_prob = -0.5 * (var_term + 2.0 * log_std + math.log(2.0 * math.pi))
        return log_prob.sum(dim=-1)

    def act(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        hx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, _, next_hx = self.actor(obs, state, hx)
        log_std, std = self._dist_stats(mean)
        action = mean + std * torch.randn_like(mean)
        log_prob = self._log_prob(action, mean, log_std, std)
        return action, log_prob, mean, next_hx

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        actor_hx: torch.Tensor,
        critic_hx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, _, _ = self.actor(obs, state, actor_hx)
        log_std, std = self._dist_stats(mean)
        log_prob = self._log_prob(actions, mean, log_std, std)
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum().expand_as(log_prob)
        value, _ = self.critic(obs, state, critic_hx)
        return log_prob, entropy, value

    def value(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        hx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic(obs, state, hx)
