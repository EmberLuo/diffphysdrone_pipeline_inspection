from __future__ import annotations

import torch
from torch import nn


class LidarDepthFusionModel(nn.Module):
    def __init__(self, dim_obs: int = 9, dim_action: int = 4) -> None:
        super().__init__()
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 192, bias=False),
        )
        self.lidar_stem = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 3), padding=(2, 1)),
            nn.ELU(),
            nn.Conv2d(4, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16 * 30 * 3, 192),
            nn.LayerNorm(192),
        )
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, obs, v: torch.Tensor, hx=None):
        if not isinstance(obs, (tuple, list)) or len(obs) != 2:
            raise ValueError("LidarDepthFusionModel expects obs=(depth_obs, lidar_obs).")
        depth_obs, lidar_obs = obs
        fused = self.depth_stem(depth_obs) + self.lidar_stem(lidar_obs) + self.v_proj(v)
        hx = self.gru(self.act(fused), hx)
        act = self.fc(self.act(hx))
        return act, None, hx


Model = LidarDepthFusionModel


if __name__ == "__main__":
    model = LidarDepthFusionModel(10, 6)
    depth = torch.randn(2, 1, 12, 16)
    lidar = torch.randn(2, 1, 120, 6)
    state = torch.randn(2, 10)
    action, _, hidden = model((depth, lidar), state)
    print(action.shape, hidden.shape)
