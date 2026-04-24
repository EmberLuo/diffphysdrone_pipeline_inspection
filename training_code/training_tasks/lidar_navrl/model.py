import torch
from torch import nn


class LidarNavRLModel(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
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

    def forward(self, x: torch.Tensor, v, hx=None):
        lidar_feat = self.stem(x)
        x = self.act(lidar_feat + self.v_proj(v))
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, None, hx


Model = LidarNavRLModel


if __name__ == "__main__":
    LidarNavRLModel(10, 6)
