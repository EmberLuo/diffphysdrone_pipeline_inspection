from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import quadsim_cuda  # noqa: E402
from training_code.env_cuda import Env as DepthEnv  # noqa: E402


class Env(DepthEnv):
    def __init__(
        self,
        *args,
        lidar_range: float = 4.0,
        lidar_hbeams: int = 120,
        lidar_vbeams: int = 6,
        lidar_vfov: tuple[float, float] = (-10.0, 20.0),
        **kwargs,
    ) -> None:
        self.lidar_range = float(lidar_range)
        self.lidar_hbeams = int(lidar_hbeams)
        self.lidar_vbeams = int(lidar_vbeams)
        self.lidar_vfov = (float(lidar_vfov[0]), float(lidar_vfov[1]))
        super().__init__(*args, **kwargs)

    def _lidar_frame(self) -> torch.Tensor:
        fwd = self.R[:, :, 0].clone()
        fwd[:, 2] = 0.0
        fwd = F.normalize(fwd, p=2, dim=-1, eps=1e-6)

        up = torch.zeros_like(fwd)
        up[:, 2] = 1.0
        left = torch.cross(up, fwd, dim=-1)
        return torch.stack([fwd, left, up], dim=-1).contiguous()

    def render_lidar(self) -> torch.Tensor:
        distances = torch.empty(
            (self.batch_size, self.lidar_hbeams, self.lidar_vbeams),
            device=self.device,
            dtype=self.p.dtype,
        )
        quadsim_cuda.render_lidar(
            distances,
            self.balls,
            self.cyl,
            self.cyl_h,
            self.voxels,
            self._lidar_frame(),
            self.p,
            self.drone_radius,
            self.n_drones_per_group,
            self.lidar_range,
            self.lidar_vfov[0],
            self.lidar_vfov[1],
        )
        scan = self.lidar_range - distances.clamp(0.0, self.lidar_range)
        return scan.clamp_(0.0, self.lidar_range)[:, None]
