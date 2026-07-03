#!/usr/bin/env python3
"""Sweep LiDAR configs to find which recovers coverage of collision-relevant,
NON-ground obstacles that the depth camera misses.

vfov / range are free (no model or CUDA change). vbeams changes the tensor
shape (would need a model edit) but the probe can still evaluate it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training_code.training_tasks.lidar_navrl.env import Env

DEVICE = torch.device("cuda")
FOV_X_HALF_TAN = 0.53
CAM_ANGLE = 10
N_RESETS = 25
BATCH = 256


def make_env(vfov, lidar_range, hbeams, vbeams, single=False):
    return Env(
        BATCH, 64, 48, 0.4, DEVICE,
        fov_x_half_tan=FOV_X_HALF_TAN, single=single, cam_angle=CAM_ANGLE,
        lidar_range=lidar_range, lidar_hbeams=hbeams, lidar_vbeams=vbeams,
        lidar_vfov=vfov,
    )


@torch.no_grad()
def evaluate(vfov, lidar_range, hbeams=120, vbeams=6, single=False):
    env = make_env(vfov, lidar_range, hbeams, vbeams, single)
    B = env.batch_size

    tot_nonground = 0
    depth_hit = 0
    lidar_hit = 0
    lidar_only = 0
    active_fracs = []

    for _ in range(N_RESETS):
        env.reset()
        env.v = F.normalize(env.p_target - env.p) * 1.5 + torch.randn_like(env.v) * 0.2

        lidar = env.render_lidar()
        active_fracs.append((lidar[:, 0] > 1e-4).float().mean(dim=(1, 2)))

        vec = env.find_vec_to_nearest_pt()[0]        # [B,3] drone->nearest
        dist = vec.norm(dim=-1)
        dir_unit = vec / dist.clamp_min(1e-6)[:, None]

        # ground = nearest point is (almost) straight down -> dz very negative.
        # (nearest_pt kernel places ground at z=-1 directly under the drone)
        is_ground = dir_unit[:, 2] < -0.85
        nonground = ~is_ground
        # only collision-relevant (close) non-ground obstacles matter
        relevant = nonground & (dist <= 3.0)

        # depth FOV test
        Rc = env.R @ env.R_cam
        f = (vec * Rc[:, :, 0]).sum(-1)
        l = (vec * Rc[:, :, 1]).sum(-1)
        u = (vec * Rc[:, :, 2]).sum(-1)
        fov_y = FOV_X_HALF_TAN * 48 / 64
        in_depth = (f > 0) & (l.abs() <= FOV_X_HALF_TAN * f.abs()) & (u.abs() <= fov_y * f.abs())

        # lidar coverage test (gravity-aligned, 360 az, vfov elevation, range)
        fwd = env.R[:, :, 0].clone(); fwd[:, 2] = 0
        fwd = F.normalize(fwd, dim=-1, eps=1e-6)
        up = torch.zeros_like(fwd); up[:, 2] = 1.0
        left = torch.cross(up, fwd, dim=-1)
        df = (vec * fwd).sum(-1); dl = (vec * left).sum(-1); du = (vec * up).sum(-1)
        horiz = torch.sqrt(df**2 + dl**2).clamp_min(1e-6)
        elev = torch.rad2deg(torch.atan2(du, horiz))
        in_lidar = (dist <= lidar_range) & (elev >= vfov[0]) & (elev <= vfov[1])

        tot_nonground += relevant.sum().item()
        depth_hit += (relevant & in_depth).sum().item()
        lidar_hit += (relevant & in_lidar).sum().item()
        lidar_only += (relevant & in_lidar & ~in_depth).sum().item()

    active = torch.cat(active_fracs).mean().item() * 100
    n = max(1, tot_nonground)
    return {
        "n_relevant": tot_nonground,
        "depth_cov": depth_hit / n * 100,
        "lidar_cov": lidar_hit / n * 100,
        "lidar_only": lidar_only / n * 100,
        "active_beam_pct": active,
    }


def main():
    configs = [
        ("baseline      [-10,+20] r4 6vb", (-10.0, 20.0), 4.0, 120, 6),
        ("down-tilt     [-30,+15] r4 6vb", (-30.0, 15.0), 4.0, 120, 6),
        ("down-tilt     [-45,+15] r4 6vb", (-45.0, 15.0), 4.0, 120, 6),
        ("wide-range    [-30,+15] r6 6vb", (-30.0, 15.0), 6.0, 120, 6),
        ("more-vbeams   [-30,+15] r4 12vb", (-30.0, 15.0), 4.0, 120, 12),
        ("more-vbeams   [-45,+20] r6 16vb", (-45.0, 20.0), 6.0, 120, 16),
    ]
    print(f"{'config':34s} | {'n_rel':>6s} | depth% | lidar% | Lonly% | act%")
    print("-" * 82)
    for name, vfov, rng, hb, vb in configs:
        r = evaluate(vfov, rng, hb, vb, single=False)
        print(f"{name:34s} | {r['n_relevant']:6d} | "
              f"{r['depth_cov']:5.1f}  | {r['lidar_cov']:5.1f}  | "
              f"{r['lidar_only']:5.1f}  | {r['active_beam_pct']:4.1f}")
    print("\n(n_rel = # of close non-ground nearest-obstacle samples; "
          "Lonly = lidar sees & depth misses)")


if __name__ == "__main__":
    main()
