#!/usr/bin/env python3
"""Diagnostic probe: how complementary is the LiDAR vs the depth camera?

For a batch of (multi-agent) states we compute the collision-relevant nearest
obstacle (the same signal used for the avoidance loss) and ask, per drone:
  - is that obstacle inside the depth-camera FOV?
  - is it inside the LiDAR coverage (360 az x vfov)?
We also dump raw distribution stats of both sensor tensors, and how coarsely
the LiDAR samples a neighbouring drone.
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

# match the fusion multi_agent training defaults
LIDAR_RANGE = 4.0
LIDAR_HBEAMS = 120
LIDAR_VBEAMS = 6
VFOV = (-10.0, 20.0)
FOV_X_HALF_TAN = 0.53   # nominal navigation default (no config file)
CAM_ANGLE = 10


def build_env(single: bool, batch_size: int = 256) -> Env:
    return Env(
        batch_size, 64, 48, 0.4, DEVICE,
        fov_x_half_tan=FOV_X_HALF_TAN,
        single=single,
        cam_angle=CAM_ANGLE,
        lidar_range=LIDAR_RANGE,
        lidar_hbeams=LIDAR_HBEAMS,
        lidar_vbeams=LIDAR_VBEAMS,
        lidar_vfov=VFOV,
    )


@torch.no_grad()
def probe(single: bool, n_resets: int = 20):
    env = build_env(single)
    B = env.batch_size

    depth_fov_hits = 0
    lidar_fov_hits = 0
    both = 0
    neither = 0
    total = 0
    near_dists = []

    depth_vals = []
    lidar_vals = []
    lidar_active_frac = []  # fraction of beams that see something within range
    elev_all = []
    below_vfov = []
    within_range = []

    for _ in range(n_resets):
        env.reset()
        # give a forward-ish velocity toward target so attitude is realistic
        env.v = F.normalize(env.p_target - env.p) * 1.5 + torch.randn_like(env.v) * 0.2

        # --- raw sensor tensors ---
        depth, _ = env.render(1 / 15)
        depth_obs = 3 / depth.clamp_(0.3, 24) - 0.6
        lidar = env.render_lidar()  # [B,1,H,V], value = range - dist (closeness)
        depth_vals.append(depth_obs.flatten())
        lidar_vals.append(lidar.flatten())
        # a beam is "active" if it hit something inside range -> closeness > 0
        active = (lidar[:, 0] > 1e-4).float().mean(dim=(1, 2))
        lidar_active_frac.append(active)

        # --- collision-relevant nearest obstacle ---
        vec = env.find_vec_to_nearest_pt()  # [10 subdiv, B, 3] direction drone->nearest
        d = vec[0]  # use current position sub-step
        dist = d.norm(dim=-1)
        near_dists.append(dist)

        # camera frame columns
        Rc = env.R @ env.R_cam
        fwd_c = Rc[:, :, 0]
        left_c = Rc[:, :, 1]
        up_c = Rc[:, :, 2]
        f = (d * fwd_c).sum(-1)
        l = (d * left_c).sum(-1)
        u = (d * up_c).sum(-1)
        fov_y = FOV_X_HALF_TAN * 48 / 64
        in_depth = (f > 0) & (l.abs() <= FOV_X_HALF_TAN * f.abs()) & (u.abs() <= fov_y * f.abs())

        # lidar frame
        fwd = env.R[:, :, 0].clone()
        fwd[:, 2] = 0
        fwd = F.normalize(fwd, dim=-1, eps=1e-6)
        up = torch.zeros_like(fwd)
        up[:, 2] = 1.0
        left = torch.cross(up, fwd, dim=-1)
        dl_f = (d * fwd).sum(-1)
        dl_l = (d * left).sum(-1)
        dl_u = (d * up).sum(-1)
        horiz = torch.sqrt(dl_f**2 + dl_l**2).clamp_min(1e-6)
        elev = torch.rad2deg(torch.atan2(dl_u, horiz))
        in_lidar = (dist <= LIDAR_RANGE) & (elev >= VFOV[0]) & (elev <= VFOV[1])
        # azimuth always covered (360)

        depth_fov_hits += in_depth.sum().item()
        lidar_fov_hits += in_lidar.sum().item()
        both += (in_depth & in_lidar).sum().item()
        neither += (~in_depth & ~in_lidar).sum().item()
        total += B

        # elevation of nearest obstacle in the (gravity-aligned) lidar frame
        elev_all.append(elev)
        below_vfov.append(elev < VFOV[0])          # can't be seen by lidar (too low)
        within_range.append(dist <= LIDAR_RANGE)

    depth_vals = torch.cat(depth_vals)
    lidar_vals = torch.cat(lidar_vals)
    near = torch.cat(near_dists)
    active = torch.cat(lidar_active_frac)
    elev_all = torch.cat(elev_all)
    below = torch.cat(below_vfov)
    inrange = torch.cat(within_range)

    tag = "SINGLE-agent" if single else "MULTI-agent"
    print(f"\n================ {tag} ({total} drone-samples) ================")
    print("Nearest-obstacle coverage (the collision-relevant signal):")
    print(f"  in depth FOV        : {depth_fov_hits/total*100:5.1f}%")
    print(f"  in LiDAR coverage   : {lidar_fov_hits/total*100:5.1f}%")
    print(f"  in BOTH             : {both/total*100:5.1f}%")
    print(f"  LiDAR-only (depth misses, lidar sees): {(lidar_fov_hits-both)/total*100:5.1f}%")
    print(f"  in NEITHER          : {neither/total*100:5.1f}%")
    print(f"  nearest-obstacle distance: mean={near.mean():.2f} m, "
          f"p10={near.quantile(0.1):.2f}, median={near.median():.2f}")
    print("Nearest-obstacle elevation in gravity-aligned frame:")
    print(f"  elevation: mean={elev_all.mean():.1f} deg, "
          f"p10={elev_all.quantile(0.1):.1f}, median={elev_all.median():.1f}, p90={elev_all.quantile(0.9):.1f}")
    print(f"  below LiDAR vfov ({VFOV[0]} deg): {below.float().mean()*100:.1f}%  "
          f"(mostly the ground directly under the drone)")
    print(f"  within {LIDAR_RANGE} m range     : {inrange.float().mean()*100:.1f}%")
    print("Raw sensor stats:")
    print(f"  depth_obs : min={depth_vals.min():.3f} max={depth_vals.max():.3f} "
          f"mean={depth_vals.mean():.3f} std={depth_vals.std():.3f}")
    print(f"  lidar_obs : min={lidar_vals.min():.3f} max={lidar_vals.max():.3f} "
          f"mean={lidar_vals.mean():.3f} std={lidar_vals.std():.3f}")
    print(f"  lidar frac of beams active (hit within {LIDAR_RANGE}m): "
          f"mean={active.mean()*100:.1f}%  (so {100-active.mean()*100:.1f}% of scan is empty=0)")


if __name__ == "__main__":
    probe(single=True)
    probe(single=False)
