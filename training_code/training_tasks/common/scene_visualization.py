from __future__ import annotations

from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def log_training_scene_snapshot(
    writer: SummaryWriter,
    step: int,
    sample_idx: int,
    env: Any,
    p_history: torch.Tensor,
) -> None:
    _log_observer_snapshot(writer, step, sample_idx, env, p_history)
    _log_depth_snapshot(writer, step, sample_idx, env)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _scene_group_bounds(env: Any, sample_idx: int) -> tuple[int, int]:
    n_drones = max(1, int(getattr(env, "n_drones_per_group", 1)))
    batch_base = (sample_idx // n_drones) * n_drones
    batch_end = min(
        batch_base + n_drones,
        int(getattr(env, "batch_size", sample_idx + 1)),
    )
    return batch_base, batch_end


def _filter_scene_rows(values: np.ndarray, size_slice: slice, max_size: float) -> np.ndarray:
    if values.size == 0:
        return values
    mask = np.isfinite(values).all(axis=1)
    sizes = values[:, size_slice]
    mask &= (sizes > 0).all(axis=1)
    mask &= (sizes < max_size).all(axis=1)
    return values[mask]


def _set_scene_axes(ax: Any, points: np.ndarray) -> None:
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = max(float((maxs - mins).max()), 1.0) + 1.0
    half = span / 2

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 0.65))


def _log_observer_snapshot(
    writer: SummaryWriter,
    step: int,
    sample_idx: int,
    env: Any,
    p_history: torch.Tensor,
) -> None:
    batch_base, batch_end = _scene_group_bounds(env, sample_idx)

    trajectory = _to_numpy(p_history[:, sample_idx])
    group_positions = _to_numpy(env.p[batch_base:batch_end])
    target = _to_numpy(env.p_target[sample_idx])
    z_mid = float(np.median(trajectory[:, 2]))
    y_mid = float(np.median(trajectory[:, 1]))

    balls = _filter_scene_rows(
        _to_numpy(env.balls[batch_base]), slice(3, 4), 6.0
    )
    voxels = _filter_scene_rows(
        _to_numpy(env.voxels[batch_base]), slice(3, 6), 20.0
    )
    cylinders = _filter_scene_rows(
        _to_numpy(env.cyl[batch_base]), slice(2, 3), 5.0
    )
    cylinders_h = _filter_scene_rows(
        _to_numpy(env.cyl_h[batch_base]), slice(2, 3), 5.0
    )

    points = [trajectory, group_positions, target[None, :]]
    if len(balls):
        points.append(balls[:, :3])
    if len(voxels):
        points.append(voxels[:, :3])
    if len(cylinders):
        points.append(
            np.column_stack(
                [cylinders[:, 0], cylinders[:, 1], np.full(len(cylinders), z_mid)]
            )
        )
    if len(cylinders_h):
        points.append(
            np.column_stack(
                [
                    cylinders_h[:, 0],
                    np.full(len(cylinders_h), y_mid),
                    cylinders_h[:, 1],
                ]
            )
        )

    fig = plt.figure(figsize=(7.0, 5.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    _set_scene_axes(ax, np.concatenate(points, axis=0))

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], 2),
        np.linspace(y_lim[0], y_lim[1], 2),
    )
    ax.plot_surface(
        xx,
        yy,
        np.full_like(xx, -1.0),
        color="#e5e7eb",
        alpha=0.18,
        linewidth=0,
    )

    if len(voxels):
        ax.bar3d(
            voxels[:, 0] - voxels[:, 3],
            voxels[:, 1] - voxels[:, 4],
            voxels[:, 2] - voxels[:, 5],
            2 * voxels[:, 3],
            2 * voxels[:, 4],
            2 * voxels[:, 5],
            color="#94a3b8",
            alpha=0.18,
            shade=True,
            linewidth=0.1,
        )

    if len(balls):
        ax.scatter(
            balls[:, 0],
            balls[:, 1],
            balls[:, 2],
            s=np.clip((balls[:, 3] * 90) ** 2, 16, 900),
            c="#d97706",
            alpha=0.28,
            depthshade=True,
        )

    for cx, cy, radius in cylinders[:80]:
        ax.plot(
            [cx, cx],
            [cy, cy],
            [z_lim[0], z_lim[1]],
            color="#0891b2",
            alpha=0.24,
            linewidth=float(np.clip(radius * 8, 0.5, 3.0)),
        )

    for cx, cz, radius in cylinders_h[:80]:
        ax.plot(
            [cx, cx],
            [y_lim[0], y_lim[1]],
            [cz, cz],
            color="#0f766e",
            alpha=0.24,
            linewidth=float(np.clip(radius * 8, 0.5, 3.0)),
        )

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="#2563eb",
        linewidth=2.2,
    )
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        color="#16a34a",
        s=55,
        marker="o",
    )
    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        trajectory[-1, 2],
        color="#111827",
        s=45,
        marker="^",
    )
    ax.scatter(target[0], target[1], target[2], color="#dc2626", s=80, marker="*")
    ax.scatter(
        group_positions[:, 0],
        group_positions[:, 1],
        group_positions[:, 2],
        color="#111827",
        s=18,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"observer camera, sample {sample_idx}")
    ax.view_init(elev=24, azim=-58)
    try:
        ax.set_proj_type("persp")
    except AttributeError:
        pass
    fig.tight_layout()

    writer.add_figure("scene/observer_camera", fig, step)
    plt.close(fig)


def _log_depth_snapshot(
    writer: SummaryWriter,
    step: int,
    sample_idx: int,
    env: Any,
) -> None:
    depth, _ = env.render(1 / 15)
    depth = depth[sample_idx].detach().float().clamp(0.3, 24)
    inv_depth = 1.0 / depth
    inv_depth = (inv_depth - inv_depth.amin()) / (
        inv_depth.amax() - inv_depth.amin()
    ).clamp_min(1e-6)
    writer.add_image("scene/onboard_depth_camera", inv_depth.unsqueeze(0).cpu(), step)
