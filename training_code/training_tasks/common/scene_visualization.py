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
    _log_observer_depth_camera(writer, step, sample_idx, env, p_history)
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


def _local_scene_limits(
    trajectory: np.ndarray,
    group_positions: np.ndarray,
    target: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    key_points = np.concatenate(
        [trajectory, group_positions, target[None, :]],
        axis=0,
    )
    key_points = key_points[np.isfinite(key_points).all(axis=1)]
    if key_points.size == 0:
        return (-5.0, 5.0), (-5.0, 5.0), (-1.5, 4.0)

    xy_min = key_points[:, :2].min(axis=0)
    xy_max = key_points[:, :2].max(axis=0)
    xy_center = (xy_min + xy_max) / 2
    xy_span = max(float((xy_max - xy_min).max()) + 5.0, 8.0)
    xy_half = xy_span / 2

    z_min = min(float(key_points[:, 2].min()) - 1.2, -1.2)
    z_max = max(float(key_points[:, 2].max()) + 2.0, 3.8)
    if z_max - z_min < 4.0:
        z_mid = (z_min + z_max) / 2
        z_min = z_mid - 2.0
        z_max = z_mid + 2.0

    return (
        (float(xy_center[0] - xy_half), float(xy_center[0] + xy_half)),
        (float(xy_center[1] - xy_half), float(xy_center[1] + xy_half)),
        (z_min, z_max),
    )


def _filter_balls_to_limits(
    balls: np.ndarray,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    z_lim: tuple[float, float],
) -> np.ndarray:
    if len(balls) == 0:
        return balls
    r = balls[:, 3]
    mask = balls[:, 0] + r >= x_lim[0]
    mask &= balls[:, 0] - r <= x_lim[1]
    mask &= balls[:, 1] + r >= y_lim[0]
    mask &= balls[:, 1] - r <= y_lim[1]
    mask &= balls[:, 2] + r >= z_lim[0]
    mask &= balls[:, 2] - r <= z_lim[1]
    return balls[mask]


def _filter_voxels_to_limits(
    voxels: np.ndarray,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    z_lim: tuple[float, float],
) -> np.ndarray:
    if len(voxels) == 0:
        return voxels
    mask = voxels[:, 0] + voxels[:, 3] >= x_lim[0]
    mask &= voxels[:, 0] - voxels[:, 3] <= x_lim[1]
    mask &= voxels[:, 1] + voxels[:, 4] >= y_lim[0]
    mask &= voxels[:, 1] - voxels[:, 4] <= y_lim[1]
    mask &= voxels[:, 2] + voxels[:, 5] >= z_lim[0]
    mask &= voxels[:, 2] - voxels[:, 5] <= z_lim[1]
    return voxels[mask]


def _filter_vertical_cylinders_to_limits(
    cylinders: np.ndarray,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> np.ndarray:
    if len(cylinders) == 0:
        return cylinders
    r = cylinders[:, 2]
    mask = cylinders[:, 0] + r >= x_lim[0]
    mask &= cylinders[:, 0] - r <= x_lim[1]
    mask &= cylinders[:, 1] + r >= y_lim[0]
    mask &= cylinders[:, 1] - r <= y_lim[1]
    return cylinders[mask]


def _filter_horizontal_cylinders_to_limits(
    cylinders_h: np.ndarray,
    x_lim: tuple[float, float],
    z_lim: tuple[float, float],
) -> np.ndarray:
    if len(cylinders_h) == 0:
        return cylinders_h
    r = cylinders_h[:, 2]
    mask = cylinders_h[:, 0] + r >= x_lim[0]
    mask &= cylinders_h[:, 0] - r <= x_lim[1]
    mask &= cylinders_h[:, 1] + r >= z_lim[0]
    mask &= cylinders_h[:, 1] - r <= z_lim[1]
    return cylinders_h[mask]


def _set_scene_axes(
    ax: Any,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    z_lim: tuple[float, float],
) -> None:
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(
            (
                x_lim[1] - x_lim[0],
                y_lim[1] - y_lim[0],
                (z_lim[1] - z_lim[0]) * 1.5,
            )
        )


def _plot_sphere(
    ax: Any,
    center: np.ndarray,
    radius: float,
    color: str,
    alpha: float,
    resolution: int = 16,
) -> None:
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def _plot_drone(
    ax: Any,
    center: np.ndarray,
    radius: float,
    color: str = "#111827",
    alpha: float = 0.85,
    resolution: int = 14,
) -> None:
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * 0.5 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def _plot_vertical_cylinder(
    ax: Any,
    cx: float,
    cy: float,
    radius: float,
    z_lim: tuple[float, float],
    color: str,
    alpha: float,
    resolution: int = 18,
) -> None:
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(z_lim[0], z_lim[1], 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = cx + radius * np.cos(theta_grid)
    y = cy + radius * np.sin(theta_grid)
    ax.plot_surface(x, y, z_grid, color=color, alpha=alpha, linewidth=0, shade=True)


def _plot_horizontal_cylinder(
    ax: Any,
    cx: float,
    cz: float,
    radius: float,
    y_lim: tuple[float, float],
    color: str,
    alpha: float,
    resolution: int = 18,
) -> None:
    theta = np.linspace(0, 2 * np.pi, resolution)
    y = np.linspace(y_lim[0], y_lim[1], 2)
    theta_grid, y_grid = np.meshgrid(theta, y)
    x = cx + radius * np.cos(theta_grid)
    z = cz + radius * np.sin(theta_grid)
    ax.plot_surface(x, y_grid, z, color=color, alpha=alpha, linewidth=0, shade=True)


def _camera_rotation_look_at(
    camera_pos: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    forward = target_pos - camera_pos
    forward = forward / forward.norm(p=2).clamp_min(1e-6)
    up_hint = torch.tensor([0.0, 0.0, 1.0], device=camera_pos.device)
    if torch.abs(torch.dot(forward, up_hint)) > 0.95:
        up_hint = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device)
    left = torch.cross(up_hint, forward, dim=0)
    left = left / left.norm(p=2).clamp_min(1e-6)
    up = torch.cross(forward, left, dim=0)
    up = up / up.norm(p=2).clamp_min(1e-6)
    return torch.stack([forward, left, up], dim=-1)


def _observer_camera_pose(
    trajectory: torch.Tensor,
    group_positions: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_points = torch.cat([trajectory, group_positions, target[None]], dim=0)
    focus = key_points.mean(dim=0)
    xy_min = key_points[:, :2].amin(dim=0)
    xy_max = key_points[:, :2].amax(dim=0)
    xy_span = (xy_max - xy_min).amax().clamp_min(8.0)
    distance = xy_span * 1.2 + 3.0
    view_dir = torch.tensor([-0.9, -1.2, 0.55], device=trajectory.device)
    view_dir = view_dir / view_dir.norm(p=2)
    camera_pos = focus + view_dir * distance
    camera_pos[2] = torch.maximum(camera_pos[2], focus[2] + xy_span * 0.35)
    return camera_pos, _camera_rotation_look_at(camera_pos, focus)


def _depth_to_rgb(depth: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(depth) & (depth < 99.0)
    if valid.any():
        near = torch.quantile(depth[valid], 0.02)
        far = torch.quantile(depth[valid], 0.98)
        if far <= near:
            far = near + 1.0
    else:
        near = depth.new_tensor(0.0)
        far = depth.new_tensor(1.0)

    normalized = ((depth.clamp(float(near), float(far)) - near) / (far - near)).cpu()
    inverse = 1.0 - normalized.numpy()
    rgb = plt.get_cmap("magma")(inverse)[..., :3]
    rgb[~valid.cpu().numpy()] = 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def _log_observer_depth_camera(
    writer: SummaryWriter,
    step: int,
    sample_idx: int,
    env: Any,
    p_history: torch.Tensor,
) -> None:
    import quadsim_cuda

    batch_base, batch_end = _scene_group_bounds(env, sample_idx)
    trajectory = p_history[:, sample_idx].detach()
    group_positions = env.p[batch_base:batch_end].detach()
    target = env.p_target[sample_idx].detach()
    camera_pos, camera_R = _observer_camera_pose(trajectory, group_positions, target)

    n_group = int(group_positions.shape[0])
    fake_batch = n_group + 1
    height, width = 720, 960
    device = env.p.device
    dtype = env.p.dtype

    canvas = torch.empty((fake_batch, height, width), device=device, dtype=dtype)
    flow = torch.empty((fake_batch, 0, height, width), device=device, dtype=dtype)
    balls = env.balls[batch_base : batch_base + 1].repeat(fake_batch, 1, 1)
    cylinders = env.cyl[batch_base : batch_base + 1].repeat(fake_batch, 1, 1)
    cylinders_h = env.cyl_h[batch_base : batch_base + 1].repeat(fake_batch, 1, 1)
    voxels = env.voxels[batch_base : batch_base + 1].repeat(fake_batch, 1, 1)

    pos = torch.empty((fake_batch, 3), device=device, dtype=dtype)
    pos[0] = camera_pos
    pos[1:] = group_positions
    R = torch.eye(3, device=device, dtype=dtype).repeat(fake_batch, 1, 1)
    R[0] = camera_R.to(dtype=dtype)

    quadsim_cuda.render(
        canvas,
        flow,
        balls,
        cylinders,
        cylinders_h,
        voxels,
        R,
        R,
        pos,
        pos,
        float(getattr(env, "drone_radius", 0.15)),
        fake_batch,
        0.7,
    )
    writer.add_image("scene/observer_depth_camera", _depth_to_rgb(canvas[0]), step)


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

    x_lim, y_lim, z_lim = _local_scene_limits(trajectory, group_positions, target)
    balls = _filter_balls_to_limits(balls, x_lim, y_lim, z_lim)
    voxels = _filter_voxels_to_limits(voxels, x_lim, y_lim, z_lim)
    cylinders = _filter_vertical_cylinders_to_limits(cylinders, x_lim, y_lim)
    cylinders_h = _filter_horizontal_cylinders_to_limits(cylinders_h, x_lim, z_lim)

    fig = plt.figure(figsize=(10.5, 7.0), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    _set_scene_axes(ax, x_lim, y_lim, z_lim)

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
            alpha=0.16,
            shade=True,
            linewidth=0.1,
        )

    if len(balls):
        for ball in balls[:28]:
            _plot_sphere(ax, ball[:3], float(ball[3]), "#d97706", 0.32)

    for cx, cy, radius in cylinders[:80]:
        _plot_vertical_cylinder(
            ax,
            float(cx),
            float(cy),
            float(radius),
            z_lim,
            "#0891b2",
            0.22,
        )

    for cx, cz, radius in cylinders_h[:80]:
        _plot_horizontal_cylinder(
            ax,
            float(cx),
            float(cz),
            float(radius),
            y_lim,
            "#0f766e",
            0.22,
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
    drone_radius = float(getattr(env, "drone_radius", 0.12))
    for index, position in enumerate(group_positions):
        color = "#111827" if batch_base + index == sample_idx else "#6b7280"
        _plot_drone(ax, position, drone_radius, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"observer camera, sample {sample_idx}")
    ax.view_init(elev=26, azim=-44)
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
