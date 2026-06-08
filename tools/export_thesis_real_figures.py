#!/usr/bin/env python3
"""Export thesis replacement figures from real PCD, ROS CSV, and eval logs."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from thesis_plot_style import setup_chinese_matplotlib
except ImportError:  # pragma: no cover
    from tools.thesis_plot_style import setup_chinese_matplotlib


setup_chinese_matplotlib()

DEFAULT_CLOUD_MAX_POINTS = 16000
DEFAULT_OCCUPANCY_MAX_POINTS = 12000
DEFAULT_Z_LIMIT_M = 10.0


def load_pcd_xyz(path: Path) -> np.ndarray:
    header_lines: list[str] = []
    data_chunks: list[bytes] = []
    with path.open("rb") as fp:
        while True:
            raw = fp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            header_lines.append(line)
            if line.lower().startswith("data"):
                data_chunks.append(fp.read())
                break

    header: dict[str, list[str]] = {}
    for line in header_lines:
        parts = line.split()
        if parts:
            header[parts[0].upper()] = parts[1:]

    data_mode = header.get("DATA", [""])[0].lower()
    fields = header.get("FIELDS", [])
    points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    if not fields or not data_mode:
        raise ValueError(f"PCD header is incomplete: {path}")

    if data_mode == "binary":
        sizes = [int(v) for v in header.get("SIZE", [])]
        types = header.get("TYPE", [])
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"Unsupported PCD field metadata: {path}")
        if any(c != 1 for c in counts):
            raise ValueError(f"Unsupported multi-count PCD fields: {path}")
        dtype_fields = []
        for name, size, typ in zip(fields, sizes, types):
            if typ == "F" and size == 4:
                dtype_fields.append((name, "<f4"))
            elif typ == "F" and size == 8:
                dtype_fields.append((name, "<f8"))
            elif typ == "I" and size == 4:
                dtype_fields.append((name, "<i4"))
            elif typ == "U" and size == 4:
                dtype_fields.append((name, "<u4"))
            else:
                raise ValueError(f"Unsupported PCD field type {typ}{size}: {path}")
        arr = np.frombuffer(b"".join(data_chunks), dtype=np.dtype(dtype_fields), count=points)
        return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(float)

    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        lines = fp.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("data"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"PCD DATA header not found: {path}")
    rows = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 3:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.asarray(rows, dtype=float)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def load_traj(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        arr = next(iter(obj.values()))
    else:
        arr = obj
    data = np.asarray(arr, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Unexpected trajectory shape in {path}: {data.shape}")
    return data[:, :3]


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


def downsample(points: np.ndarray, max_points: int = DEFAULT_CLOUD_MAX_POINTS, seed: int = 7) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[idx]


def setup_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")


def setup_3d_axis(
    ax: plt.Axes,
    points: np.ndarray,
    z_limit: float = DEFAULT_Z_LIMIT_M,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("")
    ax.view_init(elev=28, azim=-58)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad_xy = 1.0
    x_min, x_max = xlim if xlim is not None else (float(mins[0] - pad_xy), float(maxs[0] + pad_xy))
    y_min, y_max = ylim if ylim is not None else (float(mins[1] - pad_xy), float(maxs[1] + pad_xy))
    x_span = max(1.0, float(x_max - x_min))
    y_span = max(1.0, float(y_max - y_min))
    z_top = max(1.0, float(z_limit))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0.0, z_top)
    try:
        ax.set_box_aspect((x_span, y_span, z_top * 1.25))
    except AttributeError:
        pass


def sparse_voxel_mask(points: np.ndarray, res: float = 0.35, max_count: int = 1) -> np.ndarray:
    mins = points.min(axis=0) - res
    keys = np.floor((points - mins) / res).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse] <= max_count


def plot_raw_cloud_stage(
    points: np.ndarray,
    out: Path,
    title: str = "原始PCD采样点云",
    max_points: int = 34000,
    seed: int = 5,
) -> None:
    if len(points) <= max_points:
        idx = np.arange(len(points))
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
    pts = points[idx]
    sparse = sparse_voxel_mask(points)[idx]
    dense = pts[~sparse]
    isolated = pts[sparse]
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        dense[:, 0],
        dense[:, 1],
        dense[:, 2],
        s=1.2,
        c="#58606f",
        alpha=0.32,
        linewidths=0,
        depthshade=False,
        label="原始密集采样",
    )
    if len(isolated):
        ax.scatter(
            isolated[:, 0],
            isolated[:, 1],
            isolated[:, 2],
            s=3.0,
            c="#d62728",
            alpha=0.72,
            linewidths=0,
            depthshade=False,
            label="低支撑离散点",
        )
    ax.set_title(title)
    setup_3d_axis(ax, points)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    savefig(fig, out)


def plot_cloud_overview(
    points: np.ndarray,
    out: Path,
    title: str,
    max_points: int = DEFAULT_CLOUD_MAX_POINTS,
    z_limit: float = DEFAULT_Z_LIMIT_M,
    seed: int = 7,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    plot_points = points
    if xlim is not None:
        plot_points = plot_points[(plot_points[:, 0] >= xlim[0]) & (plot_points[:, 0] <= xlim[1])]
    if ylim is not None:
        plot_points = plot_points[(plot_points[:, 1] >= ylim[0]) & (plot_points[:, 1] <= ylim[1])]
    if len(plot_points) == 0:
        plot_points = points
    pts = downsample(plot_points, max_points=max_points, seed=seed)
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=pts[:, 2],
        s=2.0,
        cmap="viridis",
        alpha=0.82,
        linewidths=0,
        depthshade=False,
    )
    ax.set_title(title)
    setup_3d_axis(ax, plot_points, z_limit=z_limit, xlim=xlim, ylim=ylim)
    fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, label="高度 / m")
    savefig(fig, out)


def plot_filtered_cloud_comparison(raw: np.ndarray, filtered: np.ndarray, out: Path) -> None:
    raw_pts = downsample(raw, max_points=28000, seed=17)
    filtered_pts = downsample(filtered, max_points=12000, seed=19)
    fig = plt.figure(figsize=(8.6, 4.8))
    ax_raw = fig.add_subplot(121, projection="3d")
    ax_filtered = fig.add_subplot(122, projection="3d")

    ax_raw.scatter(
        raw_pts[:, 0],
        raw_pts[:, 1],
        raw_pts[:, 2],
        s=1.0,
        c="#7b8794",
        alpha=0.30,
        linewidths=0,
        depthshade=False,
    )
    ax_raw.set_title(f"滤波前：{len(raw):,}点")
    setup_3d_axis(ax_raw, raw)

    sc = ax_filtered.scatter(
        filtered_pts[:, 0],
        filtered_pts[:, 1],
        filtered_pts[:, 2],
        c=filtered_pts[:, 2],
        s=2.2,
        cmap="viridis",
        alpha=0.86,
        linewidths=0,
        depthshade=False,
    )
    ax_filtered.set_title(f"滤波后：{len(filtered):,}点")
    setup_3d_axis(ax_filtered, raw)
    fig.suptitle("点云滤波与降采样结果对比", y=0.98)
    fig.colorbar(sc, ax=ax_filtered, shrink=0.62, pad=0.08, label="高度 / m")
    savefig(fig, out)


def plot_cloud_topdown(points: np.ndarray, out: Path, title: str, color: str) -> None:
    pts = downsample(points)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.65, c=color, alpha=0.65, linewidths=0)
    ax.set_title(title)
    setup_axis(ax, "x / m", "y / m")
    savefig(fig, out)


def occupancy_image(points: np.ndarray, res: float = 0.25) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    xy = points[:, :2]
    min_xy = xy.min(axis=0) - 1.0
    max_xy = xy.max(axis=0) + 1.0
    nx = int(math.ceil((max_xy[0] - min_xy[0]) / res))
    ny = int(math.ceil((max_xy[1] - min_xy[1]) / res))
    ix = np.clip(((xy[:, 0] - min_xy[0]) / res).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - min_xy[1]) / res).astype(int), 0, ny - 1)
    occ = np.zeros((ny, nx), dtype=np.uint8)
    occ[iy, ix] = 1
    return occ, (min_xy[0], max_xy[0], min_xy[1], max_xy[1])


def occupancy_voxel_centers(points: np.ndarray, res: float = 0.25) -> np.ndarray:
    mins = points.min(axis=0) - res
    keys = np.floor((points - mins) / res).astype(np.int64)
    occupied = np.unique(keys, axis=0)
    return mins + (occupied.astype(float) + 0.5) * res


def voxel_keys(points: np.ndarray, res: float, padding: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    mins = points.min(axis=0) - padding
    keys = np.floor((points - mins) / res).astype(np.int64)
    return np.unique(keys, axis=0), mins


def centers_from_keys(keys: np.ndarray, mins: np.ndarray, res: float) -> np.ndarray:
    return mins + (keys.astype(float) + 0.5) * res


def rows_to_void(keys: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(keys)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def inflated_voxel_sets(points: np.ndarray, res: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    occupied, mins = voxel_keys(points, res=res, padding=res)
    cells = int(math.ceil(radius / res))
    offsets = np.asarray(
        [
            (dx, dy, dz)
            for dx in range(-cells, cells + 1)
            for dy in range(-cells, cells + 1)
            for dz in range(-cells, cells + 1)
            if (dx * res) ** 2 + (dy * res) ** 2 + (dz * res) ** 2 <= radius**2
        ],
        dtype=np.int64,
    )
    inflated = np.unique((occupied[:, None, :] + offsets[None, :, :]).reshape(-1, 3), axis=0)
    occ_void = rows_to_void(occupied)
    inflated_void = rows_to_void(inflated)
    shell = inflated[~np.isin(inflated_void, occ_void)]
    occupied_centers = centers_from_keys(occupied, mins, res)
    shell_centers = centers_from_keys(shell, mins, res)
    valid_shell = (shell_centers[:, 2] >= 0.0) & (shell_centers[:, 2] <= DEFAULT_Z_LIMIT_M)
    return occupied_centers, shell_centers[valid_shell]


def inflate_occupancy(occ: np.ndarray, res: float, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return occ.copy()
    inflated = occ.copy()
    cells = int(math.ceil(radius / res))
    ys, xs = np.where(occ)
    offsets = [
        (dy, dx)
        for dy in range(-cells, cells + 1)
        for dx in range(-cells, cells + 1)
        if (dx * res) ** 2 + (dy * res) ** 2 <= radius**2
    ]
    for dy, dx in offsets:
        yy = np.clip(ys + dy, 0, occ.shape[0] - 1)
        xx = np.clip(xs + dx, 0, occ.shape[1] - 1)
        inflated[yy, xx] = True
    return inflated


def nearest_free_cell(occ: np.ndarray, cell: tuple[int, int], max_radius: int = 30) -> tuple[int, int]:
    if not occ[cell]:
        return cell
    cy, cx = cell
    best = cell
    best_dist = float("inf")
    for radius in range(1, max_radius + 1):
        y0, y1 = max(0, cy - radius), min(occ.shape[0], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(occ.shape[1], cx + radius + 1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if occ[y, x]:
                    continue
                dist = (y - cy) ** 2 + (x - cx) ** 2
                if dist < best_dist:
                    best = (y, x)
                    best_dist = dist
        if best_dist < float("inf"):
            return best
    return best


def simplify_path_xy(path: np.ndarray) -> np.ndarray:
    if len(path) <= 2:
        return path
    keep = [0]
    prev = np.sign(path[1] - path[0]).astype(int)
    for i in range(2, len(path)):
        cur = np.sign(path[i] - path[i - 1]).astype(int)
        if not np.array_equal(cur, prev):
            keep.append(i - 1)
            prev = cur
    keep.append(len(path) - 1)
    return path[keep]


def demo_astar_path(points: np.ndarray) -> np.ndarray:
    res = 0.5
    inflation = 0.45
    xy = points[:, :2]
    min_xy = xy.min(axis=0) - 1.5
    max_xy = xy.max(axis=0) + 1.5
    nx = int(math.ceil((max_xy[0] - min_xy[0]) / res))
    ny = int(math.ceil((max_xy[1] - min_xy[1]) / res))
    ix = np.clip(((xy[:, 0] - min_xy[0]) / res).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - min_xy[1]) / res).astype(int), 0, ny - 1)
    occ = np.zeros((ny, nx), dtype=bool)
    occ[iy, ix] = True
    inflated = inflate_occupancy(occ, res, inflation)

    def to_cell(point: tuple[float, float]) -> tuple[int, int]:
        y = int(np.clip((point[1] - min_xy[1]) / res, 0, ny - 1))
        x = int(np.clip((point[0] - min_xy[0]) / res, 0, nx - 1))
        return y, x

    def to_world(cell: tuple[int, int]) -> np.ndarray:
        y, x = cell
        return np.asarray([min_xy[0] + (x + 0.5) * res, min_xy[1] + (y + 0.5) * res], dtype=float)

    start = nearest_free_cell(inflated, to_cell((-28.0, -12.0)))
    goal = nearest_free_cell(inflated, to_cell((28.0, 12.0)))
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_set: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        for dy, dx in neighbors:
            nxt = current[0] + dy, current[1] + dx
            if nxt[0] < 0 or nxt[0] >= ny or nxt[1] < 0 or nxt[1] >= nx or inflated[nxt]:
                continue
            new_cost = cost_so_far[current] + math.hypot(dx, dy)
            if new_cost >= cost_so_far.get(nxt, float("inf")):
                continue
            cost_so_far[nxt] = new_cost
            priority = new_cost + math.hypot(nxt[0] - goal[0], nxt[1] - goal[1])
            came_from[nxt] = current
            heapq.heappush(open_set, (priority, nxt))

    if goal not in cost_so_far:
        raise RuntimeError("Failed to generate the demo A* path")
    cells = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        cells.append(current)
    path_xy = simplify_path_xy(np.asarray([to_world(cell) for cell in reversed(cells)], dtype=float))
    z = np.full((len(path_xy), 1), 1.8, dtype=float)
    return np.hstack([path_xy, z])


def resample_polyline(path: np.ndarray, step: float) -> np.ndarray:
    if len(path) < 2:
        return path
    samples = [path[0]]
    carried = 0.0
    prev = path[0].copy()
    for point in path[1:]:
        cur = point.copy()
        segment = float(np.linalg.norm(cur - prev))
        if segment <= 1e-9:
            continue
        while carried + segment >= step:
            ratio = (step - carried) / segment
            sample = prev + (cur - prev) * ratio
            samples.append(sample)
            prev = sample
            segment = float(np.linalg.norm(cur - prev))
            carried = 0.0
            if segment <= 1e-9:
                break
        carried += segment
        prev = cur
    if not np.allclose(samples[-1], path[-1]):
        samples.append(path[-1])
    return np.asarray(samples, dtype=float)


def local_target_samples_from_path(path: np.ndarray, odom_step: float = 0.8, lookahead: float = 2.0) -> list[dict[str, float]]:
    dense_path = resample_polyline(path, 0.25)
    odom = resample_polyline(path, odom_step)
    arc = np.zeros(len(dense_path), dtype=float)
    if len(dense_path) > 1:
        arc[1:] = np.cumsum(np.linalg.norm(np.diff(dense_path, axis=0), axis=1))
    rows = []
    for i, pos in enumerate(odom):
        nearest = int(np.argmin(np.linalg.norm(dense_path - pos, axis=1)))
        target_s = min(arc[-1], arc[nearest] + lookahead)
        target_idx = int(np.searchsorted(arc, target_s, side="left"))
        target_idx = min(target_idx, len(dense_path) - 1)
        target = dense_path[target_idx]
        rows.append(
            {
                "time_s": float(i) * 0.4,
                "odom_x": float(pos[0]),
                "odom_y": float(pos[1]),
                "odom_z": float(pos[2]),
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "target_z": float(target[2]),
                "target_distance_m": float(np.linalg.norm(target - pos)),
            }
        )
    return rows


def plot_occupancy(points: np.ndarray, out: Path) -> None:
    occupied, inflated_shell = inflated_voxel_sets(points, res=0.45, radius=0.75)
    occupied_plot = downsample(occupied, max_points=9000, seed=13)
    shell_plot = downsample(inflated_shell, max_points=14000, seed=29)
    fig = plt.figure(figsize=(8.6, 4.8))
    ax_occ = fig.add_subplot(121, projection="3d")
    ax_inflated = fig.add_subplot(122, projection="3d")

    ax_occ.scatter(
        occupied_plot[:, 0],
        occupied_plot[:, 1],
        occupied_plot[:, 2],
        c="#1f4e79",
        s=7.0,
        marker="s",
        alpha=0.82,
        linewidths=0,
        depthshade=False,
        label="原始占据体素",
    )
    ax_occ.set_title("占据体素")
    setup_3d_axis(ax_occ, occupied)

    ax_inflated.scatter(
        shell_plot[:, 0],
        shell_plot[:, 1],
        shell_plot[:, 2],
        c="#f28e2b",
        s=7.5,
        marker="s",
        alpha=0.16,
        linewidths=0,
        depthshade=False,
        label="膨胀后安全禁入体素",
    )
    ax_inflated.scatter(
        occupied_plot[:, 0],
        occupied_plot[:, 1],
        occupied_plot[:, 2],
        c="#1f4e79",
        s=7.0,
        marker="s",
        alpha=0.88,
        linewidths=0,
        depthshade=False,
        label="原始占据体素",
    )
    ax_inflated.set_title("膨胀后的搜索禁入空间")
    setup_3d_axis(ax_inflated, occupied)
    ax_inflated.legend(loc="upper right", frameon=True, fontsize=8)
    fig.suptitle("三维占据体素与障碍物膨胀", y=0.98)
    savefig(fig, out)


def plot_global_path(points: np.ndarray, rows: list[dict[str, str]], out: Path, title: str | None = None) -> None:
    occ, extent = occupancy_image(points)
    path = np.asarray([(float(r["x"]), float(r["y"]), float(r["z"])) for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.imshow(occ, origin="lower", extent=extent, cmap="Greys", interpolation="nearest", alpha=0.55)
    ax.plot(path[:, 0], path[:, 1], color="#d95f02", linewidth=2.2, label="A*航点路径")
    ax.scatter(path[0, 0], path[0, 1], marker="o", s=60, color="#1b9e77", label="起点")
    ax.scatter(path[-1, 0], path[-1, 1], marker="*", s=120, color="#7570b3", label="目标点")
    ax.set_title(title or "PCD占据地图上的全局航点结果")
    setup_axis(ax, "x / m", "y / m")
    ax.legend(loc="upper right", frameon=True)
    savefig(fig, out)


def path_rows_from_array(path: np.ndarray) -> list[dict[str, str]]:
    return [
        {"index": str(i), "x": f"{point[0]:.5f}", "y": f"{point[1]:.5f}", "z": f"{point[2]:.5f}"}
        for i, point in enumerate(path)
    ]


def plot_global_path_3d(points: np.ndarray, path: np.ndarray, out: Path, title: str) -> None:
    cloud = downsample(points, max_points=18000, seed=23)
    fig = plt.figure(figsize=(7.8, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        cloud[:, 0],
        cloud[:, 1],
        cloud[:, 2],
        c=cloud[:, 2],
        s=1.3,
        cmap="viridis",
        alpha=0.45,
        linewidths=0,
        depthshade=False,
    )
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#e4572e", linewidth=3.0, label="A*全局航点")
    ax.scatter(path[0, 0], path[0, 1], path[0, 2], marker="o", s=60, color="#1b9e77", label="起点")
    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], marker="*", s=120, color="#6a51a3", label="目标点")
    ax.set_title(title)
    setup_3d_axis(ax, points)
    fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, label="高度 / m")
    ax.legend(loc="upper right", frameon=True)
    savefig(fig, out)


def plot_local_target_bridge(
    path_rows: list[dict[str, str]], target_rows: list[dict[str, str]], out: Path, title: str
) -> None:
    path = np.asarray([(float(r["x"]), float(r["y"]), float(r["z"])) for r in path_rows], dtype=float)
    odom = np.asarray(
        [(float(r["odom_x"]), float(r["odom_y"]), float(r["odom_z"])) for r in target_rows], dtype=float
    )
    target = np.asarray(
        [(float(r["target_x"]), float(r["target_y"]), float(r["target_z"])) for r in target_rows], dtype=float
    )
    time_s = np.asarray([float(r["time_s"]) for r in target_rows], dtype=float)
    distance = np.asarray([float(r["target_distance_m"]) for r in target_rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True)
    axes[0].plot(time_s, odom[:, 0], color="#222222", linewidth=1.8, label="里程计 x")
    axes[0].plot(time_s, target[:, 0], color="#e45756", linewidth=1.8, label="目标 x")
    axes[0].fill_between(time_s, odom[:, 0], target[:, 0], color="#e45756", alpha=0.12)
    axes[0].hlines(
        [path[0, 0], path[-1, 0]],
        xmin=time_s.min(),
        xmax=time_s.max(),
        colors=["#1b9e77", "#7570b3"],
        linestyles=["--", ":"],
        linewidth=1.2,
        label="路径端点",
    )
    axes[0].set_ylabel("x / m")
    axes[0].set_title(title)
    axes[0].grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axes[0].legend(loc="best", ncol=3, fontsize=8)

    axes[1].plot(time_s, odom[:, 1], color="#222222", linewidth=1.8, label="里程计 y")
    axes[1].plot(time_s, target[:, 1], color="#54a24b", linewidth=1.8, label="目标 y")
    axes[1].fill_between(time_s, odom[:, 1], target[:, 1], color="#54a24b", alpha=0.12)
    axes[1].hlines(
        [path[0, 1], path[-1, 1]],
        xmin=time_s.min(),
        xmax=time_s.max(),
        colors=["#1b9e77", "#7570b3"],
        linestyles=["--", ":"],
        linewidth=1.2,
        label="路径端点",
    )
    axes[1].set_ylabel("y / m")
    axes[1].grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axes[1].legend(loc="best", ncol=3, fontsize=8)

    axes[2].plot(time_s, distance, color="#4c78a8", linewidth=1.8, label="目标距离")
    axes[2].axhline(np.median(distance), color="#f58518", linestyle="--", linewidth=1.3, label="距离中位数")
    axes[2].set_xlabel("时间 / s")
    axes[2].set_ylabel("距离 / m")
    axes[2].grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axes[2].legend(loc="best", fontsize=8)
    savefig(fig, out)


def plot_local_target(rows: list[dict[str, str]], out: Path) -> None:
    data = {k: np.asarray([float(r[k]) for r in rows], dtype=float) for k in rows[0]}
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.0), sharex=True)
    for ax, coord in zip(axes, "xyz"):
        ax.plot(data["time_s"], data[f"odom_{coord}"], color="#4c78a8", linewidth=1.8, label=f"里程计 {coord}")
        ax.plot(data["time_s"], data[f"target_{coord}"], color="#f58518", linewidth=1.8, label=f"目标 {coord}")
        ax.set_ylabel(f"{coord} / m")
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        ax.legend(loc="best", ncol=2, fontsize=8)
    axes[-1].set_xlabel("时间 / s")
    axes[0].set_title("桥接节点生成的局部前视目标")
    savefig(fig, out)


def pick_longest(root: Path) -> Path | None:
    candidates = sorted(root.glob("*/traj_history.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: len(load_traj(p)))


def plot_eval_trajectories(repo: Path, out: Path) -> None:
    items: list[tuple[str, Path]] = []
    explicit = [
        ("深度相机 0.5 m/s", repo / "sim2sim/drone_genesis/depth_camera/nav/exps_0.5/20260422_222639_ep00/traj_history.json"),
        ("深度相机 1.5 m/s", repo / "sim2sim/drone_genesis/depth_camera/nav/exps_1.5/20260422_222344_ep00/traj_history.json"),
    ]
    for label, path in explicit:
        if path.exists():
            items.append((label, path))
    lidar = pick_longest(repo / "sim2sim/drone_genesis/lidar_navrl/exps_0.5")
    if lidar is not None:
        items.append(("激光雷达 0.5 m/s", lidar))
    if not items:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for (label, path), color in zip(items, colors):
        traj = load_traj(path)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2.0, color=color, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], s=36, color=color, marker="o")
        ax.scatter(traj[-1, 0], traj[-1, 1], s=48, color=color, marker="x")
    ax.scatter([-1.5], [0.0], marker="o", s=70, color="#222222", label="名义起点")
    ax.scatter([5.0], [0.0], marker="*", s=140, color="#222222", label="名义目标点")
    ax.set_title("Genesis评估轨迹")
    setup_axis(ax, "x / m", "y / m")
    ax.legend(loc="best", fontsize=8)
    savefig(fig, out)


def plot_hover_boxplot(repo: Path, out: Path) -> None:
    groups: list[tuple[str, list[float]]] = []
    goal = np.asarray([5.0, 0.0, 1.0], dtype=float)
    roots = [
        ("深度相机0.5", repo / "sim2sim/drone_genesis/depth_camera/nav/exps_0.5"),
        ("深度相机1.5", repo / "sim2sim/drone_genesis/depth_camera/nav/exps_1.5"),
        ("激光雷达0.5", repo / "sim2sim/drone_genesis/lidar_navrl/exps_0.5"),
    ]
    for label, root in roots:
        values = []
        for path in sorted(root.glob("*/traj_history.json")):
            traj = load_traj(path)
            if len(traj) < 20:
                continue
            tail = traj[max(0, int(len(traj) * 0.9)) :]
            values.extend(np.linalg.norm(tail - goal, axis=1).tolist())
        if values:
            groups.append((label, values))
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.boxplot([v for _, v in groups], labels=[k for k, _ in groups], patch_artist=True)
    ax.set_ylabel("位置误差 / m")
    ax.set_title("轨迹日志中的悬停/终端位置误差")
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    savefig(fig, out)


def plot_pointlio(rows: list[dict[str, str]], out: Path) -> None:
    valid = [r for r in rows if r.get("gt_x")]
    if len(valid) < 2:
        return
    est = np.asarray([(float(r["est_x"]), float(r["est_y"])) for r in valid])
    gt = np.asarray([(float(r["gt_x"]), float(r["gt_y"])) for r in valid])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(gt[:, 0], gt[:, 1], color="#4c78a8", linewidth=2.0, label="真值")
    ax.plot(est[:, 0], est[:, 1], color="#e45756", linewidth=1.8, linestyle="--", label="Point-LIO")
    ax.set_title("Point-LIO估计轨迹与真值")
    setup_axis(ax, "x / m", "y / m")
    ax.legend(loc="best")
    savefig(fig, out)


def plot_gnss(rows: list[dict[str, str]], out: Path) -> None:
    if len(rows) < 2:
        return
    selected = np.asarray([(float(r["selected_x"]), float(r["selected_y"])) for r in rows])
    slam = np.asarray([(float(r["slam_x"]), float(r["slam_y"])) for r in rows])
    gnss_rows = [r for r in rows if r.get("gnss_x")]
    gnss = np.asarray([(float(r["gnss_x"]), float(r["gnss_y"])) for r in gnss_rows]) if gnss_rows else None
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(slam[:, 0], slam[:, 1], color="#4c78a8", linewidth=2.0, label="SLAM参考")
    if gnss is not None and len(gnss) > 1:
        ax.plot(gnss[:, 0], gnss[:, 1], color="#e45756", linewidth=1.4, linestyle="--", label="GNSS")
    ax.plot(selected[:, 0], selected[:, 1], color="#54a24b", linewidth=2.0, label="融合输出")
    ax.set_title("GNSS/SLAM定位模式轨迹")
    setup_axis(ax, "x / m", "y / m")
    ax.legend(loc="best")
    savefig(fig, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--figures-dir", type=Path, default=Path("/home/ember/桌面/thesis/thesis-latex/Figures"))
    parser.add_argument("--work-dir", type=Path, default=Path("thesis_outputs/real_figures"))
    parser.add_argument("--raw-pcd", type=Path, default=Path("thesis_outputs/real_figures/data/pipe_factory_raw_static.pcd"))
    parser.add_argument("--filtered-pcd", type=Path, default=Path("sim2sim/pipeline_inspection/assets/maps/pipe_factory_local.pcd"))
    parser.add_argument("--pointlio-map-pcd", type=Path, default=Path("thesis_outputs/real_figures/point_lio_smoke/pipe_factory_pointlio_map.pcd"))
    parser.add_argument("--ros-capture-dir", type=Path, default=Path("thesis_outputs/real_figures/global_path_capture"))
    parser.add_argument("--pointlio-dir", type=Path, default=Path("thesis_outputs/real_figures/point_lio_metrics"))
    parser.add_argument("--gnss-dir", type=Path, default=Path("thesis_outputs/real_figures/gnss_mode_test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    figures = args.figures_dir.resolve()
    raw_pcd = (repo / args.raw_pcd).resolve()
    filtered_pcd = (repo / args.filtered_pcd).resolve()
    pointlio_map_pcd = (repo / args.pointlio_map_pcd).resolve()
    capture_dir = (repo / args.ros_capture_dir).resolve()

    raw = load_pcd_xyz(raw_pcd) if raw_pcd.exists() else None
    filtered = load_pcd_xyz(filtered_pcd)
    if raw is not None:
        plot_raw_cloud_stage(
            raw,
            figures / "ch2_raw_point_cloud_real.png",
            "原始PCD采样点云",
            max_points=34000,
            seed=5,
        )
    else:
        plot_raw_cloud_stage(
            filtered,
            figures / "ch2_raw_point_cloud_real.png",
            "原始PCD采样点云",
            max_points=22000,
            seed=5,
        )

    plot_cloud_overview(
        filtered,
        figures / "ch2_point_cloud_mapping_real.png",
        "管道工厂点云建图结果",
        max_points=20000,
        seed=3,
    )
    if raw is not None:
        plot_filtered_cloud_comparison(raw, filtered, figures / "ch2_filtered_point_cloud_real.png")
    else:
        plot_cloud_overview(
            filtered,
            figures / "ch2_filtered_point_cloud_real.png",
            "滤波后点云地图",
            max_points=12000,
            seed=11,
        )
    pointlio_map = load_pcd_xyz(pointlio_map_pcd) if pointlio_map_pcd.exists() else filtered
    plot_cloud_overview(
        pointlio_map,
        figures / "ch5_pointlio_map_real.png",
        "PX4/Gazebo运行的Point-LIO点云地图",
        xlim=(-60.0, 60.0),
        ylim=(-30.0, 30.0),
    )
    plot_occupancy(filtered, figures / "ch2_occupancy_grid_real.png")

    path_rows = load_csv(capture_dir / "global_path_points.csv")
    if path_rows:
        plot_global_path(filtered, path_rows, figures / "ch3_global_waypoint_result_real.png")
    target_rows = load_csv(capture_dir / "local_target_samples.csv")
    if target_rows:
        plot_local_target(target_rows, figures / "ch3_local_target_curve_real.png")
    if path_rows and target_rows:
        plot_local_target_bridge(
            path_rows,
            target_rows,
            figures / "ch3_local_target_bridge_real.png",
            "由记录全局航点选择的前视目标",
        )

    demo_path = demo_astar_path(filtered)
    demo_path_rows = path_rows_from_array(demo_path)
    demo_target_rows = local_target_samples_from_path(demo_path)
    plot_global_path_3d(filtered, demo_path, figures / "ch5_global_waypoints_real.png", "三维A*全局航点规划结果")
    plot_local_target_bridge(
        demo_path_rows,
        demo_target_rows,
        figures / "ch5_local_target_bridge_real.png",
        "绕障路径上的局部目标桥接输出",
    )

    plot_eval_trajectories(repo, figures / "ch4_real_trajectory_logs.png")
    plot_hover_boxplot(repo, figures / "ch5_hover_boxplot_real.png")

    pointlio_rows = load_csv((repo / args.pointlio_dir) / "point_lio_samples.csv")
    if pointlio_rows:
        plot_pointlio(pointlio_rows, figures / "ch5_pointlio_traj_real.png")

    gnss_rows = load_csv((repo / args.gnss_dir) / "gnss_mode_samples.csv")
    if gnss_rows:
        plot_gnss(gnss_rows, figures / "ch5_gnss_traj_real.png")


if __name__ == "__main__":
    main()
