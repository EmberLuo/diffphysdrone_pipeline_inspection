#!/usr/bin/env python3
"""Run an offline global-planning comparison on the pipe-factory PCD map.

The ROS runtime uses the C++ global A* node.  This script provides the
repeatable batch layer needed for thesis tables: it loads the same PCD map,
builds an inflated voxel occupancy grid, and compares inflated A*, uninflated
A*, and RRT* over several start-goal pairs.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from thesis_plot_style import setup_chinese_matplotlib


@dataclass(frozen=True)
class Task:
    name: str
    start: tuple[float, float, float]
    goal: tuple[float, float, float]


@dataclass
class PlanResult:
    method: str
    task: Task
    success: bool
    planning_time_s: float
    path: np.ndarray
    expanded_nodes: int
    adjusted_start: tuple[float, float, float] | None = None
    adjusted_goal: tuple[float, float, float] | None = None


class VoxelMap:
    def __init__(
        self,
        points: np.ndarray,
        resolution: float,
        margin_xy: float,
        flight_z_min: float,
        flight_z_max: float,
        inflation_radius: float,
    ) -> None:
        self.points = points
        self.resolution = resolution
        min_xy = points[:, :2].min(axis=0) - margin_xy
        max_xy = points[:, :2].max(axis=0) + margin_xy
        self.origin = np.asarray([min_xy[0], min_xy[1], flight_z_min], dtype=float)
        self.max_bound = np.asarray([max_xy[0], max_xy[1], flight_z_max], dtype=float)
        shape_xyz = np.ceil((self.max_bound - self.origin) / resolution).astype(int) + 1
        self.shape = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))

        self.occupied = np.zeros(self.shape, dtype=bool)
        cells = self.world_to_cell(points)
        valid = self.valid_cells(cells)
        cells = cells[valid]
        self.occupied[cells[:, 2], cells[:, 1], cells[:, 0]] = True

        radius_cells = max(0, int(math.ceil(inflation_radius / resolution)))
        if radius_cells > 0:
            ranges = np.arange(-radius_cells, radius_cells + 1)
            dz, dy, dx = np.meshgrid(ranges, ranges, ranges, indexing="ij")
            structure = (dx * dx + dy * dy + dz * dz) <= radius_cells * radius_cells
            self.inflated = binary_dilation(self.occupied, structure=structure)
        else:
            self.inflated = self.occupied.copy()

        self.clearance_tree = cKDTree(points)

    def world_to_cell(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        return np.floor((pts - self.origin[None, :]) / self.resolution).astype(int)

    def cell_to_world(self, cell: tuple[int, int, int] | np.ndarray) -> np.ndarray:
        arr = np.asarray(cell, dtype=float)
        return self.origin + (arr + 0.5) * self.resolution

    def valid_cells(self, cells: np.ndarray) -> np.ndarray:
        return (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < self.shape[2])
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < self.shape[1])
            & (cells[:, 2] >= 0)
            & (cells[:, 2] < self.shape[0])
        )

    def is_free_cell(self, cell: tuple[int, int, int], occupied: np.ndarray) -> bool:
        x, y, z = cell
        if x < 0 or y < 0 or z < 0 or x >= self.shape[2] or y >= self.shape[1] or z >= self.shape[0]:
            return False
        return not bool(occupied[z, y, x])

    def nearest_free_cell(
        self,
        point: tuple[float, float, float],
        occupied: np.ndarray,
        max_radius: int = 24,
    ) -> tuple[int, int, int]:
        base = tuple(int(v) for v in self.world_to_cell(np.asarray(point))[0])
        if self.is_free_cell(base, occupied):
            return base
        bx, by, bz = base
        best: tuple[int, int, int] | None = None
        best_d2 = float("inf")
        for radius in range(1, max_radius + 1):
            for z in range(max(0, bz - radius), min(self.shape[0], bz + radius + 1)):
                for y in range(max(0, by - radius), min(self.shape[1], by + radius + 1)):
                    for x in range(max(0, bx - radius), min(self.shape[2], bx + radius + 1)):
                        if occupied[z, y, x]:
                            continue
                        d2 = (x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2
                        if d2 < best_d2:
                            best = (x, y, z)
                            best_d2 = d2
            if best is not None:
                return best
        raise RuntimeError(f"no free cell found near {point}")

    def edge_is_free(self, a: tuple[int, int, int], b: tuple[int, int, int], occupied: np.ndarray) -> bool:
        av = np.asarray(a, dtype=float)
        bv = np.asarray(b, dtype=float)
        steps = int(np.ceil(np.linalg.norm(bv - av))) + 1
        for i in range(steps + 1):
            t = i / max(1, steps)
            cell = tuple(int(v) for v in np.rint(av + (bv - av) * t))
            if not self.is_free_cell(cell, occupied):
                return False
        return True


NEIGHBORS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def load_pcd_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as fp:
        header: list[str] = []
        while True:
            line = fp.readline()
            if not line:
                raise ValueError(f"PCD header is incomplete: {path}")
            text = line.decode("utf-8", errors="ignore").strip()
            header.append(text)
            if text.startswith("DATA"):
                data_kind = text.split()[1].lower()
                break
        meta: dict[str, list[str]] = {}
        for line in header:
            parts = line.split()
            if parts:
                meta[parts[0].upper()] = parts[1:]
        fields = meta.get("FIELDS", [])
        if not {"x", "y", "z"}.issubset(fields):
            raise ValueError(f"PCD must contain x/y/z fields: {path}")
        points = int((meta.get("POINTS") or meta.get("WIDTH") or ["0"])[0])
        if data_kind != "ascii":
            raise ValueError("This thesis export script expects the prepared ASCII PCD map")
        arr = np.loadtxt(fp, dtype=float, max_rows=points)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, [fields.index("x"), fields.index("y"), fields.index("z")]]


def astar_plan(method: str, task: Task, grid: VoxelMap, occupied: np.ndarray) -> PlanResult:
    started = time.perf_counter()
    start = grid.nearest_free_cell(task.start, occupied)
    goal = grid.nearest_free_cell(task.goal, occupied)
    open_heap: list[tuple[float, int, tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0, start))
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    cost: dict[tuple[int, int, int], float] = {start: 0.0}
    expanded = 0
    push_count = 1

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        expanded += 1
        if current == goal:
            break
        cx, cy, cz = current
        for dx, dy, dz in NEIGHBORS_26:
            nxt = (cx + dx, cy + dy, cz + dz)
            if not grid.is_free_cell(nxt, occupied):
                continue
            step_cost = math.sqrt(dx * dx + dy * dy + dz * dz)
            new_cost = cost[current] + step_cost
            if new_cost >= cost.get(nxt, float("inf")):
                continue
            cost[nxt] = new_cost
            h = math.dist(nxt, goal)
            came_from[nxt] = current
            heapq.heappush(open_heap, (new_cost + h, push_count, nxt))
            push_count += 1

    elapsed = time.perf_counter() - started
    if goal not in cost:
        return PlanResult(method, task, False, elapsed, np.empty((0, 3)), expanded)

    cells = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        cells.append(cur)
    path = np.asarray([grid.cell_to_world(cell) for cell in reversed(cells)], dtype=float)
    return PlanResult(
        method=method,
        task=task,
        success=True,
        planning_time_s=elapsed,
        path=simplify_path(path),
        expanded_nodes=expanded,
        adjusted_start=tuple(grid.cell_to_world(start).tolist()),
        adjusted_goal=tuple(grid.cell_to_world(goal).tolist()),
    )


def rrt_star_plan(task: Task, grid: VoxelMap, occupied: np.ndarray, seed: int) -> PlanResult:
    rng = random.Random(seed)
    started = time.perf_counter()
    start = grid.nearest_free_cell(task.start, occupied)
    goal = grid.nearest_free_cell(task.goal, occupied)
    free_cells = np.argwhere(~occupied)
    free_xyz = np.column_stack([free_cells[:, 2], free_cells[:, 1], free_cells[:, 0]])

    nodes: list[tuple[int, int, int]] = [start]
    parent: list[int] = [-1]
    costs: list[float] = [0.0]
    max_iter = 9000
    step_cells = 5.0
    near_radius = 9.0
    goal_index: int | None = None

    for it in range(max_iter):
        expanded = it + 1
        if rng.random() < 0.10:
            sample = goal
        else:
            row = free_xyz[rng.randrange(len(free_xyz))]
            sample = (int(row[0]), int(row[1]), int(row[2]))

        node_arr = np.asarray(nodes, dtype=float)
        sample_arr = np.asarray(sample, dtype=float)
        nearest_idx = int(np.argmin(np.linalg.norm(node_arr - sample_arr[None, :], axis=1)))
        nearest = np.asarray(nodes[nearest_idx], dtype=float)
        direction = sample_arr - nearest
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        new_arr = nearest + direction / norm * min(step_cells, norm)
        new = tuple(int(v) for v in np.rint(new_arr))
        if not grid.is_free_cell(new, occupied) or not grid.edge_is_free(nodes[nearest_idx], new, occupied):
            continue

        near_indices = np.where(np.linalg.norm(node_arr - np.asarray(new)[None, :], axis=1) <= near_radius)[0]
        best_parent = nearest_idx
        best_cost = costs[nearest_idx] + math.dist(nodes[nearest_idx], new)
        for idx in near_indices:
            candidate = nodes[int(idx)]
            if not grid.edge_is_free(candidate, new, occupied):
                continue
            candidate_cost = costs[int(idx)] + math.dist(candidate, new)
            if candidate_cost < best_cost:
                best_parent = int(idx)
                best_cost = candidate_cost

        nodes.append(new)
        parent.append(best_parent)
        costs.append(best_cost)
        new_idx = len(nodes) - 1

        for idx in near_indices:
            idx = int(idx)
            rewired_cost = best_cost + math.dist(new, nodes[idx])
            if rewired_cost < costs[idx] and grid.edge_is_free(new, nodes[idx], occupied):
                parent[idx] = new_idx
                costs[idx] = rewired_cost

        if math.dist(new, goal) <= step_cells and grid.edge_is_free(new, goal, occupied):
            nodes.append(goal)
            parent.append(new_idx)
            costs.append(best_cost + math.dist(new, goal))
            goal_index = len(nodes) - 1
            break
    else:
        expanded = max_iter

    elapsed = time.perf_counter() - started
    if goal_index is None:
        return PlanResult("RRT*", task, False, elapsed, np.empty((0, 3)), expanded)

    cells = []
    cur = goal_index
    while cur >= 0:
        cells.append(nodes[cur])
        cur = parent[cur]
    path = np.asarray([grid.cell_to_world(cell) for cell in reversed(cells)], dtype=float)
    return PlanResult(
        method="RRT*",
        task=task,
        success=True,
        planning_time_s=elapsed,
        path=simplify_path(path),
        expanded_nodes=expanded,
        adjusted_start=tuple(grid.cell_to_world(start).tolist()),
        adjusted_goal=tuple(grid.cell_to_world(goal).tolist()),
    )


def simplify_path(path: np.ndarray) -> np.ndarray:
    if len(path) <= 2:
        return path
    keep = [0]
    prev = np.sign(path[1] - path[0]).astype(int)
    for idx in range(2, len(path)):
        cur = np.sign(path[idx] - path[idx - 1]).astype(int)
        if not np.array_equal(cur, prev):
            keep.append(idx - 1)
            prev = cur
    keep.append(len(path) - 1)
    return path[keep]


def resample_path(path: np.ndarray, step: float = 0.25) -> np.ndarray:
    if len(path) < 2:
        return path
    samples = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        dist = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(dist / step)))
        for i in range(1, n + 1):
            samples.append(a + (b - a) * (i / n))
    return np.asarray(samples, dtype=float)


def path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def min_clearance(path: np.ndarray, tree: cKDTree) -> float:
    if len(path) == 0:
        return float("nan")
    samples = resample_path(path, step=0.25)
    dists, _ = tree.query(samples)
    return float(np.min(dists))


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def default_tasks(z: float) -> list[Task]:
    return [
        Task("T1_cross_corridor", (-28.0, -12.0, z), (28.0, 12.0, z)),
        Task("T2_reverse_cross", (-25.0, 13.0, z), (25.0, -13.0, z)),
        Task("T3_long_axis", (-30.0, -2.0, z), (30.0, 2.0, z)),
        Task("T4_diagonal", (-18.0, -15.0, z), (18.0, 15.0, z)),
        Task("T5_short_dense", (-8.0, 14.0, z), (26.0, -6.0, z)),
    ]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_comparison(grid: VoxelMap, results: list[PlanResult], figures_dir: Path) -> None:
    setup_chinese_matplotlib()
    task_name = "T1_cross_corridor"
    selected = [r for r in results if r.task.name == task_name and r.success]
    if not selected:
        return
    points = grid.points
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(points[:, 0], points[:, 1], s=0.6, color="#8a8a8a", alpha=0.25, linewidths=0, label="PCD点云投影")
    colors = {"A*+膨胀": "#d95f02", "A*无膨胀": "#4c78a8", "RRT*": "#54a24b"}
    for result in selected:
        ax.plot(
            result.path[:, 0],
            result.path[:, 1],
            linewidth=2.2,
            color=colors.get(result.method, "#222222"),
            label=result.method,
        )
    start = selected[0].path[0]
    goal = selected[0].path[-1]
    ax.scatter(start[0], start[1], s=55, marker="o", color="#1b9e77", label="起点")
    ax.scatter(goal[0], goal[1], s=120, marker="*", color="#6a51a3", label="目标点")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("全局规划算法路径对比")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "ch4_global_planning_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_paths(results: list[PlanResult], output_dir: Path) -> None:
    rows = []
    for result in results:
        if not result.success:
            continue
        for idx, point in enumerate(result.path):
            rows.append(
                {
                    "task": result.task.name,
                    "method": result.method,
                    "index": idx,
                    "x": f"{point[0]:.5f}",
                    "y": f"{point[1]:.5f}",
                    "z": f"{point[2]:.5f}",
                }
            )
    write_csv(output_dir / "global_planning_paths.csv", rows, ["task", "method", "index", "x", "y", "z"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pcd", type=Path, default=Path("sim2sim/pipeline_inspection/assets/maps/pipe_factory_local.pcd"))
    parser.add_argument("--output-dir", type=Path, default=Path("thesis_outputs/global_planning_comparison"))
    parser.add_argument("--figures-dir", type=Path, default=Path("/home/ember/桌面/thesis/thesis-latex/Figures"))
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--inflation-radius", type=float, default=0.45)
    parser.add_argument("--margin-xy", type=float, default=2.0)
    parser.add_argument("--flight-z", type=float, default=1.8)
    parser.add_argument("--flight-z-min", type=float, default=1.0)
    parser.add_argument("--flight-z-max", type=float, default=2.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    output_dir = (repo / args.output_dir).resolve()
    figures_dir = args.figures_dir.resolve()
    pcd = (repo / args.pcd).resolve()

    points = load_pcd_xyz(pcd)
    grid = VoxelMap(
        points=points,
        resolution=args.resolution,
        margin_xy=args.margin_xy,
        flight_z_min=args.flight_z_min,
        flight_z_max=args.flight_z_max,
        inflation_radius=args.inflation_radius,
    )

    tasks = default_tasks(args.flight_z)
    results: list[PlanResult] = []
    for task_idx, task in enumerate(tasks):
        results.append(astar_plan("A*+膨胀", task, grid, grid.inflated))
        results.append(astar_plan("A*无膨胀", task, grid, grid.occupied))
        results.append(rrt_star_plan(task, grid, grid.inflated, seed=2026 + task_idx))

    detail_rows: list[dict[str, object]] = []
    for result in results:
        length = path_length(result.path) if result.success else float("nan")
        clearance = min_clearance(result.path, grid.clearance_tree) if result.success else float("nan")
        detail_rows.append(
            {
                "task": result.task.name,
                "method": result.method,
                "success": int(result.success),
                "planning_time_ms": f"{result.planning_time_s * 1000:.3f}",
                "path_length_m": "" if not math.isfinite(length) else f"{length:.3f}",
                "min_clearance_m": "" if not math.isfinite(clearance) else f"{clearance:.3f}",
                "waypoint_count": len(result.path) if result.success else "",
                "expanded_nodes": result.expanded_nodes,
                "start_x": f"{result.path[0, 0]:.3f}" if result.success else "",
                "start_y": f"{result.path[0, 1]:.3f}" if result.success else "",
                "start_z": f"{result.path[0, 2]:.3f}" if result.success else "",
                "goal_x": f"{result.path[-1, 0]:.3f}" if result.success else "",
                "goal_y": f"{result.path[-1, 1]:.3f}" if result.success else "",
                "goal_z": f"{result.path[-1, 2]:.3f}" if result.success else "",
            }
        )

    detail_fields = [
        "task",
        "method",
        "success",
        "planning_time_ms",
        "path_length_m",
        "min_clearance_m",
        "waypoint_count",
        "expanded_nodes",
        "start_x",
        "start_y",
        "start_z",
        "goal_x",
        "goal_y",
        "goal_z",
    ]
    write_csv(output_dir / "global_planning_comparison_detail.csv", detail_rows, detail_fields)

    summary_rows = []
    for method in ["A*+膨胀", "A*无膨胀", "RRT*"]:
        rows = [r for r in detail_rows if r["method"] == method]
        successes = [r for r in rows if int(r["success"]) == 1]
        time_mean, time_std = mean_std(float(r["planning_time_ms"]) for r in successes)
        length_mean, length_std = mean_std(float(r["path_length_m"]) for r in successes)
        clearance_mean, clearance_std = mean_std(float(r["min_clearance_m"]) for r in successes)
        expanded_mean, expanded_std = mean_std(float(r["expanded_nodes"]) for r in successes)
        waypoint_mean, waypoint_std = mean_std(float(r["waypoint_count"]) for r in successes)
        summary_rows.append(
            {
                "method": method,
                "success_rate": f"{100.0 * len(successes) / max(1, len(rows)):.1f}",
                "planning_time_ms_mean": f"{time_mean:.3f}",
                "planning_time_ms_std": f"{time_std:.3f}",
                "path_length_m_mean": f"{length_mean:.3f}",
                "path_length_m_std": f"{length_std:.3f}",
                "min_clearance_m_mean": f"{clearance_mean:.3f}",
                "min_clearance_m_std": f"{clearance_std:.3f}",
                "waypoint_count_mean": f"{waypoint_mean:.1f}",
                "waypoint_count_std": f"{waypoint_std:.1f}",
                "expanded_nodes_mean": f"{expanded_mean:.1f}",
                "expanded_nodes_std": f"{expanded_std:.1f}",
            }
        )
    summary_fields = [
        "method",
        "success_rate",
        "planning_time_ms_mean",
        "planning_time_ms_std",
        "path_length_m_mean",
        "path_length_m_std",
        "min_clearance_m_mean",
        "min_clearance_m_std",
        "waypoint_count_mean",
        "waypoint_count_std",
        "expanded_nodes_mean",
        "expanded_nodes_std",
    ]
    write_csv(output_dir / "global_planning_comparison_summary.csv", summary_rows, summary_fields)
    export_paths(results, output_dir)
    plot_comparison(grid, results, figures_dir)

    print(f"wrote {output_dir / 'global_planning_comparison_detail.csv'}")
    print(f"wrote {output_dir / 'global_planning_comparison_summary.csv'}")
    print(f"wrote {figures_dir / 'ch4_global_planning_comparison.png'}")


if __name__ == "__main__":
    main()
