from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Any

from airsim.types import Pose, Quaternionr, Vector3r


PIPE_PREFIX = "pipe_swarm_obstacle"
PIPE_ASSET = "Cylinder"

# These four cubes form the central wall and pass-through hole in the
# original swarm scene. The pipe scene removes them and replaces the
# opening with a lattice of cylindrical obstacles.
CENTRAL_WALL_OBJECTS = (
    "1M_Cube_Chamfer4_9",
    "1M_Cube_Chamfer5",
    "1M_Cube_Chamfer6",
    "1M_Cube_Chamfer22",
)


@dataclass(frozen=True)
class PipeSpec:
    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    diameter: float


PIPE_SPECS = (
    PipeSpec("cross_y_upper", (3.05, -2.35, -0.62), (3.05, 2.35, -0.62), 0.18),
    PipeSpec("cross_y_lower", (2.95, -2.35, 0.58), (2.95, 2.35, 0.58), 0.18),
    PipeSpec("cross_y_mid_offset", (3.38, -2.10, -0.23), (3.38, 2.10, -0.23), 0.14),
    PipeSpec("rail_x_left", (1.85, -1.35, -0.08), (4.25, -1.35, -0.08), 0.16),
    PipeSpec("rail_x_right", (1.85, 1.35, 0.12), (4.25, 1.35, 0.12), 0.16),
    PipeSpec("vertical_front_left", (2.30, -0.82, -0.90), (2.30, -0.82, 0.90), 0.14),
    PipeSpec("vertical_front_right", (2.30, 0.82, -0.90), (2.30, 0.82, 0.90), 0.14),
    PipeSpec("vertical_back_left", (3.85, -0.82, -0.90), (3.85, -0.82, 0.90), 0.14),
    PipeSpec("vertical_back_right", (3.85, 0.82, -0.90), (3.85, 0.82, 0.90), 0.14),
    PipeSpec("diagonal_a", (2.05, -1.80, -0.73), (4.05, 1.80, 0.23), 0.12),
    PipeSpec("diagonal_b", (2.05, 1.80, -0.67), (4.05, -1.80, 0.27), 0.12),
)


def _vec(values: tuple[float, float, float]) -> Vector3r:
    return Vector3r(float(values[0]), float(values[1]), float(values[2]))


def _unit_direction(start: tuple[float, float, float], end: tuple[float, float, float]) -> tuple[float, float, float, float]:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    dz = float(end[2] - start[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-6:
        raise ValueError(f"pipe endpoints are too close: {start} -> {end}")
    return dx / length, dy / length, dz / length, length


def _quat_from_z_axis(direction: tuple[float, float, float]) -> Quaternionr:
    """Rotate Unreal's default cylinder axis (+Z) onto the requested direction."""
    dx, dy, dz = direction
    dot = max(-1.0, min(1.0, dz))
    if dot > 1.0 - 1e-6:
        return Quaternionr(0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + 1e-6:
        return Quaternionr(1.0, 0.0, 0.0, 0.0)

    cx = -dy
    cy = dx
    cz = 0.0
    s = math.sqrt((1.0 + dot) * 2.0)
    inv_s = 1.0 / s
    return Quaternionr(cx * inv_s, cy * inv_s, cz * inv_s, s * 0.5)


def _pipe_pose_and_scale(spec: PipeSpec) -> tuple[Pose, Vector3r, float]:
    dx, dy, dz, length = _unit_direction(spec.start, spec.end)
    center = tuple((a + b) * 0.5 for a, b in zip(spec.start, spec.end))
    pose = Pose(_vec(center), _quat_from_z_axis((dx, dy, dz)))
    scale = Vector3r(spec.diameter, spec.diameter, length)
    return pose, scale, length


def setup_pipe_scene(client: Any, log_dir: str | None = None) -> dict[str, Any]:
    scene_result: dict[str, Any] = {
        "asset": PIPE_ASSET,
        "pipe_prefix": PIPE_PREFIX,
        "removed_wall_objects": [],
        "removed_previous_pipes": [],
        "pipes": [],
    }

    for name in client.simListSceneObjects(f"{PIPE_PREFIX}.*"):
        if client.simDestroyObject(name):
            scene_result["removed_previous_pipes"].append(name)

    for name in CENTRAL_WALL_OBJECTS:
        if client.simDestroyObject(name):
            scene_result["removed_wall_objects"].append(name)

    for spec in PIPE_SPECS:
        object_name = f"{PIPE_PREFIX}_{spec.name}"
        pose, scale, length = _pipe_pose_and_scale(spec)
        spawned_name = client.simSpawnObject(object_name, PIPE_ASSET, pose, scale, False)
        if not spawned_name:
            raise RuntimeError(f"failed to spawn pipe obstacle: {object_name}")
        scene_result["pipes"].append({
            **asdict(spec),
            "length": length,
            "spawned_name": spawned_name,
            "scale": [scale.x_val, scale.y_val, scale.z_val],
        })

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "pipe_scene.json"), "w") as f:
            json.dump(scene_result, f, indent=2, sort_keys=True)

    return scene_result
