#!/usr/bin/env python3
"""Sample the local pipe factory SDF into an ASCII PCD map."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


SKIP_LINK_PREFIXES = (
    "concrete_pad",
    "center_safety_lane",
    "lane_line_",
)


def parse_pose(text: str | None) -> tuple[float, float, float, float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    values = [float(v) for v in text.split()]
    values += [0.0] * (6 - len(values))
    return tuple(values[:6])  # type: ignore[return-value]


def rotation_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def transform_point(
    point: tuple[float, float, float],
    pose: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float]:
    x, y, z, roll, pitch, yaw = pose
    rot = rotation_matrix(roll, pitch, yaw)
    px, py, pz = point
    return (
        x + rot[0][0] * px + rot[0][1] * py + rot[0][2] * pz,
        y + rot[1][0] * px + rot[1][1] * py + rot[1][2] * pz,
        z + rot[2][0] * px + rot[2][1] * py + rot[2][2] * pz,
    )


def sample_range(length: float, spacing: float) -> list[float]:
    steps = max(1, int(math.ceil(length / spacing)))
    if steps == 1:
        return [0.0]
    start = -0.5 * length
    return [start + length * i / steps for i in range(steps + 1)]


def sample_box(size: tuple[float, float, float], spacing: float) -> list[tuple[float, float, float]]:
    sx, sy, sz = size
    xs = sample_range(sx, spacing)
    ys = sample_range(sy, spacing)
    zs = sample_range(sz, spacing)
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    points: list[tuple[float, float, float]] = []

    for x in (-hx, hx):
        points.extend((x, y, z) for y in ys for z in zs)
    for y in (-hy, hy):
        points.extend((x, y, z) for x in xs for z in zs)
    for z in (-hz, hz):
        points.extend((x, y, z) for x in xs for y in ys)
    return points


def sample_cylinder(radius: float, length: float, spacing: float) -> list[tuple[float, float, float]]:
    zs = sample_range(length, spacing)
    ntheta = max(12, int(math.ceil((2.0 * math.pi * radius) / spacing)))
    points: list[tuple[float, float, float]] = []

    for z in zs:
        for i in range(ntheta):
            theta = 2.0 * math.pi * i / ntheta
            points.append((radius * math.cos(theta), radius * math.sin(theta), z))

    if radius >= 0.5:
        rings = max(2, int(math.ceil(radius / spacing)))
        for z in (-0.5 * length, 0.5 * length):
            for r_i in range(rings + 1):
                r = radius * r_i / rings
                ring_points = max(8, int(math.ceil((2.0 * math.pi * max(r, spacing)) / spacing)))
                for i in range(ring_points):
                    theta = 2.0 * math.pi * i / ring_points
                    points.append((r * math.cos(theta), r * math.sin(theta), z))
    return points


def extract_points(world_path: Path, spacing: float) -> list[tuple[float, float, float]]:
    root = ET.parse(world_path).getroot()
    points: list[tuple[float, float, float]] = []

    for model in root.findall(".//model"):
        model_pose = parse_pose(model.findtext("pose"))
        if any(abs(v) > 1e-9 for v in model_pose[3:]):
            raise ValueError("Rotated model poses are not supported by this lightweight sampler")
        model_offset = model_pose[:3]

        for link in model.findall("link"):
            name = link.attrib.get("name", "")
            if name.startswith(SKIP_LINK_PREFIXES):
                continue

            link_pose = parse_pose(link.findtext("pose"))
            link_pose = (
                link_pose[0] + model_offset[0],
                link_pose[1] + model_offset[1],
                link_pose[2] + model_offset[2],
                link_pose[3],
                link_pose[4],
                link_pose[5],
            )

            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is None:
                    continue

                local_points: list[tuple[float, float, float]]
                box = geometry.find("box")
                cylinder = geometry.find("cylinder")
                if box is not None:
                    size_text = box.findtext("size")
                    if not size_text:
                        continue
                    local_points = sample_box(tuple(float(v) for v in size_text.split()), spacing)  # type: ignore[arg-type]
                elif cylinder is not None:
                    radius = float(cylinder.findtext("radius", "0"))
                    length = float(cylinder.findtext("length", "0"))
                    local_points = sample_cylinder(radius, length, spacing)
                else:
                    continue

                points.extend(transform_point(point, link_pose) for point in local_points)
    return points


def voxel_downsample(
    points: list[tuple[float, float, float]],
    voxel: float,
) -> list[tuple[float, float, float]]:
    if voxel <= 0.0:
        return points
    buckets: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for point in points:
        key = tuple(int(math.floor(coord / voxel)) for coord in point)
        buckets.setdefault(key, point)
    return sorted(buckets.values())


def write_pcd(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("# .PCD v0.7 - Point Cloud Data file format\n")
        file.write("VERSION 0.7\n")
        file.write("FIELDS x y z intensity\n")
        file.write("SIZE 4 4 4 4\n")
        file.write("TYPE F F F F\n")
        file.write("COUNT 1 1 1 1\n")
        file.write(f"WIDTH {len(points)}\n")
        file.write("HEIGHT 1\n")
        file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write(f"POINTS {len(points)}\n")
        file.write("DATA ascii\n")
        for x, y, z in points:
            file.write(f"{x:.4f} {y:.4f} {z:.4f} 100.0\n")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world",
        type=Path,
        default=repo_root / "sim/worlds/pipe_factory_local.world",
        help="SDF world to sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "assets/maps/pipe_factory_local.pcd",
        help="Output PCD path.",
    )
    parser.add_argument("--spacing", type=float, default=0.18)
    parser.add_argument("--voxel", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = extract_points(args.world, args.spacing)
    points = voxel_downsample(points, args.voxel)
    write_pcd(args.output, points)
    print(f"Wrote {args.output} points={len(points)}")


if __name__ == "__main__":
    main()
