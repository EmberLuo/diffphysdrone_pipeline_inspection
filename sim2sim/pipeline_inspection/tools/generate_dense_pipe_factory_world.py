#!/usr/bin/env python3
"""Generate a large, dense pipe-factory Gazebo world for mapping demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_PATH = REPO_ROOT / "sim/worlds/pipe_factory_local.world"


@dataclass(frozen=True)
class Material:
    name: str
    ambient: str
    diffuse: str


MATERIALS = {
    "steel": Material("steel", "0.34 0.35 0.36 1", "0.58 0.60 0.61 1"),
    "water": Material("water", "0.05 0.22 0.36 1", "0.10 0.38 0.62 1"),
    "product": Material("product", "0.45 0.17 0.08 1", "0.77 0.30 0.14 1"),
    "gas": Material("gas", "0.48 0.42 0.12 1", "0.86 0.75 0.20 1"),
    "utility": Material("utility", "0.20 0.24 0.23 1", "0.36 0.43 0.40 1"),
    "support": Material("support", "0.13 0.14 0.15 1", "0.28 0.30 0.32 1"),
    "floor": Material("floor", "0.42 0.43 0.42 1", "0.55 0.56 0.54 1"),
    "lane": Material("lane", "0.16 0.17 0.18 1", "0.25 0.26 0.27 1"),
    "catwalk": Material("catwalk", "0.24 0.24 0.22 1", "0.43 0.43 0.39 1"),
    "tank": Material("tank", "0.44 0.45 0.43 1", "0.72 0.73 0.70 1"),
    "marker": Material("marker", "0.72 0.05 0.04 1", "0.95 0.08 0.06 1"),
}


def material_xml(key: str) -> str:
    material = MATERIALS[key]
    return f"""          <material>
            <ambient>{material.ambient}</ambient>
            <diffuse>{material.diffuse}</diffuse>
          </material>"""


def box_link(name: str, pose: str, size: str, material: str, collision: bool = True) -> str:
    collision_xml = ""
    if collision:
        collision_xml = f"""        <collision name="collision">
          <geometry><box><size>{size}</size></box></geometry>
        </collision>
"""
    return f"""      <link name="{name}">
        <pose>{pose}</pose>
{collision_xml}        <visual name="visual">
          <geometry><box><size>{size}</size></box></geometry>
{material_xml(material)}
        </visual>
      </link>
"""


def cylinder_link(
    name: str,
    pose: str,
    radius: float,
    length: float,
    material: str,
    collision: bool = True,
) -> str:
    collision_xml = ""
    if collision:
        collision_xml = f"""        <collision name="collision">
          <geometry><cylinder><radius>{radius:.3f}</radius><length>{length:.2f}</length></cylinder></geometry>
        </collision>
"""
    return f"""      <link name="{name}">
        <pose>{pose}</pose>
{collision_xml}        <visual name="visual">
          <geometry><cylinder><radius>{radius:.3f}</radius><length>{length:.2f}</length></cylinder></geometry>
{material_xml(material)}
        </visual>
      </link>
"""


def sphere_link(name: str, pose: str, radius: float, material: str) -> str:
    return f"""      <link name="{name}">
        <pose>{pose}</pose>
        <visual name="visual">
          <geometry><sphere><radius>{radius:.3f}</radius></sphere></geometry>
{material_xml(material)}
        </visual>
      </link>
"""


def model(name: str, links: list[str]) -> str:
    return f"""    <model name="{name}">
      <static>true</static>
{''.join(links)}    </model>
"""


def build_world() -> str:
    floor_links = [
        box_link("concrete_pad", "0 0 0.01 0 0 0", "120 80 0.02", "floor"),
        box_link("center_safety_lane", "0 0 0.035 0 0 0", "108 2.2 0.03", "lane", False),
        box_link("north_lane_line", "0 1.15 0.055 0 0 0", "108 0.08 0.02", "gas", False),
        box_link("south_lane_line", "0 -1.15 0.055 0 0 0", "108 0.08 0.02", "gas", False),
    ]

    pipe_links: list[str] = []
    x_levels = [
        (1.35, 0.13, "steel"),
        (2.45, 0.16, "water"),
        (3.65, 0.20, "product"),
        (5.05, 0.15, "gas"),
        (6.75, 0.11, "utility"),
    ]
    y_levels = [
        (1.90, 0.12, "water"),
        (3.10, 0.18, "product"),
        (4.55, 0.14, "steel"),
        (6.10, 0.12, "gas"),
        (7.35, 0.10, "utility"),
    ]
    y_rows = [-30, -24, -18, -12, -6, 6, 12, 18, 24, 30]
    x_cols = [-48, -40, -32, -24, -16, -8, 8, 16, 24, 32, 40, 48]

    for row_i, y in enumerate(y_rows):
        for level_i, (z, radius, material) in enumerate(x_levels):
            offset = ((row_i + level_i) % 3 - 1) * 0.32
            pipe_links.append(
                cylinder_link(
                    f"x_pipe_r{row_i:02d}_l{level_i}",
                    f"0 {y + offset:.2f} {z:.2f} 0 1.5708 0",
                    radius,
                    108.0 - 2.0 * (level_i % 2),
                    material,
                )
            )

    for col_i, x in enumerate(x_cols):
        for level_i, (z, radius, material) in enumerate(y_levels):
            offset = ((col_i + 2 * level_i) % 3 - 1) * 0.28
            pipe_links.append(
                cylinder_link(
                    f"y_pipe_c{col_i:02d}_l{level_i}",
                    f"{x + offset:.2f} 0 {z:.2f} 1.5708 0 0",
                    radius,
                    66.0 - 1.5 * (level_i % 2),
                    material,
                )
            )

    for x_i, x in enumerate([-48, -32, -16, 16, 32, 48]):
        for y_i, y in enumerate([-30, -18, -6, 6, 18, 30]):
            material = ["steel", "water", "product", "gas"][(x_i + y_i) % 4]
            radius = [0.10, 0.12, 0.14][(x_i + y_i) % 3]
            pipe_links.append(
                cylinder_link(
                    f"riser_{x_i:02d}_{y_i:02d}",
                    f"{x:.1f} {y:.1f} 4.15 0 0 0",
                    radius,
                    6.9,
                    material,
                )
            )

    support_links: list[str] = []
    for x in [-52, -36, -20, 0, 20, 36, 52]:
        for y in [-33, -21, -9, 9, 21, 33]:
            support_links.append(
                box_link(
                    f"support_{x:+.0f}_{y:+.0f}".replace("+", "p").replace("-", "m"),
                    f"{x} {y} 3.65 0 0 0",
                    "0.30 1.25 7.30",
                    "support",
                )
            )
    for y in [-27, -15, 15, 27]:
        support_links.append(box_link(f"catwalk_x_{y:+.0f}".replace("+", "p").replace("-", "m"), f"0 {y} 4.85 0 0 0", "104 0.72 0.10", "catwalk"))
    for x in [-44, -28, -12, 12, 28, 44]:
        support_links.append(box_link(f"catwalk_y_{x:+.0f}".replace("+", "p").replace("-", "m"), f"{x} 0 5.55 0 0 0", "0.72 62 0.10", "catwalk"))

    equipment_links: list[str] = []
    for idx, (x, y, radius, height) in enumerate(
        [
            (-52, 35, 2.3, 5.2),
            (-34, 35, 1.8, 4.4),
            (34, 35, 2.1, 4.8),
            (52, 35, 1.7, 4.2),
            (-52, -35, 2.0, 4.8),
            (-34, -35, 1.6, 4.0),
            (34, -35, 2.2, 5.0),
            (52, -35, 1.9, 4.6),
        ]
    ):
        equipment_links.append(cylinder_link(f"tank_{idx}", f"{x} {y} {height * 0.5:.2f} 0 0 0", radius, height, "tank"))
        equipment_links.append(sphere_link(f"inspection_marker_{idx}", f"{x} {y - (1.15 if y > 0 else -1.15):.2f} {height + 0.45:.2f} 0 0 0", 0.20, "marker"))

    for idx, (x, y, yaw) in enumerate([(-56, -8, 0.08), (-56, 8, -0.08), (56, -8, -0.08), (56, 8, 0.08)]):
        equipment_links.append(box_link(f"pump_skid_{idx}", f"{x} {y} 0.45 0 0 {yaw}", "2.8 1.25 0.9", "water"))
        equipment_links.append(cylinder_link(f"pump_discharge_{idx}", f"{x * 0.92:.2f} {y} 1.6 0 1.5708 0", 0.12, 6.0, "steel"))

    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <scene>
      <ambient>0.82 0.84 0.86 1</ambient>
      <background>0.48 0.52 0.56 1</background>
      <shadows>true</shadows>
    </scene>

    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>-48 -56 30 0 0.44 0.78</pose>
      </camera>
    </gui>

    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>

    <physics name="default_physics" default="0" type="ode">
      <gravity>0 0 -9.8066</gravity>
      <ode>
        <solver><type>quick</type><iters>10</iters><sor>1.3</sor><use_dynamic_moi_rescaling>0</use_dynamic_moi_rescaling></solver>
        <constraints><cfm>0</cfm><erp>0.2</erp><contact_max_correcting_vel>100</contact_max_correcting_vel><contact_surface_layer>0.001</contact_surface_layer></constraints>
      </ode>
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
      <magnetic_field>6.0e-6 2.3e-5 -4.2e-5</magnetic_field>
    </physics>

{model("pipe_factory_floor", floor_links)}
{model("dense_interwoven_pipe_grid", pipe_links)}
{model("pipe_support_structure", support_links)}
{model("process_tanks_and_equipment", equipment_links)}
  </world>
</sdf>
"""


def main() -> None:
    WORLD_PATH.write_text(build_world(), encoding="utf-8")
    print(f"Wrote {WORLD_PATH}")


if __name__ == "__main__":
    main()
