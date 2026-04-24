import math
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import genesis as gs
import torch
from genesis.utils.geom import quat_to_xyz

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"
DRONE_URDF = str(ASSET_DIR / "drone_ex1" / "drone_ex1.urdf")


def _read_vehicle_params_from_urdf(urdf_path: str) -> dict:
    root = ET.parse(urdf_path).getroot()

    props = root.find("properties")
    attrs = {} if props is None else props.attrib
    mass_node = root.find("./link[@name='base_link']/inertial/mass")
    mass = float(mass_node.attrib["value"]) if mass_node is not None else 0.3

    params = {
        "mass": mass,
        "kf": float(attrs.get("kf", 3.16e-10)),
        "km": float(attrs.get("km", 7.94e-12)),
        "thrust2weight": float(attrs.get("thrust2weight", 2.25)),
    }
    return params


class NavEnv:
    def __init__(self, cfg: dict, num_agents: int, show_viewer: bool = True, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        print("[NavEnv] init: start", flush=True)

        sim_cfg = cfg["sim"]
        scene_cfg = cfg["scene"]
        task_cfg = cfg["task"]
        depth_cfg = cfg["depth_camera"]
        term_cfg = cfg["termination"]
        obs_cfg = cfg["obstacles"]
        static_cfg = obs_cfg["static"]
        dynamic_cfg = obs_cfg["dynamic"]

        self.dt = float(sim_cfg.get("dt", 0.01))
        self.max_steps = int(sim_cfg.get("max_steps", 4000))
        self.reach_threshold = float(task_cfg.get("reach_threshold", 0.35))

        self.min_height = float(term_cfg.get("min_height", 0.1))
        self.max_roll_deg = float(term_cfg.get("max_roll_deg", 80.0))
        self.max_pitch_deg = float(term_cfg.get("max_pitch_deg", 80.0))
        self.max_abs_x = float(term_cfg.get("max_abs_x", 40.0))
        self.max_abs_y = float(term_cfg.get("max_abs_y", 40.0))
        self.max_abs_z = float(term_cfg.get("max_abs_z", 20.0))
        self.inter_drone_collision_dist = float(term_cfg.get("inter_drone_collision_dist", 0.28))
        self.enable_height_termination = bool(term_cfg.get("enable_height_termination", True))
        self.enable_attitude_termination = bool(term_cfg.get("enable_attitude_termination", True))
        self.enable_bounds_termination = bool(term_cfg.get("enable_bounds_termination", True))

        self.num_agents = int(num_agents)
        if self.num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {self.num_agents}")
        print(f"[NavEnv] init: num_agents={self.num_agents}", flush=True)

        self.agent_names = [f"drone_{i + 1}" for i in range(self.num_agents)]

        self.vehicle_params = _read_vehicle_params_from_urdf(DRONE_URDF)

        self.layout_cfg = task_cfg.get("layout_sampling", {})
        self.layout_max_attempts = int(self.layout_cfg.get("max_attempts", 2000))
        self.layout_start_goal_clearance = float(self.layout_cfg.get("start_goal_clearance", 0.8))
        self.layout_inter_agent_clearance = float(self.layout_cfg.get("inter_agent_clearance", 0.8))
        self.start_z_range = tuple(float(x) for x in self.layout_cfg.get("start_z_range", [0.9, 1.3]))
        self.goal_z_range = tuple(float(x) for x in self.layout_cfg.get("goal_z_range", [0.9, 1.3]))

        static_regions = static_cfg.get("regions", {})
        if "left" not in static_regions or "right" not in static_regions:
            raise ValueError("obstacles.static.regions must contain left and right.")

        self.static_count_per_region = int(static_cfg.get("count_per_region", 20))
        self.static_middle_count = int(static_cfg.get("middle_count", self.static_count_per_region))
        self.fill_middle_with_static_when_dynamic_disabled = bool(
            static_cfg.get("fill_middle_when_dynamic_disabled", True)
        )
        self.static_radius = float(static_cfg.get("radius", 0.28))
        self.static_height = float(static_cfg.get("height", 2.6))
        self.static_inter_obstacle_spacing = float(static_cfg.get("inter_obstacle_spacing", 0.1))
        self.static_placement_max_attempts = int(static_cfg.get("placement_max_attempts", 3000))
        self.static_color = tuple(static_cfg.get("obstacle_color", [0.45, 0.45, 0.48]))
        self.left_region = static_regions["left"]
        self.right_region = static_regions["right"]
        self.middle_region = static_cfg.get("middle_region", dynamic_cfg.get("region", {"x_range": [-5.0, 5.0], "y_range": [-5.0, 5.0]}))

        self.dynamic_enable = bool(dynamic_cfg.get("enable", True))
        self.dynamic_count = int(dynamic_cfg.get("count", 8))
        dyn_size = dynamic_cfg.get("size", [0.8, 0.8, 1.2])
        if len(dyn_size) != 3:
            raise ValueError("obstacles.dynamic.size must be [sx, sy, sz].")
        self.dynamic_size = torch.tensor([float(x) for x in dyn_size], dtype=torch.float32, device=self.device)
        self.dynamic_half_size = self.dynamic_size * 0.5
        self.dynamic_z_range = tuple(float(x) for x in dynamic_cfg.get("z_range", [0.3, 3.0]))
        self.dynamic_speed_range = tuple(float(x) for x in dynamic_cfg.get("speed_range", [0.4, 1.2]))
        self.dynamic_waypoint_count = int(dynamic_cfg.get("waypoint_count", 6))
        self.dynamic_waypoint_reach = float(dynamic_cfg.get("waypoint_reach", 0.25))
        self.dynamic_placement_max_attempts = int(dynamic_cfg.get("placement_max_attempts", 2000))
        self.dynamic_waypoint_max_attempts = int(dynamic_cfg.get("waypoint_max_attempts", 1000))
        self.dynamic_static_clearance = float(dynamic_cfg.get("static_clearance", 0.2))
        self.dynamic_boundary_margin_xy = float(dynamic_cfg.get("boundary_margin_xy", 0.2))
        self.dynamic_color = tuple(dynamic_cfg.get("obstacle_color", [0.8, 0.25, 0.2]))
        self.dynamic_region = dynamic_cfg.get("region", {"x_range": [-5.0, 5.0], "y_range": [-5.0, 5.0]})
        dyn_collision_cfg = dynamic_cfg.get("collision", {})
        self.dynamic_collision_drone_radius = float(dyn_collision_cfg.get("drone_radius", 0.2))

        self.static_region_specs = [
            ("left", self.left_region, self.static_count_per_region),
            ("right", self.right_region, self.static_count_per_region),
        ]
        if (not self.dynamic_enable) and self.fill_middle_with_static_when_dynamic_disabled:
            self.static_region_specs.append(("middle", self.middle_region, self.static_middle_count))

        self.num_static_total = sum(int(spec[2]) for spec in self.static_region_specs)
        if not self.dynamic_enable:
            self.dynamic_count = 0

        boundary_cfg = obs_cfg.get("boundary_walls", {})
        self.boundary_walls_enable = bool(boundary_cfg.get("enable", True))
        self.boundary_walls_thickness = float(boundary_cfg.get("thickness", 0.2))
        self.boundary_walls_height = float(boundary_cfg.get("height", 8.0))
        self.boundary_walls_x_margin = float(boundary_cfg.get("x_margin", 0.0))
        self.boundary_walls_color = tuple(boundary_cfg.get("color", [0.25, 0.25, 0.28]))
        if self.boundary_walls_enable:
            if self.boundary_walls_thickness <= 0.0:
                raise ValueError("obstacles.boundary_walls.thickness must be > 0 when enabled.")
            if self.boundary_walls_height <= 0.0:
                raise ValueError("obstacles.boundary_walls.height must be > 0 when enabled.")

        all_regions = [self.left_region, self.right_region, self.middle_region]
        x_lows, x_highs, y_lows, y_highs = [], [], [], []
        for region_cfg in all_regions:
            x_lo, x_hi, y_lo, y_hi = self._region_ranges(region_cfg)
            x_lows.append(x_lo)
            x_highs.append(x_hi)
            y_lows.append(y_lo)
            y_highs.append(y_hi)
        self.corridor_x_min = min(x_lows)
        self.corridor_x_max = max(x_highs)
        self.corridor_y_min = min(y_lows)
        self.corridor_y_max = max(y_highs)

        print(
            f"[NavEnv] init: static_per_region={self.static_count_per_region} "
            f"static_total={self.num_static_total} dynamic_count={self.dynamic_count} "
            f"dynamic_enable={self.dynamic_enable}",
            flush=True,
        )

        print("[NavEnv] init: creating scene", flush=True)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=int(sim_cfg.get("substeps", 2))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(scene_cfg.get("max_visualize_fps", 60)),
                camera_pos=tuple(scene_cfg.get("viewer_camera_pos", [0.0, 20.0, 10.0])),
                camera_lookat=tuple(scene_cfg.get("viewer_camera_lookat", [0.0, 0.0, 1.0])),
                camera_fov=float(scene_cfg.get("viewer_camera_fov", 45)),
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                background_color=tuple(scene_cfg.get("background_color", [0.9, 0.9, 0.9])),
                ambient_light=tuple(scene_cfg.get("ambient_light", [0.12, 0.12, 0.12])),
                lights=[
                    gs.options.vis.DirectionalLight(
                        dir=tuple(scene_cfg.get("sun_dir", [0.0, 0.0, -1.0])),
                        color=tuple(scene_cfg.get("sun_color", [1.0, 1.0, 1.0])),
                        intensity=float(scene_cfg.get("sun_intensity", 5.0)),
                    )
                ],
                shadow=bool(scene_cfg.get("shadow", True)),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=bool(sim_cfg.get("enable_collision", True)),
                enable_joint_limit=False,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ColorTexture(color=tuple(scene_cfg.get("plane_color", [0.7, 0.7, 0.7]))),
            ),
        )

        self.boundary_walls = []
        if self.boundary_walls_enable:
            x_span = (self.corridor_x_max - self.corridor_x_min) + 2.0 * self.boundary_walls_x_margin
            wall_z = 0.5 * self.boundary_walls_height
            wall_x = 0.5 * (self.corridor_x_min + self.corridor_x_max)
            wall_low_y = self.corridor_y_min - 0.5 * self.boundary_walls_thickness
            wall_high_y = self.corridor_y_max + 0.5 * self.boundary_walls_thickness
            wall_size = (x_span, self.boundary_walls_thickness, self.boundary_walls_height)

            for wall_y in (wall_low_y, wall_high_y):
                wall = self.scene.add_entity(
                    morph=gs.morphs.Box(
                        pos=(wall_x, wall_y, wall_z),
                        size=wall_size,
                        fixed=True,
                        collision=True,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=self.boundary_walls_color),
                    ),
                )
                self.boundary_walls.append(wall)

            print(
                "[NavEnv] boundary walls created: 2 "
                f"size=({wall_size[0]:.2f},{wall_size[1]:.2f},{wall_size[2]:.2f}) "
                f"y=({wall_low_y:.2f},{wall_high_y:.2f})",
                flush=True,
            )
        else:
            print("[NavEnv] boundary walls disabled", flush=True)

        self.static_obstacles = []
        print("[NavEnv] init: creating static obstacle entities", flush=True)
        for _ in range(self.num_static_total):
            obs = self.scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=self.static_radius,
                    height=self.static_height,
                    pos=(0.0, 0.0, self.static_height * 0.5),
                    fixed=True,
                    collision=True,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=self.static_color),
                ),
            )
            self.static_obstacles.append(obs)

        self.dynamic_obstacles = []
        if self.dynamic_enable and self.dynamic_count > 0:
            print("[NavEnv] init: creating dynamic obstacle entities", flush=True)
            for _ in range(self.dynamic_count):
                obs = self.scene.add_entity(
                    morph=gs.morphs.Box(
                        size=tuple(float(x) for x in self.dynamic_size.tolist()),
                        pos=(0.0, 0.0, 0.5 * float(self.dynamic_size[2].item())),
                        fixed=True,
                        collision=True,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=self.dynamic_color),
                    ),
                )
                self.dynamic_obstacles.append(obs)
        else:
            print("[NavEnv] init: dynamic obstacles disabled", flush=True)

        self.drones = []
        self.targets = []
        self.depth_sensors = []
        print("[NavEnv] init: creating drones/targets/sensors", flush=True)

        drone_colors = scene_cfg.get(
            "drone_colors",
            [
                [0.90, 0.15, 0.15],
                [0.15, 0.75, 0.25],
                [0.15, 0.45, 0.90],
                [0.90, 0.65, 0.15],
                [0.70, 0.20, 0.85],
                [0.10, 0.75, 0.75],
            ],
        )

        for i in range(self.num_agents):
            drone = self.scene.add_entity(
                morph=gs.morphs.Drone(file=DRONE_URDF),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=tuple(drone_colors[i % len(drone_colors)])),
                ),
            )
            self.drones.append(drone)

            target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=float(task_cfg.get("target_sphere_scale", 0.05)),
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=tuple(task_cfg.get("target_color", [0.2, 0.4, 0.9]))),
                ),
            )
            self.targets.append(target)

            pattern = gs.options.sensors.raycaster.DepthCameraPattern(
                res=(int(depth_cfg.get("width", 48)), int(depth_cfg.get("height", 36))),
                fov_horizontal=float(depth_cfg.get("fov_horizontal", 79.0)),
            )
            sensor = self.scene.add_sensor(
                gs.sensors.DepthCamera(
                    entity_idx=drone.idx,
                    link_idx_local=int(depth_cfg.get("link_idx_local", 0)),
                    pos_offset=tuple(depth_cfg.get("pos_offset", [0.25, 0.0, -0.1])),
                    euler_offset=tuple(depth_cfg.get("euler_offset", [0.0, 10.0, 0.0])),
                    pattern=pattern,
                    min_range=float(depth_cfg.get("min_range", 0.3)),
                    max_range=float(depth_cfg.get("max_range", 24.0)),
                    no_hit_value=float(depth_cfg.get("no_hit_value", 24.0)),
                )
            )
            self.depth_sensors.append(sensor)

        self.record_cam = None
        if bool(scene_cfg.get("enable_record_camera", True)):
            rec_cfg = scene_cfg.get("record_camera", {})
            self.record_cam = self.scene.add_camera(
                res=tuple(rec_cfg.get("res", [1280, 720])),
                pos=tuple(rec_cfg.get("pos", [0.0, 20.0, 10.0])),
                lookat=tuple(rec_cfg.get("lookat", [0.0, 0.0, 1.0])),
                fov=float(rec_cfg.get("fov", 45)),
                GUI=False,
            )

        print("[NavEnv] init: scene.build start", flush=True)
        t_build0 = time.perf_counter()
        self.scene.build(n_envs=1)
        build_sec = time.perf_counter() - t_build0
        print(f"[NavEnv] init: scene.build done in {build_sec:.2f}s", flush=True)
        self._env0 = torch.tensor([0], device=self.device, dtype=gs.tc_int)

        self.base_pos = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_agents, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.target_pos = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)

        self.static_pos = torch.zeros((self.num_static_total, 3), device=self.device, dtype=gs.tc_float)
        self.dynamic_pos = torch.zeros((self.dynamic_count, 3), device=self.device, dtype=gs.tc_float)
        self.dynamic_waypoints = torch.zeros(
            (self.dynamic_count, self.dynamic_waypoint_count, 3), device=self.device, dtype=gs.tc_float
        )
        self.dynamic_wp_idx = torch.zeros((self.dynamic_count,), device=self.device, dtype=torch.long)
        self.dynamic_speed = torch.zeros((self.dynamic_count,), device=self.device, dtype=gs.tc_float)

        self.step_count = 0
        print("[NavEnv] init: done", flush=True)

    @staticmethod
    def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
        half = 0.5 * yaw
        q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    @staticmethod
    def _region_ranges(region_cfg: dict) -> tuple[float, float, float, float]:
        xr = region_cfg.get("x_range", None)
        yr = region_cfg.get("y_range", None)
        if xr is None or yr is None or len(xr) != 2 or len(yr) != 2:
            raise ValueError("Region config must contain x_range/y_range with length 2.")
        return float(xr[0]), float(xr[1]), float(yr[0]), float(yr[1])

    @staticmethod
    def _sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
        return rng.random() * (hi - lo) + lo

    @staticmethod
    def _xy_dist_sq(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
        dx = a_x - b_x
        dy = a_y - b_y
        return dx * dx + dy * dy

    def _sync_static_entities(self):
        for i, obs in enumerate(self.static_obstacles):
            obs.set_pos(self.static_pos[i : i + 1], envs_idx=self._env0, zero_velocity=True)

    def _sync_dynamic_entities(self):
        for i, obs in enumerate(self.dynamic_obstacles):
            obs.set_pos(self.dynamic_pos[i : i + 1], envs_idx=self._env0, zero_velocity=True)

    def _sample_static_region(
        self,
        rng: random.Random,
        count: int,
        region_cfg: dict,
        existing: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        x_lo, x_hi, y_lo, y_hi = self._region_ranges(region_cfg)

        out: list[tuple[float, float, float]] = []
        rr = self.static_radius
        min_sep = 2.0 * rr + self.static_inter_obstacle_spacing
        min_sep_sq = min_sep * min_sep

        x_min = x_lo + rr
        x_max = x_hi - rr
        y_min = y_lo + rr
        y_max = y_hi - rr

        for _ in range(count):
            placed = False
            for _ in range(self.static_placement_max_attempts):
                px = self._sample_uniform(rng, x_min, x_max)
                py = self._sample_uniform(rng, y_min, y_max)

                valid = True
                for ox, oy, _ in existing:
                    if self._xy_dist_sq(px, py, ox, oy) < min_sep_sq:
                        valid = False
                        break
                if not valid:
                    continue

                for ox, oy, _ in out:
                    if self._xy_dist_sq(px, py, ox, oy) < min_sep_sq:
                        valid = False
                        break
                if not valid:
                    continue

                out.append((px, py, self.static_height * 0.5))
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    "Failed to sample static obstacle map without overlap. "
                    "Try lowering count_per_region or static radius."
                )

        return out

    def _dynamic_sample_candidate(self, rng: random.Random) -> tuple[float, float, float]:
        x_lo, x_hi, y_lo, y_hi = self._region_ranges(self.dynamic_region)

        hx = float(self.dynamic_half_size[0].item()) + self.dynamic_boundary_margin_xy
        hy = float(self.dynamic_half_size[1].item()) + self.dynamic_boundary_margin_xy

        x_min = x_lo + hx
        x_max = x_hi - hx
        y_min = y_lo + hy
        y_max = y_hi - hy

        z_lo = self.dynamic_z_range[0] + float(self.dynamic_half_size[2].item())
        z_hi = self.dynamic_z_range[1] - float(self.dynamic_half_size[2].item())

        if x_min >= x_max or y_min >= y_max or z_lo >= z_hi:
            raise RuntimeError("Invalid dynamic obstacle bounds; check region/size/z_range settings.")

        x = self._sample_uniform(rng, x_min, x_max)
        y = self._sample_uniform(rng, y_min, y_max)
        z = self._sample_uniform(rng, z_lo, z_hi)
        return x, y, z

    def _dynamic_ok_vs_static(self, px: float, py: float) -> bool:
        dyn_footprint = math.sqrt(
            float(self.dynamic_half_size[0].item()) ** 2 + float(self.dynamic_half_size[1].item()) ** 2
        )
        min_sep = self.static_radius + dyn_footprint + self.dynamic_static_clearance
        min_sep_sq = min_sep * min_sep

        for s in self.static_pos.tolist():
            if self._xy_dist_sq(px, py, float(s[0]), float(s[1])) < min_sep_sq:
                return False
        return True

    def _sample_dynamic_map_and_paths(self, rng: random.Random):
        for i in range(self.dynamic_count):
            found = False
            for _ in range(self.dynamic_placement_max_attempts):
                px, py, pz = self._dynamic_sample_candidate(rng)
                if self._dynamic_ok_vs_static(px, py):
                    self.dynamic_pos[i] = torch.tensor([px, py, pz], device=self.device, dtype=gs.tc_float)
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    "Failed to place dynamic obstacle without static overlap. "
                    "Try adjusting dynamic region or clearance."
                )

            for wp_idx in range(self.dynamic_waypoint_count):
                wp_found = False
                for _ in range(self.dynamic_waypoint_max_attempts):
                    wx, wy, wz = self._dynamic_sample_candidate(rng)
                    if self._dynamic_ok_vs_static(wx, wy):
                        self.dynamic_waypoints[i, wp_idx] = torch.tensor(
                            [wx, wy, wz], device=self.device, dtype=gs.tc_float
                        )
                        wp_found = True
                        break
                if not wp_found:
                    raise RuntimeError(
                        "Failed to sample dynamic obstacle waypoint. "
                        "Try reducing waypoint_count or clearance."
                    )

            sp_lo, sp_hi = self.dynamic_speed_range
            self.dynamic_speed[i] = self._sample_uniform(rng, sp_lo, sp_hi)

        self.dynamic_wp_idx.zero_()

    def _sample_points_in_region_avoiding_obstacles(
        self,
        rng: random.Random,
        count: int,
        region_cfg: dict,
        z_range: tuple[float, float],
        inter_point_clearance: float,
        obstacle_clearance: float,
    ) -> list[tuple[float, float, float]]:
        x_lo, x_hi, y_lo, y_hi = self._region_ranges(region_cfg)
        out: list[tuple[float, float, float]] = []

        min_sep_sq = inter_point_clearance * inter_point_clearance

        for _ in range(count):
            placed = False
            for _ in range(self.layout_max_attempts):
                px = self._sample_uniform(rng, x_lo, x_hi)
                py = self._sample_uniform(rng, y_lo, y_hi)
                pz = self._sample_uniform(rng, z_range[0], z_range[1])

                valid = True

                obs_min_sep = self.static_radius + obstacle_clearance
                obs_min_sep_sq = obs_min_sep * obs_min_sep
                for s in self.static_pos.tolist():
                    if self._xy_dist_sq(px, py, float(s[0]), float(s[1])) < obs_min_sep_sq:
                        valid = False
                        break
                if not valid:
                    continue

                for ox, oy, oz in out:
                    d2 = (px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2
                    if d2 < min_sep_sq:
                        valid = False
                        break
                if not valid:
                    continue

                out.append((px, py, pz))
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    "Failed to sample start/goal layout in obstacle gaps. "
                    "Try reducing num_drones or clearance constraints."
                )

        return out

    def sample_map_and_layout(self, num_drones: int, seed: int):
        if int(num_drones) != self.num_agents:
            raise ValueError(f"sample_map_and_layout num_drones mismatch: {num_drones} vs {self.num_agents}")

        print(f"[NavEnv] sample_map_and_layout: start seed={seed}", flush=True)
        rng = random.Random(int(seed))

        all_static = []
        existing_static = []
        region_summaries = []
        for region_name, region_cfg, region_count in self.static_region_specs:
            region_positions = self._sample_static_region(
                rng,
                int(region_count),
                region_cfg,
                existing=existing_static,
            )
            existing_static.extend(region_positions)
            all_static.extend(region_positions)
            region_summaries.append(f"{region_name}:{len(region_positions)}")

        self.static_pos[:] = torch.tensor(all_static, device=self.device, dtype=gs.tc_float)
        self._sync_static_entities()
        print(
            "[NavEnv] sample_map_and_layout: static obstacles placed "
            f"total={len(all_static)} ({', '.join(region_summaries)})",
            flush=True,
        )

        if self.dynamic_enable and self.dynamic_count > 0:
            self._sample_dynamic_map_and_paths(rng)
            self._sync_dynamic_entities()
            print(
                f"[NavEnv] sample_map_and_layout: dynamic obstacles placed total={self.dynamic_count}",
                flush=True,
            )
        else:
            print("[NavEnv] sample_map_and_layout: dynamic obstacles skipped (disabled)", flush=True)

        left_count = (self.num_agents + 1) // 2
        right_count = self.num_agents - left_count

        left_starts = self._sample_points_in_region_avoiding_obstacles(
            rng,
            left_count,
            self.left_region,
            self.start_z_range,
            inter_point_clearance=self.layout_inter_agent_clearance,
            obstacle_clearance=self.layout_start_goal_clearance,
        )
        right_starts = self._sample_points_in_region_avoiding_obstacles(
            rng,
            right_count,
            self.right_region,
            self.start_z_range,
            inter_point_clearance=self.layout_inter_agent_clearance,
            obstacle_clearance=self.layout_start_goal_clearance,
        )

        right_goals = self._sample_points_in_region_avoiding_obstacles(
            rng,
            left_count,
            self.right_region,
            self.goal_z_range,
            inter_point_clearance=self.layout_inter_agent_clearance,
            obstacle_clearance=self.layout_start_goal_clearance,
        )
        left_goals = self._sample_points_in_region_avoiding_obstacles(
            rng,
            right_count,
            self.left_region,
            self.goal_z_range,
            inter_point_clearance=self.layout_inter_agent_clearance,
            obstacle_clearance=self.layout_start_goal_clearance,
        )

        starts_list = left_starts + right_starts
        goals_list = right_goals + left_goals

        starts = torch.tensor(starts_list, device=self.device, dtype=torch.float32)
        goals = torch.tensor(goals_list, device=self.device, dtype=torch.float32)
        yaw = torch.atan2(goals[:, 1] - starts[:, 1], goals[:, 0] - starts[:, 0])
        print(
            f"[NavEnv] sample_map_and_layout: start/goal ready count={self.num_agents}",
            flush=True,
        )

        return self.agent_names, starts, goals, yaw

    def _update_target_visual(self):
        for i, target in enumerate(self.targets):
            target.set_pos(self.target_pos[i : i + 1], zero_velocity=True, envs_idx=self._env0)

    def _update_state(self):
        for i, drone in enumerate(self.drones):
            self.base_pos[i] = drone.get_pos()[0]
            self.base_quat[i] = drone.get_quat()[0]
            self.base_lin_vel[i] = drone.get_vel()[0]
            self.base_ang_vel[i] = drone.get_ang()[0]

    def _advance_dynamic_obstacles(self):
        if self.dynamic_count <= 0:
            return

        for i in range(self.dynamic_count):
            idx = int(self.dynamic_wp_idx[i].item())
            pos = self.dynamic_pos[i].clone()
            remaining = float(self.dynamic_speed[i].item()) * self.dt

            hops = 0
            while remaining > 1e-6 and hops < self.dynamic_waypoint_count + 2:
                target = self.dynamic_waypoints[i, idx]
                delta = target - pos
                dist = float(torch.norm(delta).item())

                if dist <= self.dynamic_waypoint_reach:
                    idx = (idx + 1) % self.dynamic_waypoint_count
                    hops += 1
                    continue

                if dist <= remaining:
                    pos = target.clone()
                    remaining -= dist
                    idx = (idx + 1) % self.dynamic_waypoint_count
                    hops += 1
                    continue

                pos = pos + delta / max(dist, 1e-6) * remaining
                remaining = 0.0

            self.dynamic_wp_idx[i] = idx
            self.dynamic_pos[i] = pos

        self._sync_dynamic_entities()

    def reset_episode(self, start_pos: torch.Tensor, yaw: torch.Tensor, target_pos: torch.Tensor):
        if start_pos.shape != (self.num_agents, 3):
            raise ValueError(f"start_pos shape must be ({self.num_agents}, 3), got {tuple(start_pos.shape)}")
        if target_pos.shape != (self.num_agents, 3):
            raise ValueError(f"target_pos shape must be ({self.num_agents}, 3), got {tuple(target_pos.shape)}")
        if yaw.shape != (self.num_agents,):
            raise ValueError(f"yaw shape must be ({self.num_agents},), got {tuple(yaw.shape)}")

        self._sync_static_entities()
        self._sync_dynamic_entities()

        q = self._quat_from_yaw(yaw.to(device=self.device, dtype=torch.float32))

        for i, drone in enumerate(self.drones):
            pos = start_pos[i : i + 1].to(device=self.device, dtype=gs.tc_float)
            quat = q[i : i + 1].to(device=self.device, dtype=gs.tc_float)
            drone.set_pos(pos, zero_velocity=True, envs_idx=self._env0)
            drone.set_quat(quat, zero_velocity=True, envs_idx=self._env0)
            drone.zero_all_dofs_velocity(envs_idx=self._env0)

        self.target_pos[:] = target_pos.to(device=self.device, dtype=gs.tc_float)
        self._update_target_visual()

        self.scene.step()
        for drone in self.drones:
            drone.zero_all_dofs_velocity(envs_idx=self._env0)

        self._update_state()
        self.step_count = 0

    def set_target_positions(self, target_pos: torch.Tensor):
        if target_pos.shape != (self.num_agents, 3):
            raise ValueError(f"target_pos shape must be ({self.num_agents}, 3), got {tuple(target_pos.shape)}")
        self.target_pos[:] = target_pos.to(device=self.device, dtype=gs.tc_float)
        self._update_target_visual()

    def get_depth(self) -> torch.Tensor:
        depth_stack = []
        for sensor in self.depth_sensors:
            depth = sensor.read_image()
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            depth_stack.append(depth.to(device=self.device, dtype=torch.float32))
        return torch.stack(depth_stack, dim=0)

    def _pairwise_collisions(self):
        collision_flags = torch.zeros((self.num_agents,), device=self.device, dtype=torch.bool)
        collision_pairs = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = torch.norm(self.base_pos[i] - self.base_pos[j]).item()
                if d < self.inter_drone_collision_dist:
                    collision_flags[i] = True
                    collision_flags[j] = True
                    collision_pairs.append((i, j))
        return collision_flags, collision_pairs

    def _dynamic_obstacle_collisions(self):
        if self.dynamic_count <= 0:
            return torch.zeros((self.num_agents,), device=self.device, dtype=torch.bool)

        drone_pos = self.base_pos.to(dtype=torch.float32)
        dyn_pos = self.dynamic_pos.to(dtype=torch.float32)

        delta = torch.abs(drone_pos[:, None, :] - dyn_pos[None, :, :])
        # dynamic_half_size is shape (3,); expand to broadcast over (num_agents, dynamic_count, 3).
        thresh = self.dynamic_half_size.view(1, 1, 3) + self.dynamic_collision_drone_radius
        inside = torch.all(delta <= thresh, dim=-1)
        return torch.any(inside, dim=-1)

    def step(self, rpm_cmd: torch.Tensor, record_frame: bool = False):
        if rpm_cmd.shape != (self.num_agents, 4):
            raise ValueError(f"rpm_cmd shape must be ({self.num_agents}, 4), got {tuple(rpm_cmd.shape)}")

        for i, drone in enumerate(self.drones):
            drone.set_propellels_rpm(rpm_cmd[i : i + 1])

        self._update_target_visual()
        self._advance_dynamic_obstacles()
        self.scene.step()
        if record_frame and self.record_cam is not None:
            self.record_cam.render()

        self.step_count += 1
        self._update_state()

        rel = self.target_pos - self.base_pos
        reached = torch.norm(rel, dim=-1) < self.reach_threshold

        euler_deg = quat_to_xyz(self.base_quat) * 180.0 / math.pi
        crash_attitude = (
            (torch.abs(euler_deg[:, 0]) > self.max_roll_deg)
            | (torch.abs(euler_deg[:, 1]) > self.max_pitch_deg)
        )
        crash_height = self.base_pos[:, 2] < self.min_height
        crash_bounds = (
            (torch.abs(self.base_pos[:, 0]) > self.max_abs_x)
            | (torch.abs(self.base_pos[:, 1]) > self.max_abs_y)
            | (torch.abs(self.base_pos[:, 2]) > self.max_abs_z)
        )

        crash = torch.zeros_like(crash_height)
        if self.enable_attitude_termination:
            crash |= crash_attitude
        if self.enable_height_termination:
            crash |= crash_height
        if self.enable_bounds_termination:
            crash |= crash_bounds

        inter_drone_collision, collision_pairs = self._pairwise_collisions()
        dynamic_collision = self._dynamic_obstacle_collisions()
        crash |= inter_drone_collision | dynamic_collision

        timeout = torch.full(
            (self.num_agents,),
            self.step_count >= self.max_steps,
            device=self.device,
            dtype=torch.bool,
        )
        done = reached | crash | timeout

        crash_reason = []
        for i in range(self.num_agents):
            reasons = []
            if bool(inter_drone_collision[i].item()):
                reasons.append("inter_drone")
            if bool(dynamic_collision[i].item()):
                reasons.append("dynamic_obstacle")
            if bool(crash_height[i].item()):
                reasons.append("height")
            if bool(crash_attitude[i].item()):
                reasons.append("attitude")
            if bool(crash_bounds[i].item()):
                reasons.append("bounds")
            crash_reason.append(reasons)

        info = {
            "reached": reached,
            "crash": crash,
            "crash_height": crash_height,
            "crash_attitude": crash_attitude,
            "crash_bounds": crash_bounds,
            "dynamic_collision": dynamic_collision,
            "inter_drone_collision": inter_drone_collision,
            "crash_reason": crash_reason,
            "timeout": timeout,
            "collision_pairs": collision_pairs,
            "done": done,
        }
        return info

    def start_recording(self):
        if self.record_cam is not None:
            self.record_cam.start_recording()

    def stop_recording(self, filename: str, fps: int = 60):
        if self.record_cam is not None:
            self.record_cam.stop_recording(save_to_filename=filename, fps=fps)
