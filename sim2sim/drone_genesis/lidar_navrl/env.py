import math
import xml.etree.ElementTree as ET
from pathlib import Path

import genesis as gs
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp
from genesis.utils.geom import quat_to_R, quat_to_xyz

THIS_DIR = Path(__file__).resolve().parent
DRONE_GENESIS_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]

ASSET_DIR = DRONE_GENESIS_DIR / "assets"
DRONE_URDF = str(ASSET_DIR / "drone_ex1" / "drone_ex1.urdf")

LIDAR_SENSOR_DIR = DRONE_GENESIS_DIR / "LidarSensor"
if not (LIDAR_SENSOR_DIR / "LidarSensor" / "lidar_sensor.py").exists():
    raise ImportError(f"LidarSensor package not found at {LIDAR_SENSOR_DIR}")
import sys
if str(LIDAR_SENSOR_DIR) not in sys.path:
    sys.path.insert(0, str(LIDAR_SENSOR_DIR))
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType


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
    def __init__(self, cfg: dict, env_name: str | None = None, show_viewer: bool = True, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.wp_device = device

        sim_cfg = cfg["sim"]
        scene_cfg = cfg["scene"]
        task_cfg = cfg["task"]
        lidar_cfg = cfg["lidar_sensor"]
        term_cfg = cfg["termination"]

        self.dt = float(sim_cfg.get("dt", 0.01))
        self.max_steps = int(sim_cfg.get("max_steps", 3000))
        self.reach_threshold = float(task_cfg.get("reach_threshold", 1.5))

        self.min_height = float(term_cfg.get("min_height", 0.1))
        self.max_roll_deg = float(term_cfg.get("max_roll_deg", 80.0))
        self.max_pitch_deg = float(term_cfg.get("max_pitch_deg", 80.0))
        self.max_abs_x = float(term_cfg.get("max_abs_x", 20.0))
        self.max_abs_y = float(term_cfg.get("max_abs_y", 20.0))
        self.max_abs_z = float(term_cfg.get("max_abs_z", 10.0))
        self.inter_drone_collision_dist = float(term_cfg.get("inter_drone_collision_dist", 0.28))
        self.enable_height_termination = bool(term_cfg.get("enable_height_termination", True))
        self.enable_attitude_termination = bool(term_cfg.get("enable_attitude_termination", True))
        self.enable_bounds_termination = bool(term_cfg.get("enable_bounds_termination", True))

        layouts = task_cfg.get("layouts", {})
        selected_env = str(env_name or task_cfg.get("default_env", "single_nav"))
        if selected_env not in layouts:
            raise ValueError(
                f"Unknown task layout '{selected_env}'. "
                f"Available layouts: {sorted(layouts.keys())}"
            )
        self.layout_env_name = selected_env
        agents_cfg = layouts[selected_env]["agents"]
        self.agent_names = [a["name"] for a in agents_cfg]
        self.num_agents = len(agents_cfg)
        self.vehicle_params = _read_vehicle_params_from_urdf(DRONE_URDF)

        # --- Lidar config ---
        self.lidar_range = float(lidar_cfg.get("max_range", 4.0))
        self.lidar_hbeams = int(lidar_cfg.get("horizontal_line_num", 120))
        self.lidar_vbeams = int(lidar_cfg.get("vertical_line_num", 6))
        self.lidar_vfov = (
            float(lidar_cfg.get("vertical_fov_deg_min", -10.0)),
            float(lidar_cfg.get("vertical_fov_deg_max", 20.0)),
        )
        self.lidar_sensor_offset = torch.tensor(
            lidar_cfg.get("sensor_pos_offset", [0.0, 0.0, 0.03]),
            device=self.device, dtype=torch.float32,
        )

        # --- Genesis scene ---
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=int(sim_cfg.get("substeps", 2))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(scene_cfg.get("max_visualize_fps", 60)),
                camera_pos=tuple(scene_cfg.get("viewer_camera_pos", [0.0, -8.0, 5.0])),
                camera_lookat=tuple(scene_cfg.get("viewer_camera_lookat", [0.0, 0.0, 1.2])),
                camera_fov=float(scene_cfg.get("viewer_camera_fov", 45)),
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                background_color=tuple(scene_cfg.get("background_color", [0.9, 0.9, 0.9])),
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
                diffuse_texture=gs.textures.ColorTexture(
                    color=tuple(scene_cfg.get("plane_color", [0.7, 0.7, 0.7])),
                ),
            ),
        )

        self.static_obstacles = []
        self._add_obstacles()

        drone_colors = scene_cfg.get("drone_colors", [[0.90, 0.15, 0.15]])
        self.drones = []
        self.targets = []
        for i in range(self.num_agents):
            drone = self.scene.add_entity(
                morph=gs.morphs.Drone(file=DRONE_URDF),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=tuple(drone_colors[i % len(drone_colors)]),
                    ),
                ),
            )
            self.drones.append(drone)

            target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=float(task_cfg.get("target_sphere_scale", 0.04)),
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=tuple(task_cfg.get("target_color", [0.2, 0.4, 0.9])),
                    ),
                ),
            )
            self.targets.append(target)

        self.record_cam = None
        if bool(scene_cfg.get("enable_record_camera", True)):
            rec_cfg = scene_cfg.get("record_camera", {})
            self.record_cam = self.scene.add_camera(
                res=tuple(rec_cfg.get("res", [1280, 720])),
                pos=tuple(rec_cfg.get("pos", [0.0, -8.0, 5.0])),
                lookat=tuple(rec_cfg.get("lookat", [0.0, 0.0, 1.2])),
                fov=float(rec_cfg.get("fov", 45)),
                GUI=False,
            )

        self.scene.build(n_envs=1)
        self._env0 = torch.tensor([0], device=self.device, dtype=gs.tc_int)

        # --- State tensors ---
        self.base_pos = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_agents, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.target_pos = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.step_count = 0
        self._update_state()

        # --- WARP lidar setup (after scene.build) ---
        wp.init()
        vertices, faces = self._build_lidar_mesh()
        self._lidar_wp_mesh, self._lidar_mesh_ids = self._create_warp_mesh(vertices, faces)

        sensor_pos, sensor_quat, _ = self._compute_lidar_sensor_pose()
        lidar_env_data = {
            "num_envs": self.num_agents,
            "sensor_pos_tensor": sensor_pos.contiguous(),
            "sensor_quat_tensor": sensor_quat.contiguous(),
            "mesh_ids": self._lidar_mesh_ids,
            "vertices": vertices,
            "faces": faces,
        }

        lidar_sensor_config = LidarConfig(
            sensor_type=LidarType.SIMPLE_GRID,
            dt=self.dt,
            max_range=float(lidar_cfg.get("max_range", 4.0)),
            update_frequency=float(lidar_cfg.get("update_frequency_hz", 15.0)),
            horizontal_line_num=self.lidar_hbeams,
            vertical_line_num=self.lidar_vbeams,
            horizontal_fov_deg_min=float(lidar_cfg.get("horizontal_fov_deg_min", -180)),
            horizontal_fov_deg_max=float(lidar_cfg.get("horizontal_fov_deg_max", 180)),
            vertical_fov_deg_min=self.lidar_vfov[0],
            vertical_fov_deg_max=self.lidar_vfov[1],
            return_pointcloud=False,
            pointcloud_in_world_frame=bool(lidar_cfg.get("pointcloud_in_world_frame", False)),
            enable_sensor_noise=bool(lidar_cfg.get("enable_sensor_noise", False)),
        )

        self.lidar_sensor = LidarSensor(
            env=lidar_env_data,
            env_cfg={},
            sensor_config=lidar_sensor_config,
            num_sensors=1,
            device=self.wp_device,
        )
        self._install_training_lidar_rays()
        self._last_lidar_scan = None
        self._last_lidar_dist = None
        self._last_lidar_points_local = None
        self._last_lidar_sensor_pos = None
        self._last_lidar_sensor_rot = None

    # --- Mesh construction helpers (from drones_nav_genesis) ---
    @staticmethod
    def _build_ground_mesh(half_extent):
        extent = float(half_extent)
        vertices = np.array(
            [[-extent, -extent, 0.0], [extent, -extent, 0.0],
             [extent, extent, 0.0], [-extent, extent, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        return vertices, faces

    @staticmethod
    def _build_box_mesh(center_xyz, size_xyz):
        cx, cy, cz = center_xyz
        hx, hy, hz = [0.5 * float(s) for s in size_xyz]
        vertices = np.array([
            [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ], dtype=np.int32)
        return vertices, faces

    @staticmethod
    def _build_cylinder_mesh(center_xyz, radius, height, segments=16):
        cx, cy, cz = center_xyz
        r, h = float(radius), float(height)
        hz = 0.5 * h
        segments = max(8, int(segments))
        vertices = np.zeros((2 + 2 * segments, 3), dtype=np.float32)
        vertices[0] = [cx, cy, cz - hz]
        vertices[1] = [cx, cy, cz + hz]
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            x, y = cx + r * np.cos(theta), cy + r * np.sin(theta)
            vertices[2 + 2 * i] = [x, y, cz - hz]
            vertices[2 + 2 * i + 1] = [x, y, cz + hz]
        faces = []
        for i in range(segments):
            j = (i + 1) % segments
            b_i, t_i = 2 + 2 * i, 2 + 2 * i + 1
            b_j, t_j = 2 + 2 * j, 2 + 2 * j + 1
            faces.append([b_i, b_j, t_j])
            faces.append([b_i, t_j, t_i])
            faces.append([0, b_j, b_i])
            faces.append([1, t_i, t_j])
        return vertices, np.asarray(faces, dtype=np.int32)

    def _build_lidar_mesh(self):
        all_verts, all_faces, v_offset = [], [], 0
        ground_v, ground_f = self._build_ground_mesh(100.0)
        all_verts.append(ground_v)
        all_faces.append(ground_f)
        v_offset += len(ground_v)

        cyl_cfg = self.cfg.get("obstacles", {}).get("static_cylinders", {})
        if bool(cyl_cfg.get("enable", False)):
            color = cyl_cfg.get("obstacle_color", [0.35, 0.35, 0.38])
            for c in cyl_cfg.get("cylinders", []):
                pos = c["pos"]
                radius = float(c.get("radius", 0.28))
                height = float(c.get("height", 2.6))
                cv, cf = self._build_cylinder_mesh(pos, radius, height)
                all_verts.append(cv)
                all_faces.append(cf + v_offset)
                v_offset += len(cv)

        return np.concatenate(all_verts), np.concatenate(all_faces)

    def _create_warp_mesh(self, vertices, faces):
        vertex_tensor = torch.tensor(vertices, device=self.device, dtype=torch.float32, requires_grad=False)
        self._lidar_vertex_tensor = vertex_tensor
        vertex_vec3_array = wp.from_torch(vertex_tensor, dtype=wp.vec3)
        face_indices = wp.from_numpy(faces.flatten().astype(np.int32), dtype=wp.int32, device=self.wp_device)
        mesh = wp.Mesh(points=vertex_vec3_array, indices=face_indices)
        mesh_ids = wp.array([mesh.id], dtype=wp.uint64)
        return mesh, mesh_ids

    @staticmethod
    def _quat_genesis_to_warp(genesis_quat):
        if genesis_quat.ndim == 1:
            return torch.stack([genesis_quat[1], genesis_quat[2], genesis_quat[3], genesis_quat[0]])
        return torch.stack([genesis_quat[:, 1], genesis_quat[:, 2], genesis_quat[:, 3], genesis_quat[:, 0]], dim=1)

    def _install_training_lidar_rays(self):
        h = torch.arange(self.lidar_hbeams, device=self.device, dtype=torch.float32)
        v = torch.arange(self.lidar_vbeams, device=self.device, dtype=torch.float32)
        az = 2.0 * math.pi * h / float(self.lidar_hbeams)
        if self.lidar_vbeams <= 1:
            elev = torch.zeros_like(v)
        else:
            elev = math.pi / 180.0 * (
                self.lidar_vfov[0]
                + (self.lidar_vfov[1] - self.lidar_vfov[0]) * v / float(self.lidar_vbeams - 1)
            )
        elev_grid, az_grid = torch.meshgrid(elev, az, indexing="ij")
        ray_vectors = torch.stack(
            [
                torch.cos(elev_grid) * torch.cos(az_grid),
                torch.cos(elev_grid) * torch.sin(az_grid),
                torch.sin(elev_grid),
            ],
            dim=-1,
        ).contiguous()
        self.lidar_sensor.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)
        self.lidar_sensor.graph = None

    def _lidar_yaw_only_rot(self):
        base_rot = quat_to_R(self.base_quat)
        fwd = base_rot[:, :, 0].clone()
        fwd[:, 2] = 0.0
        fwd = F.normalize(fwd, p=2, dim=-1, eps=1e-6)
        up = torch.zeros_like(fwd)
        up[:, 2] = 1.0
        left = torch.cross(up, fwd, dim=-1)
        return torch.stack([fwd, left, up], dim=-1)

    @staticmethod
    def _yaw_only_warp_quat(yaw_rot: torch.Tensor) -> torch.Tensor:
        yaw = torch.atan2(yaw_rot[:, 1, 0], yaw_rot[:, 0, 0])
        half = 0.5 * yaw
        quat = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
        quat[:, 2] = torch.sin(half)
        quat[:, 3] = torch.cos(half)
        return quat

    def _compute_lidar_sensor_pose(self):
        base_rot = self._lidar_yaw_only_rot()
        sensor_offset = self.lidar_sensor_offset.view(1, 3, 1).expand(self.num_agents, -1, -1)
        sensor_pos = self.base_pos + torch.bmm(base_rot, sensor_offset).squeeze(-1)
        sensor_quat = self._yaw_only_warp_quat(base_rot)
        return sensor_pos, sensor_quat, base_rot

    def get_lidar(self) -> torch.Tensor:
        sensor_pos, sensor_quat, sensor_rot = self._compute_lidar_sensor_pose()
        self.lidar_sensor.lidar_positions_tensor[:] = sensor_pos
        self.lidar_sensor.lidar_quat_tensor[:] = sensor_quat
        points_local, distances = self.lidar_sensor.update()
        dist = distances[:, 0, :, :]
        scan = self.lidar_range - dist.clamp(0.0, self.lidar_range)
        scan = scan.clamp(0.0, self.lidar_range)
        scan = scan.permute(0, 2, 1).unsqueeze(1)
        self._last_lidar_scan = scan
        self._last_lidar_dist = dist
        self._last_lidar_points_local = points_local[:, 0, :, :, :]
        self._last_lidar_sensor_pos = sensor_pos
        self._last_lidar_sensor_rot = sensor_rot
        return scan

    def get_lidar_debug_points(
        self,
        max_points: int = 720,
        min_dist: float = 0.1,
        agent_idx: int | None = None,
    ) -> torch.Tensor:
        if (
            self._last_lidar_dist is None
            or self._last_lidar_points_local is None
            or self._last_lidar_sensor_pos is None
            or self._last_lidar_sensor_rot is None
        ):
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)

        if agent_idx is None:
            agent_indices = range(self.num_agents)
        else:
            if agent_idx < 0 or agent_idx >= self.num_agents:
                raise IndexError(f"agent_idx must be in [0, {self.num_agents}), got {agent_idx}")
            agent_indices = [agent_idx]

        points_world = []
        for i in agent_indices:
            dist = self._last_lidar_dist[i]
            valid = (dist > float(min_dist)) & (dist < self.lidar_range)
            if not bool(valid.any().item()):
                continue
            points_local = self._last_lidar_points_local[i][valid]
            points_world.append(
                points_local @ self._last_lidar_sensor_rot[i].T + self._last_lidar_sensor_pos[i]
            )

        if not points_world:
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)

        points = torch.cat(points_world, dim=0)
        max_points = int(max_points)
        if max_points > 0 and points.shape[0] > max_points:
            sample_idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
            points = points[sample_idx]
        return points

    def _add_obstacles(self):
        cyl_cfg = self.cfg.get("obstacles", {}).get("static_cylinders", {})
        if not bool(cyl_cfg.get("enable", False)):
            return
        color = cyl_cfg.get("obstacle_color", [0.35, 0.35, 0.38])
        for c in cyl_cfg.get("cylinders", []):
            pos = c["pos"]
            radius = float(c.get("radius", 0.28))
            height = float(c.get("height", 2.6))
            obs = self.scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=radius,
                    height=height,
                    pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                    fixed=True,
                    collision=True,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=tuple(float(x) for x in color)),
                ),
            )
            self.static_obstacles.append(obs)

    @staticmethod
    def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
        half = 0.5 * yaw
        q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=torch.float32)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    def _update_target_visual(self):
        for i, target in enumerate(self.targets):
            target.set_pos(self.target_pos[i : i + 1], zero_velocity=True, envs_idx=self._env0)

    def _update_state(self):
        for i, drone in enumerate(self.drones):
            self.base_pos[i] = drone.get_pos()[0]
            self.base_quat[i] = drone.get_quat()[0]
            self.base_lin_vel[i] = drone.get_vel()[0]
            self.base_ang_vel[i] = drone.get_ang()[0]

    def reset_episode(self, start_pos: torch.Tensor, yaw: torch.Tensor, target_pos: torch.Tensor):
        if start_pos.shape != (self.num_agents, 3):
            raise ValueError(f"start_pos shape must be ({self.num_agents}, 3), got {tuple(start_pos.shape)}")
        if target_pos.shape != (self.num_agents, 3):
            raise ValueError(f"target_pos shape must be ({self.num_agents}, 3), got {tuple(target_pos.shape)}")
        if yaw.shape != (self.num_agents,):
            raise ValueError(f"yaw shape must be ({self.num_agents},), got {tuple(yaw.shape)}")

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

    def step(self, rpm_cmd: torch.Tensor, record_frame: bool = False):
        if rpm_cmd.shape != (self.num_agents, 4):
            raise ValueError(f"rpm_cmd shape must be ({self.num_agents}, 4), got {tuple(rpm_cmd.shape)}")

        for i, drone in enumerate(self.drones):
            drone.set_propellels_rpm(rpm_cmd[i : i + 1])

        self._update_target_visual()
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

        timeout = torch.full(
            (self.num_agents,), self.step_count >= self.max_steps,
            device=self.device, dtype=torch.bool,
        )

        collision_flags = torch.zeros((self.num_agents,), device=self.device, dtype=torch.bool)
        collision_pairs = []
        if self.num_agents > 1:
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    d = torch.norm(self.base_pos[i] - self.base_pos[j]).item()
                    if d < self.inter_drone_collision_dist:
                        collision_flags[i] = True
                        collision_flags[j] = True
                        collision_pairs.append((i, j))

        done = reached | crash | timeout
        info = {
            "reached": reached, "crash": crash,
            "crash_height": crash_height, "crash_attitude": crash_attitude,
            "crash_bounds": crash_bounds, "timeout": timeout,
            "collision": collision_flags, "collision_pairs": collision_pairs, "done": done,
        }
        return info

    def start_recording(self):
        if self.record_cam is not None:
            self.record_cam.start_recording()

    def stop_recording(self, filename: str, fps: int = 60):
        if self.record_cam is not None:
            self.record_cam.stop_recording(save_to_filename=filename, fps=fps)
