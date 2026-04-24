import math
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


class SwarmEnv:
    def __init__(self, cfg: dict, show_viewer: bool = True, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)

        sim_cfg = cfg["sim"]
        scene_cfg = cfg["scene"]
        task_cfg = cfg["task"]
        depth_cfg = cfg["depth_camera"]
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

        agents_cfg = task_cfg["layouts"][task_cfg.get("default_env", "swap")]["agents"]
        self.agent_names = [a["name"] for a in agents_cfg]
        self.num_agents = len(self.agent_names)

        self.vehicle_params = _read_vehicle_params_from_urdf(DRONE_URDF)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=int(sim_cfg.get("substeps", 2))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(scene_cfg.get("max_visualize_fps", 60)),
                camera_pos=tuple(scene_cfg.get("viewer_camera_pos", [3.0, 8.0, 5.0])),
                camera_lookat=tuple(scene_cfg.get("viewer_camera_lookat", [3.0, 0.0, 1.5])),
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
                diffuse_texture=gs.textures.ColorTexture(color=tuple(scene_cfg.get("plane_color", [0.7, 0.7, 0.7]))),
            ),
        )
        self.static_obstacles = []
        self._add_wall_hole_obstacles()

        self.drones = []
        self.targets = []
        self.depth_sensors = []

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
                    scale=float(task_cfg.get("target_sphere_scale", 0.04)),
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
                pos=tuple(rec_cfg.get("pos", [3.0, 8.0, 5.0])),
                lookat=tuple(rec_cfg.get("lookat", [3.0, 0.0, 1.5])),
                fov=float(rec_cfg.get("fov", 45)),
                GUI=False,
            )

        self.scene.build(n_envs=1)
        self._env0 = torch.tensor([0], device=self.device, dtype=gs.tc_int)

        self.base_pos = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_agents, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.target_pos = torch.zeros((self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.step_count = 0

    def _add_box_obstacle(self, pos, size, color):
        if size[0] <= 1e-6 or size[1] <= 1e-6 or size[2] <= 1e-6:
            return
        obs = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=tuple(float(x) for x in pos),
                size=tuple(float(x) for x in size),
                fixed=True,
                collision=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=tuple(float(x) for x in color)),
            ),
        )
        self.static_obstacles.append(obs)

    def _add_wall_hole_obstacles(self):
        ocfg = self.cfg.get("obstacles", {})
        if not bool(ocfg.get("enable_wall_holes", False)):
            return

        color = ocfg.get("wall_color", [0.35, 0.35, 0.38])
        walls = ocfg.get("wall_holes", [])
        for w in walls:
            x = float(w.get("x", 3.0))
            y_center = float(w.get("y_center", 0.0))
            z_center = float(w.get("z_center", 1.3))
            thickness = float(w.get("thickness", 0.12))
            wall_span_y = float(w.get("wall_span_y", 4.8))
            wall_span_z = float(w.get("wall_span_z", 2.6))
            hole_span_y = float(w.get("hole_span_y", 1.5))
            hole_span_z = float(w.get("hole_span_z", 1.2))

            side_y = max(0.0, (wall_span_y - hole_span_y) * 0.5)
            top_bot_z = max(0.0, (wall_span_z - hole_span_z) * 0.5)

            if side_y > 1e-6:
                y_off = hole_span_y * 0.5 + side_y * 0.5
                self._add_box_obstacle(
                    pos=(x, y_center - y_off, z_center),
                    size=(thickness, side_y, wall_span_z),
                    color=color,
                )
                self._add_box_obstacle(
                    pos=(x, y_center + y_off, z_center),
                    size=(thickness, side_y, wall_span_z),
                    color=color,
                )

            if top_bot_z > 1e-6:
                z_off = hole_span_z * 0.5 + top_bot_z * 0.5
                self._add_box_obstacle(
                    pos=(x, y_center, z_center - z_off),
                    size=(thickness, hole_span_y, top_bot_z),
                    color=color,
                )
                self._add_box_obstacle(
                    pos=(x, y_center, z_center + z_off),
                    size=(thickness, hole_span_y, top_bot_z),
                    color=color,
                )

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
        # One physics tick is needed so the depth sensors have valid first-frame readings.
        # Immediately clear velocities afterward to avoid a synthetic free-fall bias in step-0 state.
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
            (self.num_agents,),
            self.step_count >= self.max_steps,
            device=self.device,
            dtype=torch.bool,
        )

        collision_flags, collision_pairs = self._pairwise_collisions()
        done = reached | crash | timeout

        info = {
            "reached": reached,
            "crash": crash,
            "crash_height": crash_height,
            "crash_attitude": crash_attitude,
            "crash_bounds": crash_bounds,
            "timeout": timeout,
            "collision": collision_flags,
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
