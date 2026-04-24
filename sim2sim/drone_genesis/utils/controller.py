import math

import torch
from genesis.utils.geom import quat_to_R

from .mixer import QuadXMixer


class PX4StyleRPMController:
    def __init__(self, cfg: dict, num_envs: int, device: torch.device, vehicle_params: dict | None = None):
        self.device = device
        self.num_envs = num_envs

        pid_cfg = cfg["pid"]
        self.mixer = QuadXMixer(
            roll_p=pid_cfg["roll"]["p"],
            roll_i=pid_cfg["roll"]["i"],
            roll_d=pid_cfg["roll"]["d"],
            pitch_p=pid_cfg["pitch"]["p"],
            pitch_i=pid_cfg["pitch"]["i"],
            pitch_d=pid_cfg["pitch"]["d"],
            yaw_p=pid_cfg["yaw"]["p"],
            yaw_i=pid_cfg["yaw"]["i"],
            yaw_d=pid_cfg["yaw"]["d"],
            num_envs=num_envs,
            device=device,
        )

        self.kp_angle = torch.tensor(cfg.get("kp_angle", [4.0, 4.0, 2.0]), device=device, dtype=torch.float32)
        self.max_rate = torch.tensor(cfg.get("max_rate", [15.0, 15.0, 5.0]), device=device, dtype=torch.float32)
        self.attitude_mix_scale = float(cfg.get("attitude_mix_scale", 0.2))

        self.g = abs(float(cfg.get("gravity", 9.80665)))
        use_urdf_params = bool(cfg.get("use_urdf_params", True))
        vp = vehicle_params if (use_urdf_params and vehicle_params is not None) else {}

        self.mass = float(cfg.get("mass", vp.get("mass", 0.3)))
        self.kf = float(cfg.get("kf", vp.get("kf", 3.16e-10)))
        self.km = float(cfg.get("km", vp.get("km", 7.94e-12)))
        default_twr = vp.get("thrust2weight", 2.25)
        self.twr_max = float(cfg.get("twr_max", default_twr))

        self.rpm_min = float(cfg.get("rpm_min", 0.0))
        self.rpm_max = float(cfg.get("rpm_max", 0.0))
        self.hover_rpm_trim = float(cfg.get("hover_rpm_trim", 1.0))
        # AirSim moveByRollPitchYawThrottleAsync uses a normalized throttle where hover is ~0.297.
        self.hover_throttle = float(cfg.get("hover_throttle", 0.297))

        if self.kf <= 0:
            raise ValueError(f"Invalid kf: {self.kf}.")
        if self.twr_max <= 1.0:
            raise ValueError(f"twr_max must be > 1.0, got {self.twr_max}.")

        # For Genesis drone model: total thrust = sum_i (kf * rpm_i^2)
        self.hover_rpm = math.sqrt(self.mass * self.g / (4.0 * self.kf)) * self.hover_rpm_trim
        self.rpm_factor = self.hover_rpm * math.sqrt(self.twr_max)
        if self.rpm_max <= 0.0:
            self.rpm_max = self.rpm_factor

    def reset(self, env_ids=None):
        self.mixer.reset(env_ids)

    @staticmethod
    def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        return torch.remainder(x + math.pi, two_pi) - math.pi

    @staticmethod
    def _rpy_to_rotmat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Build world-from-body rotation matrix from roll/pitch/yaw."""
        sr, cr = torch.sin(roll), torch.cos(roll)
        sp, cp = torch.sin(pitch), torch.cos(pitch)
        sy, cy = torch.sin(yaw), torch.cos(yaw)

        rot = torch.zeros((roll.shape[0], 3, 3), device=roll.device, dtype=roll.dtype)
        rot[:, 0, 0] = cy * cp
        rot[:, 0, 1] = cy * sp * sr - sy * cr
        rot[:, 0, 2] = cy * sp * cr + sy * sr
        rot[:, 1, 0] = sy * cp
        rot[:, 1, 1] = sy * sp * sr + cy * cr
        rot[:, 1, 2] = sy * sp * cr - cy * sr
        rot[:, 2, 0] = -sp
        rot[:, 2, 1] = cp * sr
        rot[:, 2, 2] = cp * cr
        return rot

    @staticmethod
    def _attitude_error_body(rot_cur: torch.Tensor, rot_des: torch.Tensor) -> torch.Tensor:
        """SO(3) attitude error in body frame, robust near yaw wrap boundaries."""
        rot_err = torch.matmul(rot_cur.transpose(1, 2), rot_des)
        skew = 0.5 * (rot_err - rot_err.transpose(1, 2))
        return torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)

    def compute_rpm(
        self,
        a_cmd_world: torch.Tensor,
        base_quat: torch.Tensor,
        base_ang_vel: torch.Tensor,
        dt: float,
        yaw_des: torch.Tensor | None = None,
    ):
        # a_cmd_world is treated as desired net acceleration (gravity removed).
        # Convert to thrust acceleration by adding gravity back.
        g_vec = torch.tensor([0.0, 0.0, -self.g], device=self.device).expand_as(a_cmd_world)
        thrust_vec = a_cmd_world - g_vec
        thrust_norm = torch.norm(thrust_vec, dim=-1).clamp_min(1e-4)

        # Desired body up-axis in world frame.
        z_b_des = thrust_vec / thrust_norm[:, None]
        z_b_des = torch.nn.functional.normalize(z_b_des, dim=-1)

        # Desired Euler from desired body-z direction, yaw fixed at 0.
        # Sign convention aligned with Genesis world/body axis directions.
        # Positive commanded lateral acceleration should produce motion in the same world-axis direction.
        roll_des = -torch.atan2(z_b_des[:, 1], z_b_des[:, 2].clamp_min(1e-4))
        pitch_des = torch.atan2(z_b_des[:, 0], torch.sqrt(z_b_des[:, 1] ** 2 + z_b_des[:, 2] ** 2).clamp_min(1e-4))
        if yaw_des is None:
            yaw_des = torch.zeros_like(roll_des)
        else:
            yaw_des = yaw_des.to(device=self.device, dtype=torch.float32)
        rotmat = quat_to_R(base_quat)
        rot_des = self._rpy_to_rotmat(roll_des, pitch_des, yaw_des)
        # Genesis returns angular velocity in world frame; convert to body frame for body-rate PID.
        base_ang_vel_body = torch.squeeze(base_ang_vel[:, None] @ rotmat, 1)

        # Angle P controller in SO(3) to desired body rates.
        att_err_body = self._attitude_error_body(rotmat, rot_des)
        rate_des = att_err_body * self.kp_angle
        rate_des = torch.clamp(rate_des, -self.max_rate, self.max_rate)

        roll_pid, pitch_pid, yaw_pid = self.mixer.pid_attitude_command_for_mix(
            roll_vel_r=base_ang_vel_body[:, 0],
            pitch_vel_r=base_ang_vel_body[:, 1],
            yaw_vel_r=base_ang_vel_body[:, 2],
            roll_vel_d=rate_des[:, 0],
            pitch_vel_d=rate_des[:, 1],
            yaw_vel_d=rate_des[:, 2],
            dt=dt,
        )

        # Convert acceleration magnitude to collective command.
        thrust_ratio = torch.clamp(thrust_norm / self.g, 0.0, self.twr_max)
        throttle = torch.sqrt(thrust_ratio / self.twr_max)
        throttle = torch.clamp(throttle, 0.0, 1.0)

        motor_cmd = self.mixer.mix(
            throttle,
            roll_pid * self.attitude_mix_scale,
            pitch_pid * self.attitude_mix_scale,
            yaw_pid * self.attitude_mix_scale,
        )
        rpm = torch.clamp(motor_cmd * self.rpm_factor, self.rpm_min, self.rpm_max)
        return rpm

    def compute_rpm_from_rpy_throttle(
        self,
        roll_des: torch.Tensor,
        pitch_des: torch.Tensor,
        yaw_des: torch.Tensor,
        throttle_des: torch.Tensor,
        base_quat: torch.Tensor,
        base_ang_vel: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Map desired roll/pitch/yaw + AirSim-style throttle to motor RPM.

        The pipeline mirrors the reference Genesis env:
        angle error -> desired body rates -> body-rate PID -> QuadX mixer -> RPM.
        """
        roll_des = roll_des.to(device=self.device, dtype=torch.float32)
        pitch_des = pitch_des.to(device=self.device, dtype=torch.float32)
        yaw_des = yaw_des.to(device=self.device, dtype=torch.float32)
        throttle_des = throttle_des.to(device=self.device, dtype=torch.float32)

        rotmat = quat_to_R(base_quat)
        rot_des = self._rpy_to_rotmat(roll_des, pitch_des, yaw_des)
        # Genesis returns angular velocity in world frame; convert to body frame for body-rate PID.
        base_ang_vel_body = torch.squeeze(base_ang_vel[:, None] @ rotmat, 1)

        att_err_body = self._attitude_error_body(rotmat, rot_des)
        rate_des = att_err_body * self.kp_angle
        rate_des = torch.clamp(rate_des, -self.max_rate, self.max_rate)

        roll_pid, pitch_pid, yaw_pid = self.mixer.pid_attitude_command_for_mix(
            roll_vel_r=base_ang_vel_body[:, 0],
            pitch_vel_r=base_ang_vel_body[:, 1],
            yaw_vel_r=base_ang_vel_body[:, 2],
            roll_vel_d=rate_des[:, 0],
            pitch_vel_d=rate_des[:, 1],
            yaw_vel_d=rate_des[:, 2],
            dt=dt,
        )

        # Convert AirSim normalized throttle (~0.297 at hover) to thrust ratio, then to mixer throttle.
        hover_thr = max(1e-4, self.hover_throttle)
        thrust_ratio = torch.clamp(throttle_des / hover_thr, 0.0, self.twr_max)
        throttle_mix = torch.sqrt(thrust_ratio / self.twr_max)
        throttle_mix = torch.clamp(throttle_mix, 0.0, 1.0)

        motor_cmd = self.mixer.mix(
            throttle_mix,
            roll_pid * self.attitude_mix_scale,
            pitch_pid * self.attitude_mix_scale,
            yaw_pid * self.attitude_mix_scale,
        )
        rpm = torch.clamp(motor_cmd * self.rpm_factor, self.rpm_min, self.rpm_max)
        return rpm
