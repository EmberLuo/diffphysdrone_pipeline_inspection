import torch


class QuadXMixer:
    def __init__(
        self,
        roll_p,
        roll_i,
        roll_d,
        pitch_p,
        pitch_i,
        pitch_d,
        yaw_p,
        yaw_i,
        yaw_d,
        num_envs: int,
        device="cuda",
    ):
        self.roll_p = roll_p
        self.roll_i = roll_i
        self.roll_d = roll_d
        self.pitch_p = pitch_p
        self.pitch_i = pitch_i
        self.pitch_d = pitch_d
        self.yaw_p = yaw_p
        self.yaw_i = yaw_i
        self.yaw_d = yaw_d

        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        self.roll_vel_err_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.pitch_vel_err_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.yaw_vel_err_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.last_roll_vel_err = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_pitch_vel_err = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_yaw_vel_err = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.mixer_matrix = torch.tensor(
            [
                [1.0, -1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            self.roll_vel_err_sum.zero_()
            self.pitch_vel_err_sum.zero_()
            self.yaw_vel_err_sum.zero_()
            self.last_roll_vel_err.zero_()
            self.last_pitch_vel_err.zero_()
            self.last_yaw_vel_err.zero_()
            return

        self.roll_vel_err_sum[env_ids] = 0.0
        self.pitch_vel_err_sum[env_ids] = 0.0
        self.yaw_vel_err_sum[env_ids] = 0.0
        self.last_roll_vel_err[env_ids] = 0.0
        self.last_pitch_vel_err[env_ids] = 0.0
        self.last_yaw_vel_err[env_ids] = 0.0

    def pid_attitude_command_for_mix(
        self,
        roll_vel_r,
        pitch_vel_r,
        yaw_vel_r,
        roll_vel_d,
        pitch_vel_d,
        yaw_vel_d,
        dt,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        roll_vel_err = (roll_vel_d - roll_vel_r).to(torch.float32)
        pitch_vel_err = (pitch_vel_d - pitch_vel_r).to(torch.float32)
        yaw_vel_err = (yaw_vel_d - yaw_vel_r).to(torch.float32)

        self.roll_vel_err_sum += roll_vel_err
        self.pitch_vel_err_sum += pitch_vel_err
        self.yaw_vel_err_sum += yaw_vel_err

        roll_derivative = (roll_vel_err - self.last_roll_vel_err) / dt
        pitch_derivative = (pitch_vel_err - self.last_pitch_vel_err) / dt
        yaw_derivative = (yaw_vel_err - self.last_yaw_vel_err) / dt

        roll_pid = (
            self.roll_p * roll_vel_err
            + self.roll_i * self.roll_vel_err_sum * dt
            + self.roll_d * roll_derivative
        )
        pitch_pid = (
            self.pitch_p * pitch_vel_err
            + self.pitch_i * self.pitch_vel_err_sum * dt
            + self.pitch_d * pitch_derivative
        )
        yaw_pid = (
            self.yaw_p * yaw_vel_err
            + self.yaw_i * self.yaw_vel_err_sum * dt
            + self.yaw_d * yaw_derivative
        )

        self.last_roll_vel_err.copy_(roll_vel_err)
        self.last_pitch_vel_err.copy_(pitch_vel_err)
        self.last_yaw_vel_err.copy_(yaw_vel_err)

        return roll_pid, pitch_pid, yaw_pid

    def mix(self, throttle, roll_pid, pitch_pid, yaw_pid):
        controls = torch.stack([throttle, roll_pid, pitch_pid, yaw_pid])
        motor_speed = torch.matmul(self.mixer_matrix, controls)
        motor_speed = motor_speed.t()
        return torch.clamp(motor_speed, 0, 1)
