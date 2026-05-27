#!/usr/bin/env python3
import importlib.util
import math
import os
from pathlib import Path
import time

import numpy as np
import rospy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from geometry_msgs.msg import PointStamped, Quaternion
from mavros_msgs.msg import AttitudeTarget, State
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


def _load_package_module(module_name, filename):
    package_src = Path(__file__).resolve().parents[1] / "src" / "e2e_px4_controller" / filename
    spec = importlib.util.spec_from_file_location(f"_e2e_px4_controller_{module_name}", package_src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {package_src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Model = _load_package_module("model", "model.py").Model
_rotation = _load_package_module("rotation", "rotation.py")
matrix_to_quaternion = _rotation.matrix_to_quaternion
quaternion_to_matrix = _rotation.quaternion_to_matrix


class E2EPx4Controller:
    def __init__(self):
        self.depth_topic = rospy.get_param("~depth_topic", "/e2e/depth/image_raw")
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.target_topic = rospy.get_param("~target_topic", "/e2e/local_target")
        self.setpoint_topic = rospy.get_param("~setpoint_topic", "/mavros/setpoint_raw/attitude")
        self.net_weight = rospy.get_param("~net_weight", "")
        self.target_speed = float(rospy.get_param("~target_speed", 1.0))
        self.margin = float(rospy.get_param("~margin", 0.2))
        self.no_odom = bool(rospy.get_param("~no_odom", False))
        self.hover_percent = float(rospy.get_param("~hover_percent", 0.25))
        self.max_hover_percent = float(rospy.get_param("~max_hover_percent", 0.65))
        self.min_thrust = float(rospy.get_param("~min_thrust", 0.0))
        self.min_target_distance = float(rospy.get_param("~min_target_distance", 0.3))
        self.max_depth_m = float(rospy.get_param("~max_depth_m", 24.0))
        self.stale_timeout = float(rospy.get_param("~stale_timeout", 0.5))
        self.depth_stale_timeout = float(rospy.get_param("~depth_stale_timeout", 0.7))
        self.prewarm_setpoints = bool(rospy.get_param("~prewarm_setpoints", True))
        self.gate_on_px4_state = bool(rospy.get_param("~gate_on_px4_state", True))
        self.require_armed_for_policy = bool(rospy.get_param("~require_armed_for_policy", True))
        self.require_offboard_for_policy = bool(rospy.get_param("~require_offboard_for_policy", True))
        self.state_topic = rospy.get_param("~state_topic", "/mavros/state")
        self.estop_topic = rospy.get_param("~estop_topic", "/e2e/estop")
        self.safety_rate = float(rospy.get_param("~safety_rate", 20.0))
        self.geofence_enabled = bool(rospy.get_param("~geofence_enabled", True))
        self.geofence_x_min = float(rospy.get_param("~geofence_x_min", -100.0))
        self.geofence_x_max = float(rospy.get_param("~geofence_x_max", 100.0))
        self.geofence_y_min = float(rospy.get_param("~geofence_y_min", -100.0))
        self.geofence_y_max = float(rospy.get_param("~geofence_y_max", 100.0))
        self.geofence_z_min = float(rospy.get_param("~geofence_z_min", 0.1))
        self.geofence_z_max = float(rospy.get_param("~geofence_z_max", 10.0))
        self.debug = bool(rospy.get_param("~debug", False))

        if not self.net_weight or not os.path.exists(self.net_weight):
            raise rospy.ROSException("~net_weight must point to an existing .pth file")

        self.dim_obs = 7 if self.no_odom else 10
        self.model = Model(self.dim_obs, 6).eval()
        state_dict = torch.load(self.net_weight, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        torch.set_grad_enabled(False)
        _, _, self.hidden = self.model(
            torch.zeros(1, 1, 12, 16),
            torch.zeros(1, self.dim_obs),
        )

        self.bridge = CvBridge()
        self.odom = None
        self.target = None
        self.target_stamp = rospy.Time(0)
        self.odom_stamp = rospy.Time(0)
        self.depth_stamp = rospy.Time(0)
        self.last_setpoint_stamp = rospy.Time(0)
        self.px4_state = None
        self.estop = False
        self.forward = None

        self.cmd_pub = rospy.Publisher(self.setpoint_topic, AttitudeTarget, queue_size=20)
        self.debug_acc_pub = rospy.Publisher("~accel_setpoint", PointStamped, queue_size=10)

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.target_topic, PointStamped, self.target_cb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=1)
        rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        rospy.Subscriber(self.estop_topic, Bool, self.estop_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.safety_rate), self.safety_timer_cb)

        rospy.loginfo("e2e_px4_controller ready: depth=%s odom=%s target=%s setpoint=%s",
                      self.depth_topic, self.odom_topic, self.target_topic, self.setpoint_topic)

    def odom_cb(self, msg):
        self.odom = msg
        self.odom_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def target_cb(self, msg):
        self.target = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float32)
        self.target_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def state_cb(self, msg):
        self.px4_state = msg

    def estop_cb(self, msg):
        self.estop = bool(msg.data)

    def depth_to_tensor(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(img)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.dtype == np.uint16 or depth.dtype == np.int16:
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m <= 0.0] = self.max_depth_m
        depth_m = 3.0 / np.clip(depth_m, 0.3, self.max_depth_m) - 0.6

        h, w = depth_m.shape
        crop_h = round((h - h * 0.82) / 2.0)
        crop_w = round((w - w * 0.82) / 2.0)
        if crop_h > 0 and crop_w > 0:
            depth_m = depth_m[crop_h:-crop_h, crop_w:-crop_w]
        tensor = torch.as_tensor(depth_m, dtype=torch.float32)[None, None]
        tensor = F.interpolate(tensor, (36, 48), mode="nearest")
        tensor = F.max_pool2d(tensor, (3, 3))
        return tensor

    def depth_cb(self, msg):
        self.depth_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        if self.odom is None or self.target is None:
            return
        now = rospy.Time.now()
        ok, reason = self.policy_ready(now)
        if not ok:
            rospy.logwarn_throttle(1.0, "%s; publishing safety hover", reason)
            self.publish_hover(now)
            return

        p_msg = self.odom.pose.pose.position
        q_msg = self.odom.pose.pose.orientation
        v_msg = self.odom.twist.twist.linear
        pos = torch.tensor([p_msg.x, p_msg.y, p_msg.z], dtype=torch.float32)
        quat = torch.tensor([q_msg.w, q_msg.x, q_msg.y, q_msg.z], dtype=torch.float32)
        vel = torch.tensor([v_msg.x, v_msg.y, v_msg.z], dtype=torch.float32)

        target = torch.as_tensor(self.target, dtype=torch.float32)
        target_vec = target - pos
        target_dist = torch.norm(target_vec)
        if target_dist < self.min_target_distance:
            return
        target_v = target_vec / target_dist * self.target_speed

        try:
            depth = self.depth_to_tensor(msg)
            cmd = self.infer_command(depth, quat, vel, target_v)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "E2E inference failed: %s", exc)
            self.publish_hover(now)
            return
        self.publish_setpoint(cmd)

    def policy_ready(self, now):
        if self.estop:
            return False, "E-stop active"
        if self.odom is None:
            return False, "No odometry"
        if self.target is None:
            return False, "No local target"
        if (now - self.odom_stamp).to_sec() > self.stale_timeout:
            return False, "Stale odometry"
        if (now - self.target_stamp).to_sec() > self.stale_timeout:
            return False, "Stale local target"
        if self.geofence_enabled and not self.inside_geofence():
            return False, "Outside geofence"
        if self.gate_on_px4_state and self.px4_state is not None:
            if self.require_armed_for_policy and not self.px4_state.armed:
                return False, "PX4 not armed"
            if self.require_offboard_for_policy and self.px4_state.mode != "OFFBOARD":
                return False, "PX4 not in OFFBOARD"
        return True, ""

    def inside_geofence(self):
        p = self.odom.pose.pose.position
        return (
            self.geofence_x_min <= p.x <= self.geofence_x_max
            and self.geofence_y_min <= p.y <= self.geofence_y_max
            and self.geofence_z_min <= p.z <= self.geofence_z_max
        )

    def safety_timer_cb(self, _event):
        now = rospy.Time.now()
        if (now - self.last_setpoint_stamp).to_sec() < 1.0 / max(self.safety_rate, 1.0):
            return
        ready, _reason = self.policy_ready(now)
        depth_fresh = (
            self.depth_stamp != rospy.Time(0)
            and (now - self.depth_stamp).to_sec() <= self.depth_stale_timeout
        )
        if ready and depth_fresh:
            return
        if self.prewarm_setpoints or not depth_fresh:
            self.publish_hover(now)

    def publish_setpoint(self, msg):
        self.last_setpoint_stamp = rospy.Time.now()
        self.cmd_pub.publish(msg)

    def publish_hover(self, now=None):
        msg = AttitudeTarget()
        msg.header.stamp = now if now is not None else rospy.Time.now()
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )
        msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        msg.thrust = max(self.min_thrust, min(self.max_hover_percent, self.hover_percent))
        self.publish_setpoint(msg)

    def infer_command(self, depth, quat, vel, target_v):
        rot = quaternion_to_matrix(quat)
        env_rot = rot.clone()
        fwd = rot[:, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[2] = 0.0
        up[2] = 1.0
        if torch.norm(fwd) < 1e-3:
            fwd = torch.tensor([1.0, 0.0, 0.0])
        fwd = fwd / torch.norm(fwd, 2, -1, keepdim=True)
        yaw_rot = torch.stack([fwd, torch.cross(up, fwd), up], -1)
        if self.forward is None:
            self.forward = yaw_rot[:, 0]

        global_v = vel @ env_rot.T
        state = [target_v[None] @ yaw_rot, env_rot[None, 2], torch.tensor([[self.margin]])]
        if not self.no_odom:
            state.insert(0, global_v[None] @ yaw_rot)
        state = torch.cat(state, -1)

        act, _, self.hidden = self.model(depth, state, self.hidden)
        a_pred, v_pred, *_ = (yaw_rot @ act.reshape(3, -1)).unbind(-1)
        a_pred = a_pred - v_pred
        a_debug = a_pred.clone()
        a_pred[2] += 9.81

        collective_accel = torch.norm(a_pred).item()
        up_vec = a_pred / max(collective_accel, 1e-3)
        self.forward = self.forward * 5.0 + target_v
        if abs(float(up_vec[2])) > 1e-3:
            self.forward[2] = (self.forward[0] * up_vec[0] + self.forward[1] * up_vec[1]) / -up_vec[2]
        self.forward /= torch.norm(self.forward, 2, -1, True)
        left_vec = torch.cross(up_vec, self.forward)
        if torch.norm(left_vec) < 1e-3:
            left_vec = torch.tensor([0.0, 1.0, 0.0])
        left_vec /= torch.norm(left_vec)
        w, x, y, z = matrix_to_quaternion(torch.stack([self.forward, left_vec, up_vec], 1)).tolist()

        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )
        msg.orientation = Quaternion(x=x, y=y, z=z, w=w)
        msg.thrust = self.normalize_thrust(collective_accel)

        if self.debug and self.debug_acc_pub.get_num_connections() > 0:
            dbg = PointStamped()
            dbg.header.stamp = msg.header.stamp
            dbg.header.frame_id = "world"
            dbg.point.x, dbg.point.y, dbg.point.z = [float(v) for v in a_debug]
            self.debug_acc_pub.publish(dbg)
        return msg

    def normalize_thrust(self, collective_accel):
        thrust = self.hover_percent * collective_accel / 9.81
        return max(self.min_thrust, min(self.max_hover_percent, thrust))


if __name__ == "__main__":
    rospy.init_node("e2e_px4_controller")
    E2EPx4Controller()
    rospy.spin()
