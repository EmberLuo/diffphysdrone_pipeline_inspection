#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import select
import sys
import termios
import tty

import rospy
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand


def wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value, low, high):
    return max(low, min(high, value))


def yaw_from_quat(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class KeyboardTeleopPosCmd(object):
    def __init__(self):
        rospy.init_node("keyboard_teleop_pos_cmd", anonymous=True)

        self.cmd_topic = rospy.get_param("~cmd_topic", "/planning/pos_cmd")
        self.odom_topic = rospy.get_param("~odom_topic", "/mavros/local_position/odom")
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.step_xy = float(rospy.get_param("~step_xy", 0.15))
        self.step_z = float(rospy.get_param("~step_z", 0.10))
        self.step_yaw = float(rospy.get_param("~step_yaw", 0.10))
        self.max_z = float(rospy.get_param("~max_z", 3.0))
        self.min_z = float(rospy.get_param("~min_z", 0.2))

        self.pub = rospy.Publisher(self.cmd_topic, PositionCommand, queue_size=20)
        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=10)

        self.odom_ok = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_yaw = 0.0
        self.target_initialized = False

    def odom_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        self.current_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.odom_ok = True

        if not self.target_initialized:
            self.target_x = self.current_x
            self.target_y = self.current_y
            self.target_z = clamp(self.current_z, self.min_z, self.max_z)
            self.target_yaw = self.current_yaw
            self.target_initialized = True
            rospy.loginfo(
                "Keyboard teleop target initialized at x=%.2f y=%.2f z=%.2f yaw=%.2f",
                self.target_x,
                self.target_y,
                self.target_z,
                self.target_yaw,
            )

    def print_help(self):
        text = """
Keyboard Teleop (publishing quadrotor_msgs/PositionCommand)
  W/S : body forward/backward
  A/D : body left/right
  R/F : up/down
  Q/E : yaw left/right
  Space: emergency hover (freeze at current odom)
  H   : help
  X   : exit

Notes:
  - This node only publishes /planning/pos_cmd.
  - OFFBOARD/ARM/LAND must still be done by your existing scripts.
  - Running with exploration_manager in parallel uses "recent message wins".
"""
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def move_body(self, dx_body, dy_body):
        c = math.cos(self.current_yaw)
        s = math.sin(self.current_yaw)
        dx_map = c * dx_body - s * dy_body
        dy_map = s * dx_body + c * dy_body
        self.target_x += dx_map
        self.target_y += dy_map

    def update_target_from_key(self, key):
        key = key.lower()
        if key == "w":
            self.move_body(self.step_xy, 0.0)
        elif key == "s":
            self.move_body(-self.step_xy, 0.0)
        elif key == "a":
            self.move_body(0.0, self.step_xy)
        elif key == "d":
            self.move_body(0.0, -self.step_xy)
        elif key == "r":
            self.target_z = clamp(self.target_z + self.step_z, self.min_z, self.max_z)
        elif key == "f":
            self.target_z = clamp(self.target_z - self.step_z, self.min_z, self.max_z)
        elif key == "q":
            self.target_yaw = wrap_pi(self.target_yaw + self.step_yaw)
        elif key == "e":
            self.target_yaw = wrap_pi(self.target_yaw - self.step_yaw)
        elif key == " ":
            self.target_x = self.current_x
            self.target_y = self.current_y
            self.target_z = clamp(self.current_z, self.min_z, self.max_z)
            self.target_yaw = self.current_yaw
            rospy.logwarn("Emergency hover: freezing target at current odom pose.")
        elif key == "h":
            self.print_help()
        elif key == "x":
            rospy.loginfo("Keyboard teleop exit requested.")
            return False
        else:
            return True

        rospy.loginfo(
            "Target -> x=%.2f y=%.2f z=%.2f yaw=%.2f",
            self.target_x,
            self.target_y,
            self.target_z,
            self.target_yaw,
        )
        return True

    def publish_cmd(self):
        cmd = PositionCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "map"
        cmd.position.x = self.target_x
        cmd.position.y = self.target_y
        cmd.position.z = self.target_z
        cmd.velocity.x = 0.0
        cmd.velocity.y = 0.0
        cmd.velocity.z = 0.0
        cmd.acceleration.x = 0.0
        cmd.acceleration.y = 0.0
        cmd.acceleration.z = 0.0
        cmd.jerk.x = 0.0
        cmd.jerk.y = 0.0
        cmd.jerk.z = 0.0
        cmd.yaw = self.target_yaw
        cmd.yaw_dot = 0.0
        cmd.kx = [0.0, 0.0, 0.0]
        cmd.kv = [0.0, 0.0, 0.0]
        cmd.trajectory_id = 0
        cmd.trajectory_flag = getattr(PositionCommand, "TRAJECTORY_STATUS_READY", 1)
        self.pub.publish(cmd)

    def run(self):
        self.print_help()
        rospy.loginfo("Waiting for odometry on %s ...", self.odom_topic)

        while not rospy.is_shutdown() and not self.target_initialized:
            rospy.sleep(0.05)

        if rospy.is_shutdown():
            return

        rate = rospy.Rate(self.publish_rate)
        rospy.loginfo("Keyboard teleop started. Publishing to %s", self.cmd_topic)

        stdin_fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)

        try:
            keep_running = True
            while not rospy.is_shutdown() and keep_running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                if ready:
                    key = sys.stdin.read(1)
                    keep_running = self.update_target_from_key(key)

                self.publish_cmd()
                rate.sleep()
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)


if __name__ == "__main__":
    try:
        KeyboardTeleopPosCmd().run()
    except rospy.ROSInterruptException:
        pass
