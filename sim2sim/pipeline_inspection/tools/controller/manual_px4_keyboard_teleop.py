#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


class ManualPx4KeyboardTeleop:
    def __init__(self, args):
        self.args = args
        rospy.init_node("manual_px4_keyboard_teleop", anonymous=True)

        self.state = State()
        self.pose = None
        self.pose_ok = False

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = args.takeoff_altitude
        self.target_yaw = 0.0
        self.target_initialized = False

        self.pub = rospy.Publisher(args.setpoint_topic, PoseStamped, queue_size=20)
        rospy.Subscriber(args.state_topic, State, self.state_cb, queue_size=10)
        rospy.Subscriber(args.pose_topic, PoseStamped, self.pose_cb, queue_size=10)

        self.arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        self.pose = msg
        self.pose_ok = True
        if self.target_initialized:
            return
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = clamp(msg.pose.position.z, self.args.min_z, self.args.max_z)
        if self.target_z < self.args.min_z + 1e-3:
            self.target_z = self.args.takeoff_altitude
        self.target_yaw = yaw_from_quat(msg.pose.orientation)
        self.target_initialized = True
        rospy.loginfo(
            "Target initialized: x=%.2f y=%.2f z=%.2f yaw=%.1f deg",
            self.target_x,
            self.target_y,
            self.target_z,
            math.degrees(self.target_yaw),
        )

    def print_help(self):
        print(
            """
PX4 Keyboard Mapping Teleop

Motion keys:
  W/S : forward/backward
  A/D : left/right
  R/F : up/down
  Q/E : yaw left/right
  T   : takeoff target at current XY and configured altitude
  Space: hold current vehicle pose

PX4 keys:
  O : request OFFBOARD
  M : arm
  U : disarm
  L : AUTO.LAND

Other:
  H : help
  X : exit

This node publishes PoseStamped setpoints to /mavros/setpoint_position/local.
Run offboard_preflight.sh first if PX4 rejects OFFBOARD/ARM without RC.
"""
        )

    def wait_for_mavros(self):
        rate = rospy.Rate(10)
        rospy.loginfo("Waiting for MAVROS connection on %s ...", self.args.state_topic)
        while not rospy.is_shutdown() and not self.state.connected:
            rate.sleep()
        rospy.loginfo("MAVROS connected")

    def request_mode(self, mode):
        try:
            result = self.mode_srv(base_mode=0, custom_mode=mode)
            if result.mode_sent:
                rospy.loginfo("Requested PX4 mode: %s", mode)
            else:
                rospy.logwarn("PX4 rejected mode request: %s", mode)
        except rospy.ServiceException as exc:
            rospy.logwarn("SetMode failed: %s", exc)

    def request_arm(self, value):
        try:
            result = self.arm_srv(value)
            if result.success:
                rospy.loginfo("%s requested", "ARM" if value else "DISARM")
            else:
                rospy.logwarn("%s request rejected", "ARM" if value else "DISARM")
        except rospy.ServiceException as exc:
            rospy.logwarn("Arming failed: %s", exc)

    def hold_current_pose(self):
        if not self.pose_ok:
            rospy.logwarn("No local pose yet; cannot hold current pose")
            return
        p = self.pose.pose.position
        self.target_x = p.x
        self.target_y = p.y
        self.target_z = clamp(p.z, self.args.min_z, self.args.max_z)
        self.target_yaw = yaw_from_quat(self.pose.pose.orientation)
        self.target_initialized = True
        rospy.loginfo("Holding current pose")

    def takeoff_target(self):
        if self.pose_ok:
            self.target_x = self.pose.pose.position.x
            self.target_y = self.pose.pose.position.y
            self.target_yaw = yaw_from_quat(self.pose.pose.orientation)
        self.target_z = clamp(self.args.takeoff_altitude, self.args.min_z, self.args.max_z)
        self.target_initialized = True
        rospy.loginfo("Takeoff target set to z=%.2f", self.target_z)

    def move_body(self, forward, left):
        yaw = self.target_yaw
        dx = math.cos(yaw) * forward - math.sin(yaw) * left
        dy = math.sin(yaw) * forward + math.cos(yaw) * left
        self.target_x += dx
        self.target_y += dy

    def update_from_key(self, key):
        key = key.lower()
        if key == "w":
            self.move_body(self.args.step_xy, 0.0)
        elif key == "s":
            self.move_body(-self.args.step_xy, 0.0)
        elif key == "a":
            self.move_body(0.0, self.args.step_xy)
        elif key == "d":
            self.move_body(0.0, -self.args.step_xy)
        elif key == "r":
            self.target_z = clamp(self.target_z + self.args.step_z, self.args.min_z, self.args.max_z)
        elif key == "f":
            self.target_z = clamp(self.target_z - self.args.step_z, self.args.min_z, self.args.max_z)
        elif key == "q":
            self.target_yaw = wrap_pi(self.target_yaw + self.args.step_yaw)
        elif key == "e":
            self.target_yaw = wrap_pi(self.target_yaw - self.args.step_yaw)
        elif key == "t":
            self.takeoff_target()
        elif key == " ":
            self.hold_current_pose()
        elif key == "o":
            self.request_mode("OFFBOARD")
            return True
        elif key == "m":
            self.request_arm(True)
            return True
        elif key == "u":
            self.request_arm(False)
            return True
        elif key == "l":
            self.request_mode("AUTO.LAND")
            return True
        elif key == "h":
            self.print_help()
            return True
        elif key == "x":
            rospy.loginfo("Exit requested")
            return False
        else:
            return True

        self.target_initialized = True
        rospy.loginfo(
            "Target: x=%.2f y=%.2f z=%.2f yaw=%.1f deg",
            self.target_x,
            self.target_y,
            self.target_z,
            math.degrees(self.target_yaw),
        )
        return True

    def publish_setpoint(self):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.args.frame_id
        msg.pose.position.x = self.target_x
        msg.pose.position.y = self.target_y
        msg.pose.position.z = self.target_z
        qx, qy, qz, qw = quat_from_yaw(self.target_yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pub.publish(msg)

    def run(self):
        if self.args.wait_connected:
            self.wait_for_mavros()

        self.print_help()
        rospy.loginfo("Publishing setpoints to %s", self.args.setpoint_topic)
        rospy.loginfo("Press T first for a takeoff-height target, then O and M for OFFBOARD/ARM.")

        stdin_fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)
        rate = rospy.Rate(self.args.rate)

        try:
            keep_running = True
            while not rospy.is_shutdown() and keep_running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                if ready:
                    keep_running = self.update_from_key(sys.stdin.read(1))
                self.publish_setpoint()
                rate.sleep()
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard PX4/MAVROS position teleop for mapping.")
    parser.add_argument("--setpoint-topic", default="/mavros/setpoint_position/local")
    parser.add_argument("--pose-topic", default="/mavros/local_position/pose")
    parser.add_argument("--state-topic", default="/mavros/state")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--step-xy", type=float, default=0.5)
    parser.add_argument("--step-z", type=float, default=0.25)
    parser.add_argument("--step-yaw-deg", type=float, default=10.0)
    parser.add_argument("--takeoff-altitude", type=float, default=1.5)
    parser.add_argument("--min-z", type=float, default=0.2)
    parser.add_argument("--max-z", type=float, default=4.0)
    parser.add_argument("--no-wait-connected", dest="wait_connected", action="store_false")
    parser.set_defaults(wait_connected=True)
    args = parser.parse_args(rospy.myargv()[1:])
    args.step_yaw = math.radians(args.step_yaw_deg)
    return args


if __name__ == "__main__":
    try:
        ManualPx4KeyboardTeleop(parse_args()).run()
    except rospy.ROSInterruptException:
        pass
