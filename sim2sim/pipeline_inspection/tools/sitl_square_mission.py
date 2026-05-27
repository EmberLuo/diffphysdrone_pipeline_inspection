#!/usr/bin/env python3

import argparse
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class SquareMission:
    def __init__(self, args):
        self.args = args
        self.state = State()
        self.pose = None
        self.pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        rospy.Subscriber("/mavros/state", State, self._state_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        rospy.wait_for_service("/mavros/cmd/arming", timeout=args.timeout)
        rospy.wait_for_service("/mavros/set_mode", timeout=args.timeout)
        self.arming = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        self.pose = msg

    def wait_connected(self):
        deadline = time.time() + self.args.timeout
        rate = rospy.Rate(self.args.rate)
        while not rospy.is_shutdown() and not self.state.connected:
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for MAVROS FCU connection")
            rate.sleep()

    def make_pose(self, x, y, z, yaw=0.0):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        return msg

    def publish_for(self, target, duration):
        end_time = time.time() + duration
        rate = rospy.Rate(self.args.rate)
        while not rospy.is_shutdown() and time.time() < end_time:
            target.header.stamp = rospy.Time.now()
            self.pub.publish(target)
            rate.sleep()

    def arm_offboard(self, start):
        self.publish_for(start, 3.0)
        if self.state.mode != "OFFBOARD":
            result = self.set_mode(custom_mode="OFFBOARD")
            if not result.mode_sent:
                raise RuntimeError("Failed to request OFFBOARD mode")
        if not self.state.armed:
            result = self.arming(True)
            if not result.success:
                raise RuntimeError("Failed to arm vehicle")

    def run(self):
        self.wait_connected()
        start = self.make_pose(0.0, 0.0, self.args.altitude)
        self.arm_offboard(start)

        side = self.args.side
        waypoints = [
            self.make_pose(0.0, 0.0, self.args.altitude, 0.0),
            self.make_pose(side, 0.0, self.args.altitude, 0.0),
            self.make_pose(side, side, self.args.altitude, math.pi / 2.0),
            self.make_pose(0.0, side, self.args.altitude, math.pi),
            self.make_pose(0.0, 0.0, self.args.altitude, -math.pi / 2.0),
        ]
        for idx, waypoint in enumerate(waypoints):
            rospy.loginfo("Publishing square waypoint %d/%d", idx + 1, len(waypoints))
            self.publish_for(waypoint, self.args.hold)


def parse_args():
    parser = argparse.ArgumentParser(description="Fly a small MAVROS OFFBOARD square for SITL mapping.")
    parser.add_argument("--altitude", type=float, default=1.5)
    parser.add_argument("--side", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=8.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(rospy.myargv()[1:])


def main():
    rospy.init_node("sitl_square_mission")
    SquareMission(parse_args()).run()


if __name__ == "__main__":
    main()
