#!/usr/bin/env python3
"""Fly a coverage path over the dense pipe factory for Point-LIO mapping."""

from __future__ import annotations

import argparse
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


class PipeFactoryCoverageMission:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = State()
        self.pose: PoseStamped | None = None
        self.pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=20)
        rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=10)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=10)
        rospy.wait_for_service("/mavros/cmd/arming", timeout=args.timeout)
        rospy.wait_for_service("/mavros/set_mode", timeout=args.timeout)
        self.arming = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    def _state_cb(self, msg: State) -> None:
        self.state = msg

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.pose = msg

    def wait_connected(self) -> None:
        deadline = time.time() + self.args.timeout
        rate = rospy.Rate(self.args.rate)
        while not rospy.is_shutdown() and not self.state.connected:
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for MAVROS FCU connection")
            rate.sleep()

    def wait_pose(self) -> None:
        deadline = time.time() + self.args.timeout
        rate = rospy.Rate(self.args.rate)
        while not rospy.is_shutdown() and self.pose is None:
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for local position")
            rate.sleep()

    def make_pose(self, x: float, y: float, z: float, yaw: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        qx, qy, qz, qw = quat_from_yaw(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def publish_for(self, target: PoseStamped, duration: float) -> None:
        rate = rospy.Rate(self.args.rate)
        end_time = time.time() + duration
        while not rospy.is_shutdown() and time.time() < end_time:
            target.header.stamp = rospy.Time.now()
            self.pub.publish(target)
            rate.sleep()

    def distance_to(self, target: PoseStamped) -> float | None:
        if self.pose is None:
            return None
        p = self.pose.pose.position
        t = target.pose.position
        return math.sqrt((p.x - t.x) ** 2 + (p.y - t.y) ** 2 + (p.z - t.z) ** 2)

    def fly_to(self, target: PoseStamped) -> None:
        rate = rospy.Rate(self.args.rate)
        deadline = time.time() + self.args.segment_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            target.header.stamp = rospy.Time.now()
            self.pub.publish(target)
            distance = self.distance_to(target)
            if distance is not None and distance <= self.args.position_tolerance:
                self.publish_for(target, self.args.dwell)
                return
            rate.sleep()
        rospy.logwarn("Continuing after segment timeout at x=%.1f y=%.1f", target.pose.position.x, target.pose.position.y)

    def arm_offboard(self, start: PoseStamped) -> None:
        self.publish_for(start, 5.0)

        rate = rospy.Rate(self.args.rate)
        deadline = time.time() + self.args.timeout
        last_mode_request = 0.0
        last_arm_request = 0.0
        while not rospy.is_shutdown() and (self.state.mode != "OFFBOARD" or not self.state.armed):
            now = time.time()
            start.header.stamp = rospy.Time.now()
            self.pub.publish(start)

            if self.state.mode != "OFFBOARD" and now - last_mode_request > 1.0:
                result = self.set_mode(base_mode=0, custom_mode="OFFBOARD")
                rospy.loginfo("OFFBOARD request sent=%s current_mode=%s", result.mode_sent, self.state.mode)
                last_mode_request = now

            if self.state.mode == "OFFBOARD" and not self.state.armed and now - last_arm_request > 1.0:
                result = self.arming(True)
                rospy.loginfo("Arm request success=%s armed=%s", result.success, self.state.armed)
                last_arm_request = now

            if now > deadline:
                raise RuntimeError(f"Timed out entering OFFBOARD armed state; mode={self.state.mode} armed={self.state.armed}")
            rate.sleep()

    def build_waypoints(self) -> list[PoseStamped]:
        waypoints: list[PoseStamped] = []
        lanes = [float(v) for v in self.args.lanes.split(",")]
        xs = [self.args.min_x, self.args.max_x]
        for pass_i, altitude in enumerate(self.args.altitudes):
            ordered_lanes = lanes if pass_i % 2 == 0 else list(reversed(lanes))
            for lane_i, y in enumerate(ordered_lanes):
                start_x, end_x = xs if lane_i % 2 == 0 else list(reversed(xs))
                yaw = 0.0 if end_x > start_x else math.pi
                waypoints.append(self.make_pose(start_x, y, altitude, yaw))
                waypoints.append(self.make_pose(end_x, y, altitude, yaw))

            ordered_cols = self.args.cross_columns
            for col_i, x in enumerate(ordered_cols):
                start_y, end_y = (self.args.min_y, self.args.max_y) if col_i % 2 == 0 else (self.args.max_y, self.args.min_y)
                yaw = math.pi / 2.0 if end_y > start_y else -math.pi / 2.0
                waypoints.append(self.make_pose(x, start_y, altitude, yaw))
                waypoints.append(self.make_pose(x, end_y, altitude, yaw))
        return waypoints

    def run(self) -> None:
        self.wait_connected()
        self.wait_pose()
        start = self.make_pose(0.0, 0.0, self.args.altitudes[0], 0.0)
        self.arm_offboard(start)
        self.fly_to(start)

        waypoints = self.build_waypoints()
        for idx, waypoint in enumerate(waypoints):
            p = waypoint.pose.position
            rospy.loginfo("Coverage waypoint %03d/%03d: x=%.1f y=%.1f z=%.1f", idx + 1, len(waypoints), p.x, p.y, p.z)
            self.fly_to(waypoint)

        self.fly_to(self.make_pose(0.0, 0.0, self.args.altitudes[0], 0.0))
        if self.args.land:
            self.set_mode(base_mode=0, custom_mode="AUTO.LAND")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-x", type=float, default=-52.0)
    parser.add_argument("--max-x", type=float, default=52.0)
    parser.add_argument("--min-y", type=float, default=-15.0)
    parser.add_argument("--max-y", type=float, default=15.0)
    parser.add_argument("--lanes", default="-15,-3,3,15")
    parser.add_argument("--cross-columns", type=float, nargs="+", default=[-44.0, -28.0, -12.0, 12.0, 28.0, 44.0])
    parser.add_argument("--altitudes", type=float, nargs="+", default=[6.7])
    parser.add_argument("--dwell", type=float, default=1.5)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--position-tolerance", type=float, default=1.4)
    parser.add_argument("--segment-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--land", action="store_true")
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    rospy.init_node("sitl_pipe_factory_coverage_mission")
    PipeFactoryCoverageMission(parse_args()).run()


if __name__ == "__main__":
    main()
