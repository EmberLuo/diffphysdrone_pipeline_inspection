#!/usr/bin/env python3
"""Capture global A* path and lookahead target samples from ROS topics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry, Path as RosPath


def _pose_msg(x: float, y: float, z: float, frame_id: str) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    return msg


def _odom_msg(x: float, y: float, z: float, frame_id: str) -> Odometry:
    msg = Odometry()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.child_frame_id = "base_link"
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = z
    msg.pose.pose.orientation.w = 1.0
    return msg


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _resample_path(
    path: list[tuple[float, float, float]],
    step: float,
) -> list[tuple[float, float, float]]:
    if len(path) < 2:
        return path
    samples = [path[0]]
    carried = 0.0
    prev = path[0]
    for cur in path[1:]:
        seg = _dist(prev, cur)
        if seg <= 1e-9:
            prev = cur
            continue
        while carried + seg >= step:
            t = (step - carried) / seg
            point = tuple(prev[i] + (cur[i] - prev[i]) * t for i in range(3))
            samples.append(point)
            prev = point
            seg = _dist(prev, cur)
            carried = 0.0
            if seg <= 1e-9:
                break
        carried += seg
        prev = cur
    if samples[-1] != path[-1]:
        samples.append(path[-1])
    return samples


class GlobalPathCapture:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.path: list[tuple[float, float, float]] = []
        self.captured_path: list[tuple[float, float, float]] = []
        self.latest_target: tuple[float, float, float] | None = None
        self.target_samples: list[dict[str, float]] = []

        self.odom_pub = rospy.Publisher(args.odom_topic, Odometry, queue_size=10)
        self.goal_pub = rospy.Publisher(args.goal_topic, PoseStamped, queue_size=1, latch=True)
        rospy.Subscriber(args.path_topic, RosPath, self.path_cb, queue_size=1)
        rospy.Subscriber(args.target_topic, PointStamped, self.target_cb, queue_size=10)

    def path_cb(self, msg: RosPath) -> None:
        self.path = [
            (p.pose.position.x, p.pose.position.y, p.pose.position.z)
            for p in msg.poses
        ]

    def target_cb(self, msg: PointStamped) -> None:
        self.latest_target = (msg.point.x, msg.point.y, msg.point.z)

    def wait_for_path(self) -> None:
        start = tuple(self.args.start)
        goal = tuple(self.args.goal)
        rate = rospy.Rate(self.args.publish_rate)
        deadline = rospy.Time.now() + rospy.Duration(self.args.timeout)
        goal_msg = _pose_msg(goal[0], goal[1], goal[2], self.args.frame_id)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self.odom_pub.publish(_odom_msg(start[0], start[1], start[2], self.args.frame_id))
            goal_msg.header.stamp = rospy.Time.now()
            self.goal_pub.publish(goal_msg)
            if len(self.path) >= 2:
                self.captured_path = list(self.path)
                rospy.loginfo("Captured global path with %d points", len(self.path))
                return
            rate.sleep()
        raise TimeoutError("Timed out waiting for /global_path")

    def record_targets(self) -> None:
        if not self.captured_path:
            return
        start = tuple(self.args.start)
        goal = tuple(self.args.goal)
        path = list(self.captured_path)
        if _dist(path[0], goal) < _dist(path[0], start):
            path = list(reversed(path))
        samples = _resample_path(path, self.args.odom_step)
        rate = rospy.Rate(self.args.publish_rate)
        t0 = rospy.Time.now()
        for x, y, z in samples:
            if rospy.is_shutdown():
                break
            self.odom_pub.publish(_odom_msg(x, y, z, self.args.frame_id))
            rate.sleep()
            if self.latest_target is None:
                continue
            tx, ty, tz = self.latest_target
            self.target_samples.append(
                {
                    "time_s": (rospy.Time.now() - t0).to_sec(),
                    "odom_x": x,
                    "odom_y": y,
                    "odom_z": z,
                    "target_x": tx,
                    "target_y": ty,
                    "target_z": tz,
                    "target_distance_m": _dist((x, y, z), (tx, ty, tz)),
                }
            )

    def write_outputs(self) -> None:
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "global_path_points.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["index", "x", "y", "z"])
            writer.writeheader()
            for i, (x, y, z) in enumerate(self.captured_path):
                writer.writerow({"index": i, "x": f"{x:.5f}", "y": f"{y:.5f}", "z": f"{z:.5f}"})

        with (output_dir / "local_target_samples.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "time_s",
                    "odom_x",
                    "odom_y",
                    "odom_z",
                    "target_x",
                    "target_y",
                    "target_z",
                    "target_distance_m",
                ],
            )
            writer.writeheader()
            for row in self.target_samples:
                writer.writerow({k: f"{v:.5f}" for k, v in row.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", nargs=3, type=float, default=[-28.0, 0.0, 1.8])
    parser.add_argument("--goal", nargs=3, type=float, default=[28.0, 0.0, 1.8])
    parser.add_argument("--frame_id", default="world")
    parser.add_argument("--odom_topic", default="/Odometry")
    parser.add_argument("--goal_topic", default="/move_base_simple/goal")
    parser.add_argument("--path_topic", default="/global_path")
    parser.add_argument("--target_topic", default="/e2e/local_target")
    parser.add_argument("--publish_rate", type=float, default=20.0)
    parser.add_argument("--odom_step", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    rospy.init_node("capture_global_path_targets")
    capture = GlobalPathCapture(parse_args())
    capture.wait_for_path()
    capture.record_targets()
    capture.write_outputs()


if __name__ == "__main__":
    main()
