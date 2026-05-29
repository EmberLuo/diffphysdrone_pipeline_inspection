#!/usr/bin/env python3
"""Record complete task closed-loop metrics for thesis experiments."""

import csv
import math
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from std_msgs.msg import Bool, String

from global_planning_metrics_recorder import _load_pcd_xyz, _min_cloud_distance


def _pos_from_odom(msg):
    p = msg.pose.pose.position
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def _pos_from_pose(msg):
    p = msg.pose.position
    return np.array([p.x, p.y, p.z], dtype=np.float64)


class ClosedLoopMetricsRecorder:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.path_topic = rospy.get_param("~path_topic", "/global_path")
        self.estop_topic = rospy.get_param("~estop_topic", "/e2e/estop")
        self.state_topic = rospy.get_param("~state_topic", "/inspection/state")
        self.map_path = rospy.get_param("~map_path", "")
        self.output_dir = Path(rospy.get_param("~output_dir", "assets/validation/thesis_closed_loop"))
        self.goal_radius = float(rospy.get_param("~goal_radius", 0.5))
        self.hover_window_sec = float(rospy.get_param("~hover_window_sec", 5.0))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.output_dir / "closed_loop_samples.csv"
        self.events_path = self.output_dir / "closed_loop_events.csv"
        self.summary_path = self.output_dir / "closed_loop_summary.csv"
        self.cloud = _load_pcd_xyz(self.map_path)

        self.goal = None
        self.start_time = rospy.Time.now()
        self.first_odom_time = None
        self.last_time = None
        self.last_pos = None
        self.path_length = 0.0
        self.min_obstacle_distance = float("inf")
        self.goal_errors = []
        self.hover_errors = []
        self.replans = 0
        self.safe_triggers = 0
        self.manual_takeovers = 0
        self.completed = False

        self.samples_fp = open(self.samples_path, "w", newline="", encoding="utf-8")
        self.samples_writer = csv.DictWriter(
            self.samples_fp,
            fieldnames=["time_s", "x", "y", "z", "goal_error_m", "min_obstacle_distance_m", "path_length_m"],
        )
        self.samples_writer.writeheader()
        self.events_fp = open(self.events_path, "w", newline="", encoding="utf-8")
        self.events_writer = csv.DictWriter(self.events_fp, fieldnames=["time_s", "event", "detail"])
        self.events_writer.writeheader()

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_cb, queue_size=1)
        rospy.Subscriber(self.path_topic, RosPath, self.path_cb, queue_size=1)
        rospy.Subscriber(self.estop_topic, Bool, self.estop_cb, queue_size=1)
        rospy.Subscriber(self.state_topic, String, self.state_cb, queue_size=10)
        rospy.on_shutdown(self.close)
        rospy.loginfo("closed_loop_metrics_recorder ready: output=%s", self.output_dir)

    def _event(self, name, detail=""):
        now = rospy.Time.now()
        self.events_writer.writerow(
            {"time_s": f"{(now - self.start_time).to_sec():.3f}", "event": name, "detail": detail}
        )
        self.events_fp.flush()

    def goal_cb(self, msg):
        self.goal = _pos_from_pose(msg)
        self._event("goal", ",".join(f"{x:.3f}" for x in self.goal))

    def path_cb(self, _msg):
        self.replans += 1
        self._event("path", f"replan={self.replans}")

    def estop_cb(self, msg):
        if bool(msg.data):
            self.safe_triggers += 1
            self._event("estop", "true")

    def state_cb(self, msg):
        text = msg.data.strip().lower()
        if "safe" in text or "failsafe" in text or "protect" in text:
            self.safe_triggers += 1
        if "manual" in text or "takeover" in text:
            self.manual_takeovers += 1
        self._event("state", msg.data)

    def odom_cb(self, msg):
        now = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        pos = _pos_from_odom(msg)
        if self.first_odom_time is None:
            self.first_odom_time = now
        if self.last_pos is not None:
            self.path_length += float(np.linalg.norm(pos - self.last_pos))
        self.last_pos = pos
        self.last_time = now

        min_dist = _min_cloud_distance([pos], self.cloud)
        if min_dist != "":
            self.min_obstacle_distance = min(self.min_obstacle_distance, min_dist)
        goal_error = ""
        if self.goal is not None:
            goal_error = float(np.linalg.norm(pos - self.goal))
            self.goal_errors.append(goal_error)
            if goal_error <= self.goal_radius:
                self.completed = True
                self.hover_errors.append(goal_error)
        row = {
            "time_s": f"{(now - self.start_time).to_sec():.3f}",
            "x": f"{pos[0]:.5f}",
            "y": f"{pos[1]:.5f}",
            "z": f"{pos[2]:.5f}",
            "goal_error_m": "" if goal_error == "" else f"{goal_error:.5f}",
            "min_obstacle_distance_m": "" if min_dist == "" else f"{min_dist:.5f}",
            "path_length_m": f"{self.path_length:.5f}",
        }
        self.samples_writer.writerow(row)
        self.samples_fp.flush()
        self.write_summary()

    def write_summary(self):
        total_time = ""
        if self.first_odom_time is not None and self.last_time is not None:
            total_time = (self.last_time - self.first_odom_time).to_sec()
        goal_err = self.goal_errors[-1] if self.goal_errors else ""
        hover_err = math.sqrt(sum(e * e for e in self.hover_errors) / len(self.hover_errors)) if self.hover_errors else ""
        min_dist = "" if self.min_obstacle_distance == float("inf") else self.min_obstacle_distance
        with open(self.summary_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["metric", "value", "unit"])
            writer.writeheader()
            values = [
                ("CLOSED_LOOP_SUCCESS_RATE", 100.0 if self.completed else 0.0, "%"),
                ("CLOSED_LOOP_SUCCESS_NUM", 1 if self.completed else 0, "count"),
                ("CLOSED_LOOP_TOTAL_NUM", 1, "count"),
                ("CLOSED_LOOP_TIME_MEAN", total_time, "s"),
                ("CLOSED_LOOP_PATH_LENGTH", self.path_length, "m"),
                ("CLOSED_LOOP_GOAL_ERR", goal_err, "m"),
                ("CLOSED_LOOP_HOVER_ERR", hover_err, "m"),
                ("CLOSED_LOOP_MIN_OBS_DIST", min_dist, "m"),
                ("CLOSED_LOOP_REPLAN_NUM", self.replans, "count"),
                ("CLOSED_LOOP_SAFE_TRIGGER", self.safe_triggers, "count"),
                ("CLOSED_LOOP_MANUAL_TAKEOVER", self.manual_takeovers, "count"),
            ]
            for metric, value, unit in values:
                writer.writerow({"metric": metric, "value": "" if value == "" else value, "unit": unit})

    def close(self):
        try:
            self.write_summary()
        finally:
            if not self.samples_fp.closed:
                self.samples_fp.close()
            if not self.events_fp.closed:
                self.events_fp.close()


if __name__ == "__main__":
    rospy.init_node("closed_loop_metrics_recorder")
    ClosedLoopMetricsRecorder()
    rospy.spin()
