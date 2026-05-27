#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict

import rospy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


MODE_ORDER = ("normal", "degraded", "lost")


def pos_tuple(msg):
    p = msg.pose.pose.position
    return (p.x, p.y, p.z)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def continuity_label(max_jump):
    if max_jump <= 0.3:
        return "good"
    if max_jump <= 0.8:
        return "acceptable"
    return "poor"


class GnssModeMetricsRecorder:
    def __init__(self):
        self.gnss_odom_topic = rospy.get_param("~gnss_odom_topic", "/gnss/odom")
        self.slam_odom_topic = rospy.get_param("~slam_odom_topic", "/slam/odom")
        self.selected_odom_topic = rospy.get_param("~selected_odom_topic", "/Odometry")
        self.gnss_mode_topic = rospy.get_param("~gnss_mode_topic", "/gnss/mode")
        self.localization_mode_topic = rospy.get_param("~localization_mode_topic", "/localization/mode")
        self.control_point_topic = rospy.get_param("~control_point_topic", "/e2e_px4_controller/accel_setpoint")
        self.control_vector_topic = rospy.get_param("~control_vector_topic", "/control/accel_setpoint")
        self.output_dir = rospy.get_param(
            "~output_dir",
            "sim2sim/pipeline_inspection/assets/validation/gnss_mode_test",
        )
        self.sample_rate = float(rospy.get_param("~sample_rate", 20.0))

        os.makedirs(self.output_dir, exist_ok=True)
        self.samples_path = os.path.join(self.output_dir, "gnss_mode_samples.csv")
        self.summary_path = os.path.join(self.output_dir, "gnss_mode_summary.csv")

        self.latest_gnss = None
        self.latest_slam = None
        self.latest_selected = None
        self.gnss_mode = "lost"
        self.localization_mode = "unavailable"
        self.start_time = rospy.Time.now()
        self.last_selected_pos = None
        self.last_selected_time = None
        self.last_control = None
        self.last_control_time = None
        self.last_control_rate = 0.0
        self.mode_change_time = {mode: None for mode in MODE_ORDER}
        self.switch_delay = {mode: None for mode in MODE_ORDER}
        self.rows_by_mode = defaultdict(list)

        self.samples_fp = open(self.samples_path, "w", newline="", encoding="utf-8")
        self.samples_writer = csv.DictWriter(
            self.samples_fp,
            fieldnames=[
                "time_s",
                "gnss_mode",
                "localization_mode",
                "gnss_x",
                "gnss_y",
                "gnss_z",
                "slam_x",
                "slam_y",
                "slam_z",
                "selected_x",
                "selected_y",
                "selected_z",
                "selected_error_m",
                "selected_jump_m",
                "control_rate_mps3",
            ],
        )
        self.samples_writer.writeheader()
        self.samples_fp.flush()

        rospy.Subscriber(self.gnss_odom_topic, Odometry, self.gnss_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.slam_odom_topic, Odometry, self.slam_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.selected_odom_topic, Odometry, self.selected_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.gnss_mode_topic, String, self.gnss_mode_cb, queue_size=1)
        rospy.Subscriber(self.localization_mode_topic, String, self.localization_mode_cb, queue_size=1)
        rospy.Subscriber(self.control_point_topic, PointStamped, self.control_point_cb, queue_size=10)
        rospy.Subscriber(self.control_vector_topic, Vector3Stamped, self.control_vector_cb, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / self.sample_rate), self.timer_cb)
        rospy.on_shutdown(self.close)
        rospy.loginfo("gnss_mode_metrics_recorder ready: output=%s", self.output_dir)

    def gnss_cb(self, msg):
        self.latest_gnss = msg

    def slam_cb(self, msg):
        self.latest_slam = msg

    def selected_cb(self, msg):
        self.latest_selected = msg

    def gnss_mode_cb(self, msg):
        mode = msg.data.strip().lower()
        if mode not in MODE_ORDER:
            return
        if mode != self.gnss_mode:
            self.mode_change_time[mode] = rospy.Time.now()
        self.gnss_mode = mode

    def localization_mode_cb(self, msg):
        self.localization_mode = msg.data.strip().lower()
        expected = "gnss" if self.gnss_mode == "normal" else "slam"
        if self.localization_mode == expected and self.switch_delay[self.gnss_mode] is None:
            t0 = self.mode_change_time.get(self.gnss_mode)
            if t0 is not None:
                self.switch_delay[self.gnss_mode] = max(0.0, (rospy.Time.now() - t0).to_sec())

    def control_point_cb(self, msg):
        p = msg.point
        self.update_control((p.x, p.y, p.z), msg.header.stamp if msg.header.stamp else rospy.Time.now())

    def control_vector_cb(self, msg):
        v = msg.vector
        self.update_control((v.x, v.y, v.z), msg.header.stamp if msg.header.stamp else rospy.Time.now())

    def update_control(self, value, stamp):
        if self.last_control is not None and self.last_control_time is not None:
            dt = (stamp - self.last_control_time).to_sec()
            if dt > 1e-6:
                self.last_control_rate = dist(value, self.last_control) / dt
        self.last_control = value
        self.last_control_time = stamp

    def timer_cb(self, _event):
        if self.samples_fp.closed:
            return
        if self.latest_slam is None or self.latest_selected is None:
            return

        now = rospy.Time.now()
        selected_pos = pos_tuple(self.latest_selected)
        slam_pos = pos_tuple(self.latest_slam)
        gnss_pos = pos_tuple(self.latest_gnss) if self.latest_gnss is not None else ("", "", "")
        selected_error = dist(selected_pos, slam_pos)
        selected_jump = 0.0
        if self.last_selected_pos is not None:
            selected_jump = dist(selected_pos, self.last_selected_pos)
        self.last_selected_pos = selected_pos
        self.last_selected_time = now

        row = {
            "time_s": f"{(now - self.start_time).to_sec():.3f}",
            "gnss_mode": self.gnss_mode,
            "localization_mode": self.localization_mode,
            "gnss_x": "" if gnss_pos[0] == "" else f"{gnss_pos[0]:.4f}",
            "gnss_y": "" if gnss_pos[1] == "" else f"{gnss_pos[1]:.4f}",
            "gnss_z": "" if gnss_pos[2] == "" else f"{gnss_pos[2]:.4f}",
            "slam_x": f"{slam_pos[0]:.4f}",
            "slam_y": f"{slam_pos[1]:.4f}",
            "slam_z": f"{slam_pos[2]:.4f}",
            "selected_x": f"{selected_pos[0]:.4f}",
            "selected_y": f"{selected_pos[1]:.4f}",
            "selected_z": f"{selected_pos[2]:.4f}",
            "selected_error_m": f"{selected_error:.4f}",
            "selected_jump_m": f"{selected_jump:.4f}",
            "control_rate_mps3": f"{self.last_control_rate:.4f}",
        }
        self.samples_writer.writerow(row)
        self.samples_fp.flush()
        self.rows_by_mode[self.gnss_mode].append(
            {
                "selected_error_m": selected_error,
                "selected_jump_m": selected_jump,
                "control_rate_mps3": self.last_control_rate,
            }
        )
        self.write_summary()

    def write_summary(self):
        with open(self.summary_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "mode",
                    "mean_position_error_m",
                    "max_position_jump_m",
                    "switch_delay_s",
                    "max_control_rate_mps3",
                    "continuity_label",
                    "sample_count",
                ],
            )
            writer.writeheader()
            for mode in MODE_ORDER:
                rows = self.rows_by_mode.get(mode, [])
                if not rows:
                    writer.writerow(
                        {
                            "mode": mode,
                            "mean_position_error_m": "",
                            "max_position_jump_m": "",
                            "switch_delay_s": "",
                            "max_control_rate_mps3": "",
                            "continuity_label": "missing",
                            "sample_count": 0,
                        }
                    )
                    continue
                mean_error = sum(r["selected_error_m"] for r in rows) / len(rows)
                max_jump = max(r["selected_jump_m"] for r in rows)
                max_rate = max(r["control_rate_mps3"] for r in rows)
                delay = self.switch_delay[mode]
                writer.writerow(
                    {
                        "mode": mode,
                        "mean_position_error_m": f"{mean_error:.4f}",
                        "max_position_jump_m": f"{max_jump:.4f}",
                        "switch_delay_s": "" if delay is None else f"{delay:.4f}",
                        "max_control_rate_mps3": f"{max_rate:.4f}",
                        "continuity_label": continuity_label(max_jump),
                        "sample_count": len(rows),
                    }
                )

    def close(self):
        try:
            self.write_summary()
        finally:
            if not self.samples_fp.closed:
                self.samples_fp.close()


if __name__ == "__main__":
    rospy.init_node("gnss_mode_metrics_recorder")
    GnssModeMetricsRecorder()
    rospy.spin()
