#!/usr/bin/env python3
"""Record Point-LIO odometry samples and write thesis-ready metrics."""

import csv
import glob
import math
import os
from pathlib import Path

import rospy
from nav_msgs.msg import Odometry


def _pos(msg):
    p = msg.pose.pose.position
    return (p.x, p.y, p.z)


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _pcd_point_count(path):
    try:
        with open(path, "rb") as fp:
            for raw in fp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("POINTS"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
                if line.startswith("DATA"):
                    break
    except OSError:
        return 0
    return 0


class PointLioMetricsRecorder:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.gt_odom_topic = rospy.get_param("~gt_odom_topic", "/mavros/local_position/odom")
        self.pcd_dir = rospy.get_param("~pcd_dir", "")
        self.prepared_map_path = rospy.get_param("~prepared_map_path", "")
        self.output_dir = Path(rospy.get_param("~output_dir", "assets/validation/thesis_pointlio"))
        self.sample_rate = float(rospy.get_param("~sample_rate", 30.0))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.output_dir / "point_lio_samples.csv"
        self.summary_path = self.output_dir / "point_lio_summary.csv"

        self.latest_odom = None
        self.latest_gt = None
        self.odom_times = []
        self.rows = []
        self.start_time = rospy.Time.now()

        self.fp = open(self.samples_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.fp,
            fieldnames=[
                "time_s",
                "est_x",
                "est_y",
                "est_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "position_error_m",
            ],
        )
        self.writer.writeheader()

        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.gt_odom_topic, Odometry, self.gt_cb, queue_size=1, tcp_nodelay=True)
        rospy.Timer(rospy.Duration(1.0 / self.sample_rate), self.timer_cb)
        rospy.on_shutdown(self.close)
        rospy.loginfo("point_lio_metrics_recorder ready: output=%s", self.output_dir)

    def odom_cb(self, msg):
        stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        self.odom_times.append(stamp.to_sec())
        self.latest_odom = msg

    def gt_cb(self, msg):
        self.latest_gt = msg

    def timer_cb(self, _event):
        if self.latest_odom is None or self.fp.closed:
            return
        now = rospy.Time.now()
        est = _pos(self.latest_odom)
        gt = _pos(self.latest_gt) if self.latest_gt is not None else ("", "", "")
        err = "" if self.latest_gt is None else _dist(est, gt)
        row = {
            "time_s": f"{(now - self.start_time).to_sec():.3f}",
            "est_x": f"{est[0]:.5f}",
            "est_y": f"{est[1]:.5f}",
            "est_z": f"{est[2]:.5f}",
            "gt_x": "" if gt[0] == "" else f"{gt[0]:.5f}",
            "gt_y": "" if gt[1] == "" else f"{gt[1]:.5f}",
            "gt_z": "" if gt[2] == "" else f"{gt[2]:.5f}",
            "position_error_m": "" if err == "" else f"{err:.5f}",
        }
        self.writer.writerow(row)
        self.fp.flush()
        if err != "":
            self.rows.append({"position_error_m": err})
        self.write_summary()

    def _frequency_stats(self):
        if len(self.odom_times) < 2:
            return "", ""
        dts = [b - a for a, b in zip(self.odom_times[:-1], self.odom_times[1:]) if b > a]
        if not dts:
            return "", ""
        freqs = [1.0 / dt for dt in dts]
        return sum(freqs) / len(freqs), min(freqs)

    def _pcd_stats(self):
        raw_points = 0
        if self.pcd_dir:
            for path in glob.glob(os.path.join(self.pcd_dir, "scans*.pcd")):
                raw_points += _pcd_point_count(path)
        filtered_points = _pcd_point_count(self.prepared_map_path) if self.prepared_map_path else 0
        file_size_mb = ""
        if self.prepared_map_path and os.path.exists(self.prepared_map_path):
            file_size_mb = os.path.getsize(self.prepared_map_path) / (1024.0 * 1024.0)
        return raw_points, filtered_points, file_size_mb

    def write_summary(self):
        mean_freq, min_freq = self._frequency_stats()
        errors = [r["position_error_m"] for r in self.rows]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else ""
        max_err = max(errors) if errors else ""
        raw_points, filtered_points, file_size_mb = self._pcd_stats()

        with open(self.summary_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["metric", "value", "unit"])
            writer.writeheader()
            rows = [
                ("PLIO_FREQ_MEAN", mean_freq, "Hz"),
                ("PLIO_FREQ_MIN", min_freq, "Hz"),
                ("PLIO_RMSE", rmse, "m"),
                ("PLIO_MAX_ERR", max_err, "m"),
                ("PCD_RAW_POINTS", raw_points, "points"),
                ("PCD_FILTERED_POINTS", filtered_points, "points"),
                ("PCD_FILE_SIZE", file_size_mb, "MB"),
                ("MAPPING_TIME", "", "s"),
            ]
            for metric, value, unit in rows:
                writer.writerow(
                    {
                        "metric": metric,
                        "value": "" if value == "" else f"{float(value):.5f}",
                        "unit": unit,
                    }
                )

    def close(self):
        try:
            self.write_summary()
        finally:
            if not self.fp.closed:
                self.fp.close()


if __name__ == "__main__":
    rospy.init_node("point_lio_metrics_recorder")
    PointLioMetricsRecorder()
    rospy.spin()
