#!/usr/bin/env python3
"""Record global A* path and lookahead-target metrics for thesis tables."""

import csv
import math
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath


def _point_xyz(point):
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def _pose_xyz(pose_stamped):
    p = pose_stamped.pose.position
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def _load_pcd_xyz(path, max_points=50000):
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        rospy.logwarn("PCD map not found: %s", path)
        return None
    header = []
    with open(path, "rb") as fp:
        while True:
            line = fp.readline()
            if not line:
                return None
            text = line.decode("utf-8", errors="ignore").strip()
            header.append(text)
            if text.startswith("DATA"):
                data_kind = text.split()[1].lower()
                break
        meta = {}
        for line in header:
            parts = line.split()
            if parts:
                meta[parts[0].upper()] = parts[1:]
        fields = meta.get("FIELDS", [])
        if not {"x", "y", "z"}.issubset(set(fields)):
            return None
        points = int((meta.get("POINTS") or meta.get("WIDTH") or ["0"])[0])
        if data_kind == "ascii":
            arr = np.loadtxt(fp, dtype=np.float64, max_rows=points)
            if arr.ndim == 1:
                arr = arr[None, :]
            xyz = arr[:, [fields.index("x"), fields.index("y"), fields.index("z")]]
        elif data_kind == "binary":
            sizes = [int(x) for x in meta["SIZE"]]
            types = meta["TYPE"]
            counts = [int(x) for x in meta.get("COUNT", ["1"] * len(fields))]
            dtype_fields = []
            for name, size, typ, count in zip(fields, sizes, types, counts):
                if typ == "F" and size == 4:
                    dt = "<f4"
                elif typ == "F" and size == 8:
                    dt = "<f8"
                elif typ == "U" and size == 4:
                    dt = "<u4"
                elif typ == "I" and size == 4:
                    dt = "<i4"
                else:
                    dt = f"V{size}"
                dtype_fields.append((name, dt, (count,) if count > 1 else ()))
            raw = fp.read()
            arr = np.frombuffer(raw, dtype=np.dtype(dtype_fields), count=points)
            xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
        else:
            rospy.logwarn("Unsupported PCD DATA format: %s", data_kind)
            return None
    if xyz.shape[0] > max_points:
        idx = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
        xyz = xyz[idx]
    return xyz


def _path_length(points):
    if len(points) < 2:
        return 0.0
    return float(sum(np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))))


def _min_cloud_distance(points, cloud):
    if cloud is None or len(points) == 0:
        return ""
    best = float("inf")
    cloud = np.asarray(cloud)
    for p in points:
        diff = cloud - p[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        best = min(best, float(np.sqrt(d2.min())))
    return best


class GlobalPlanningMetricsRecorder:
    def __init__(self):
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.path_topic = rospy.get_param("~path_topic", "/global_path")
        self.target_topic = rospy.get_param("~target_topic", "/e2e/local_target")
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.map_path = rospy.get_param("~map_path", "")
        self.output_dir = Path(rospy.get_param("~output_dir", "assets/validation/thesis_global_planning"))
        self.target_switch_distance = float(rospy.get_param("~target_switch_distance", 0.15))
        self.resolution = rospy.get_param("~resolution", "")
        self.inflation_radius = rospy.get_param("~inflation_radius", "")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.output_dir / "global_planning_samples.csv"
        self.summary_path = self.output_dir / "global_planning_summary.csv"
        self.cloud = _load_pcd_xyz(self.map_path)

        self.last_goal_time = None
        self.last_path_points = []
        self.last_target = None
        self.target_switches = 0
        self.replans = 0
        self.rows = []

        self.fp = open(self.samples_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.fp,
            fieldnames=[
                "event",
                "time_s",
                "planning_latency_s",
                "waypoint_count",
                "path_length_m",
                "min_obstacle_distance_m",
                "target_switches",
                "replans",
            ],
        )
        self.writer.writeheader()

        rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_cb, queue_size=1)
        rospy.Subscriber(self.path_topic, RosPath, self.path_cb, queue_size=1)
        rospy.Subscriber(self.target_topic, PointStamped, self.target_cb, queue_size=10)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        rospy.on_shutdown(self.close)
        rospy.loginfo("global_planning_metrics_recorder ready: output=%s", self.output_dir)

    def goal_cb(self, _msg):
        self.last_goal_time = rospy.Time.now()

    def odom_cb(self, _msg):
        pass

    def path_cb(self, msg):
        now = rospy.Time.now()
        points = [_pose_xyz(p) for p in msg.poses]
        latency = "" if self.last_goal_time is None else (now - self.last_goal_time).to_sec()
        path_len = _path_length(points)
        min_dist = _min_cloud_distance(points, self.cloud)
        self.replans += 1
        row = {
            "event": "path",
            "time_s": f"{now.to_sec():.3f}",
            "planning_latency_s": "" if latency == "" else f"{latency:.5f}",
            "waypoint_count": len(points),
            "path_length_m": f"{path_len:.5f}",
            "min_obstacle_distance_m": "" if min_dist == "" else f"{min_dist:.5f}",
            "target_switches": self.target_switches,
            "replans": self.replans,
        }
        self.writer.writerow(row)
        self.fp.flush()
        self.rows.append(row)
        self.last_path_points = points
        self.write_summary()

    def target_cb(self, msg):
        target = _point_xyz(msg.point)
        if self.last_target is not None and np.linalg.norm(target - self.last_target) > self.target_switch_distance:
            self.target_switches += 1
        self.last_target = target

    def write_summary(self):
        path_rows = [r for r in self.rows if r["event"] == "path"]
        last = path_rows[-1] if path_rows else {}
        with open(self.summary_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["metric", "value", "unit"])
            writer.writeheader()
            values = [
                ("VOXEL_RES", self.resolution, "m"),
                ("INFLATION_RADIUS", self.inflation_radius, "m"),
                ("GLOBAL_PLAN_TIME", last.get("planning_latency_s", ""), "s"),
                ("GLOBAL_PATH_LENGTH", last.get("path_length_m", ""), "m"),
                ("GLOBAL_WAYPOINT_NUM", last.get("waypoint_count", ""), "count"),
                ("GLOBAL_MIN_OBS_DIST", last.get("min_obstacle_distance_m", ""), "m"),
                ("LOCAL_TARGET_SWITCH_NUM", self.target_switches, "count"),
                ("GLOBAL_REPLAN_NUM", self.replans, "count"),
            ]
            for metric, value, unit in values:
                writer.writerow({"metric": metric, "value": value, "unit": unit})

    def close(self):
        try:
            self.write_summary()
        finally:
            if not self.fp.closed:
                self.fp.close()


if __name__ == "__main__":
    rospy.init_node("global_planning_metrics_recorder")
    GlobalPlanningMetricsRecorder()
    rospy.spin()
