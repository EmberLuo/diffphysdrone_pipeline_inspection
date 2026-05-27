#!/usr/bin/env python3

from threading import Lock

import rospy
import rostopic
import sensor_msgs.point_cloud2 as pc2
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from sensor_msgs.msg import PointCloud, PointCloud2


class PointCloudToLivoxNode:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/livox/lidar2")
        self.output_topic = rospy.get_param("~output_topic", "/livox/lidar")
        self.input_type = rospy.get_param("~input_type", "auto").strip().lower()
        self.frame_id = rospy.get_param("~frame_id", "").strip()
        self.lidar_id = int(rospy.get_param("~lidar_id", 1))
        self.scan_period = float(rospy.get_param("~scan_period", 0.1))

        self._lock = Lock()
        self._sub = None
        self._pub = rospy.Publisher(self.output_topic, CustomMsg, queue_size=5)

        self._setup_subscriber()

    def _setup_subscriber(self):
        msg_class = self._resolve_input_class()
        if msg_class is PointCloud2:
            rospy.loginfo("Subscribed as PointCloud2: %s", self.input_topic)
            self._sub = rospy.Subscriber(
                self.input_topic, PointCloud2, self._pointcloud2_cb, queue_size=5
            )
        else:
            rospy.loginfo("Subscribed as PointCloud: %s", self.input_topic)
            self._sub = rospy.Subscriber(
                self.input_topic, PointCloud, self._pointcloud_cb, queue_size=5
            )

    def _resolve_input_class(self):
        if self.input_type in ("pointcloud2", "pc2", "sensor_msgs/pointcloud2"):
            return PointCloud2
        if self.input_type in ("pointcloud", "pc", "sensor_msgs/pointcloud"):
            return PointCloud

        topic_type, _, _ = rostopic.get_topic_type(self.input_topic, blocking=False)
        if topic_type == "sensor_msgs/PointCloud":
            return PointCloud
        if topic_type == "sensor_msgs/PointCloud2":
            return PointCloud2

        rospy.logwarn(
            "Cannot auto-detect type for %s yet, fallback to PointCloud2. "
            "You can set ~input_type:=pointcloud if needed.",
            self.input_topic,
        )
        return PointCloud2

    def _pointcloud_cb(self, msg):
        points = [(p.x, p.y, p.z, 0.0) for p in msg.points]
        self._publish_custom(points, msg.header)

    def _pointcloud2_cb(self, msg):
        fields = {field.name for field in msg.fields}
        intensity_field = None
        if "intensity" in fields:
            intensity_field = "intensity"
        elif "reflectivity" in fields:
            intensity_field = "reflectivity"

        field_names = ("x", "y", "z", intensity_field) if intensity_field else ("x", "y", "z")
        raw_points = pc2.read_points(msg, field_names=field_names, skip_nans=True)
        if intensity_field:
            points = list(raw_points)
        else:
            points = [(x, y, z, 0.0) for x, y, z in raw_points]
        self._publish_custom(points, msg.header)

    def _publish_custom(self, points_xyzi, header):
        if not points_xyzi:
            return

        with self._lock:
            out = CustomMsg()
            out.header = header
            if self.frame_id:
                out.header.frame_id = self.frame_id

            stamp = out.header.stamp if out.header.stamp and not out.header.stamp.is_zero() else rospy.Time.now()
            out.header.stamp = stamp
            out.timebase = stamp.to_nsec()
            out.lidar_id = self.lidar_id
            out.rsvd = [0, 0, 0]
            out.point_num = len(points_xyzi)

            period_ns = int(max(self.scan_period, 0.0) * 1e9)
            denom = max(len(points_xyzi), 1)
            step_ns = int(period_ns / denom) if period_ns > 0 else 0

            for idx, (x, y, z, intensity) in enumerate(points_xyzi):
                p = CustomPoint()
                p.offset_time = idx * step_ns
                p.x = float(x)
                p.y = float(y)
                p.z = float(z)
                if hasattr(p, "reflectivity"):
                    p.reflectivity = int(max(0, min(255, round(float(intensity)))))
                p.tag = 0
                p.line = 0
                out.points.append(p)

            self._pub.publish(out)


def main():
    rospy.init_node("pointcloud_to_livox")
    PointCloudToLivoxNode()
    rospy.loginfo("pointcloud_to_livox is running.")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
