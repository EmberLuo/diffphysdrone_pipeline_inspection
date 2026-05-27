#!/usr/bin/env python3
import copy

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class LocalizationModeManager:
    def __init__(self):
        self.slam_odom_topic = rospy.get_param("~slam_odom_topic", "/slam/odom")
        self.gnss_odom_topic = rospy.get_param("~gnss_odom_topic", "/gnss/odom")
        self.gnss_mode_topic = rospy.get_param("~gnss_mode_topic", "/gnss/mode")
        self.selected_odom_topic = rospy.get_param("~selected_odom_topic", "/Odometry")
        self.localization_mode_topic = rospy.get_param("~localization_mode_topic", "/localization/mode")
        self.publish_rate = float(rospy.get_param("~publish_rate", 50.0))
        self.smoothing_alpha = float(rospy.get_param("~smoothing_alpha", 0.25))
        self.gnss_timeout = float(rospy.get_param("~gnss_timeout", 0.5))
        self.slam_timeout = float(rospy.get_param("~slam_timeout", 0.5))
        self.frame_id = rospy.get_param("~frame_id", "")

        self.latest_slam = None
        self.latest_gnss = None
        self.latest_slam_stamp = rospy.Time(0)
        self.latest_gnss_stamp = rospy.Time(0)
        self.gnss_mode = "lost"
        self.last_output = None

        self.odom_pub = rospy.Publisher(self.selected_odom_topic, Odometry, queue_size=20)
        self.mode_pub = rospy.Publisher(self.localization_mode_topic, String, queue_size=10, latch=True)
        rospy.Subscriber(self.slam_odom_topic, Odometry, self.slam_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.gnss_odom_topic, Odometry, self.gnss_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.gnss_mode_topic, String, self.mode_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_cb)
        rospy.loginfo(
            "localization_mode_manager ready: slam=%s gnss=%s selected=%s",
            self.slam_odom_topic,
            self.gnss_odom_topic,
            self.selected_odom_topic,
        )

    def slam_cb(self, msg):
        self.latest_slam = msg
        self.latest_slam_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def gnss_cb(self, msg):
        self.latest_gnss = msg
        self.latest_gnss_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def mode_cb(self, msg):
        mode = msg.data.strip().lower()
        if mode in ("normal", "degraded", "lost"):
            self.gnss_mode = mode

    def fresh(self, stamp, timeout, now):
        return stamp != rospy.Time(0) and (now - stamp).to_sec() <= timeout

    def choose_source(self, now):
        gnss_fresh = self.fresh(self.latest_gnss_stamp, self.gnss_timeout, now)
        slam_fresh = self.fresh(self.latest_slam_stamp, self.slam_timeout, now)

        if self.gnss_mode == "normal" and gnss_fresh:
            return self.latest_gnss, "gnss"
        if slam_fresh:
            return self.latest_slam, "slam"
        if gnss_fresh:
            return self.latest_gnss, "gnss_stale_mode"
        return None, "unavailable"

    def smooth_position(self, msg):
        if self.last_output is None:
            return msg
        alpha = min(max(self.smoothing_alpha, 0.0), 1.0)
        out = msg
        p = out.pose.pose.position
        prev = self.last_output.pose.pose.position
        p.x = alpha * p.x + (1.0 - alpha) * prev.x
        p.y = alpha * p.y + (1.0 - alpha) * prev.y
        p.z = alpha * p.z + (1.0 - alpha) * prev.z
        return out

    def timer_cb(self, _event):
        now = rospy.Time.now()
        source, mode = self.choose_source(now)
        self.mode_pub.publish(String(data=mode))
        if source is None:
            return

        out = copy.deepcopy(source)
        out.header.stamp = now
        if self.frame_id:
            out.header.frame_id = self.frame_id
        out = self.smooth_position(out)
        self.last_output = copy.deepcopy(out)
        self.odom_pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("localization_mode_manager")
    LocalizationModeManager()
    rospy.spin()
