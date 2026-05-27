#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path


class GlobalPathTargetBridge:
    def __init__(self):
        self.path_topic = rospy.get_param("~path_topic", "/global_path")
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.target_topic = rospy.get_param("~target_topic", "/e2e/local_target")
        self.lookahead_distance = float(rospy.get_param("~lookahead_distance", 2.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.goal_hold_distance = float(rospy.get_param("~goal_hold_distance", 0.4))
        self.frame_id = rospy.get_param("~frame_id", "world")

        self.path = []
        self.odom = None
        self.path_stamp = rospy.Time(0)

        self.pub = rospy.Publisher(self.target_topic, PointStamped, queue_size=10)
        rospy.Subscriber(self.path_topic, Path, self.path_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1, tcp_nodelay=True)
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_cb)
        rospy.loginfo("global_path_target_bridge ready: path=%s target=%s", self.path_topic, self.target_topic)

    def path_cb(self, msg):
        self.path = [
            (p.pose.position.x, p.pose.position.y, p.pose.position.z)
            for p in msg.poses
        ]
        self.path_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def odom_cb(self, msg):
        self.odom = msg

    @staticmethod
    def dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def choose_target(self, pos):
        if not self.path:
            return None
        nearest_i = min(range(len(self.path)), key=lambda i: self.dist(pos, self.path[i]))
        if self.dist(pos, self.path[-1]) <= self.goal_hold_distance:
            return self.path[-1]

        remain = self.lookahead_distance
        prev = self.path[nearest_i]
        for i in range(nearest_i + 1, len(self.path)):
            cur = self.path[i]
            seg = self.dist(prev, cur)
            if seg >= remain and seg > 1e-6:
                t = remain / seg
                return (
                    prev[0] + (cur[0] - prev[0]) * t,
                    prev[1] + (cur[1] - prev[1]) * t,
                    prev[2] + (cur[2] - prev[2]) * t,
                )
            remain -= seg
            prev = cur
        return self.path[-1]

    def timer_cb(self, _event):
        if self.odom is None or not self.path:
            return
        p = self.odom.pose.pose.position
        target = self.choose_target((p.x, p.y, p.z))
        if target is None:
            return
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.point.x, msg.point.y, msg.point.z = target
        self.pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("global_path_target_bridge")
    GlobalPathTargetBridge()
    rospy.spin()
