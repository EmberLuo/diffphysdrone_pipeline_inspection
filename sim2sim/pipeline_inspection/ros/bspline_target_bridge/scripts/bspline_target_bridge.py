#!/usr/bin/env python3
import bisect
import math

import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from traj_utils.msg import Bspline
from visualization_msgs.msg import Marker


class BsplineCurve:
    def __init__(self, msg):
        self.order = int(msg.order)
        # Fast-Drone's Bspline.msg uses "order" as the polynomial degree.
        self.degree = self.order
        self.knots = list(msg.knots)
        self.points = [(p.x, p.y, p.z) for p in msg.pos_pts]
        self.start_time = msg.start_time
        self.traj_id = msg.traj_id

        if not self.knots or not self.points:
            raise ValueError("empty B-spline")
        if len(self.knots) < len(self.points) + self.degree + 1:
            raise ValueError("invalid knot/control-point count")

        self.t_min = self.knots[self.degree]
        self.t_max = self.knots[len(self.points)]

    def evaluate(self, t):
        t = max(self.t_min, min(self.t_max, t))
        k = bisect.bisect_right(self.knots, t) - 1
        k = max(self.degree, min(k, len(self.points) - 1))

        d = [list(self.points[j]) for j in range(k - self.degree, k + 1)]
        for r in range(1, self.degree + 1):
            for j in range(self.degree, r - 1, -1):
                i = k - self.degree + j
                denom = self.knots[i + self.degree + 1 - r] - self.knots[i]
                alpha = 0.0 if abs(denom) < 1e-9 else (t - self.knots[i]) / denom
                for axis in range(3):
                    d[j][axis] = (1.0 - alpha) * d[j - 1][axis] + alpha * d[j][axis]
        return d[self.degree]


class BsplineTargetBridge:
    def __init__(self):
        self.bspline_topic = rospy.get_param("~bspline_topic", "/drone_0_planning/bspline")
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.target_topic = rospy.get_param("~target_topic", "/e2e/local_target")
        self.lookahead_time = float(rospy.get_param("~lookahead_time", 1.0))
        self.extra_lookahead_time = float(rospy.get_param("~extra_lookahead_time", 0.5))
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))
        self.min_target_distance = float(rospy.get_param("~min_target_distance", 0.4))
        self.marker_topic = rospy.get_param("~marker_topic", "/e2e/local_target_marker")

        self.curve = None
        self.odom = None
        self.target_pub = rospy.Publisher(self.target_topic, PointStamped, queue_size=20)
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=1)

        rospy.Subscriber(self.bspline_topic, Bspline, self.bspline_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1, tcp_nodelay=True)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_cb)
        rospy.loginfo("bspline_target_bridge ready: bspline=%s odom=%s target=%s",
                      self.bspline_topic, self.odom_topic, self.target_topic)

    def bspline_cb(self, msg):
        try:
            self.curve = BsplineCurve(msg)
            rospy.loginfo("Received B-spline traj_id=%s duration=%.2f",
                          msg.traj_id, self.curve.t_max - self.curve.t_min)
        except ValueError as exc:
            rospy.logwarn("Ignoring invalid B-spline: %s", exc)

    def odom_cb(self, msg):
        self.odom = msg

    def timer_cb(self, _event):
        if self.curve is None or self.odom is None:
            return
        now = rospy.Time.now()
        t = (now - self.curve.start_time).to_sec() + self.lookahead_time
        target = self.curve.evaluate(t)

        pos = self.odom.pose.pose.position
        dx = target[0] - pos.x
        dy = target[1] - pos.y
        dz = target[2] - pos.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < self.min_target_distance and t < self.curve.t_max:
            target = self.curve.evaluate(t + self.extra_lookahead_time)

        msg = PointStamped()
        msg.header.stamp = now
        msg.header.frame_id = "world"
        msg.point.x, msg.point.y, msg.point.z = target
        self.target_pub.publish(msg)
        self.publish_marker(msg)

    def publish_marker(self, target):
        if self.marker_pub.get_num_connections() <= 0:
            return
        marker = Marker()
        marker.header = target.header
        marker.ns = "e2e_local_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = target.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.3
        marker.color.a = 0.9
        self.marker_pub.publish(marker)


if __name__ == "__main__":
    rospy.init_node("bspline_target_bridge")
    BsplineTargetBridge()
    rospy.spin()
