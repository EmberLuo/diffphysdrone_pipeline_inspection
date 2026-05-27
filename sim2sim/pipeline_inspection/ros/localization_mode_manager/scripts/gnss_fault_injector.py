#!/usr/bin/env python3
import copy
import math
import random
import time

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class GnssFaultInjector:
    def __init__(self):
        self.source_odom_topic = rospy.get_param("~source_odom_topic", "/slam/odom")
        self.gnss_odom_topic = rospy.get_param("~gnss_odom_topic", "/gnss/odom")
        self.mode_topic = rospy.get_param("~mode_topic", "/gnss/mode")
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))
        self.normal_duration = float(rospy.get_param("~normal_duration", 20.0))
        self.degraded_duration = float(rospy.get_param("~degraded_duration", 20.0))
        self.normal_noise_std = float(rospy.get_param("~normal_noise_std", 0.05))
        self.degraded_noise_std = float(rospy.get_param("~degraded_noise_std", 0.5))
        self.jump_magnitude = float(rospy.get_param("~jump_magnitude", 1.5))
        self.jump_period = float(rospy.get_param("~jump_period", 5.0))
        self.seed = int(rospy.get_param("~seed", 7))
        self.frame_id = rospy.get_param("~frame_id", "")
        self.child_frame_id = rospy.get_param("~child_frame_id", "gnss")

        self.rng = random.Random(self.seed)
        self.start_time = time.monotonic()
        self.latest_source = None
        self.current_jump = (0.0, 0.0, 0.0)
        self.current_jump_index = None

        self.odom_pub = rospy.Publisher(self.gnss_odom_topic, Odometry, queue_size=20)
        self.mode_pub = rospy.Publisher(self.mode_topic, String, queue_size=10, latch=True)
        rospy.Subscriber(self.source_odom_topic, Odometry, self.source_cb, queue_size=1, tcp_nodelay=True)
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_cb)
        rospy.loginfo(
            "gnss_fault_injector ready: source=%s gnss=%s mode=%s",
            self.source_odom_topic,
            self.gnss_odom_topic,
            self.mode_topic,
        )

    def source_cb(self, msg):
        self.latest_source = msg

    def mode_at(self):
        elapsed = time.monotonic() - self.start_time
        if elapsed < self.normal_duration:
            return "normal", elapsed
        if elapsed < self.normal_duration + self.degraded_duration:
            return "degraded", elapsed - self.normal_duration
        return "lost", elapsed - self.normal_duration - self.degraded_duration

    def update_jump(self, degraded_elapsed):
        if self.jump_period <= 0.0 or self.jump_magnitude <= 0.0:
            self.current_jump = (0.0, 0.0, 0.0)
            self.current_jump_index = None
            return

        jump_index = int(math.floor(degraded_elapsed / self.jump_period))
        if jump_index == self.current_jump_index:
            return
        self.current_jump_index = jump_index
        yaw = self.rng.uniform(-math.pi, math.pi)
        z = self.rng.uniform(-0.25, 0.25) * self.jump_magnitude
        self.current_jump = (
            self.jump_magnitude * math.cos(yaw),
            self.jump_magnitude * math.sin(yaw),
            z,
        )

    def noise3(self, std):
        return (
            self.rng.gauss(0.0, std),
            self.rng.gauss(0.0, std),
            self.rng.gauss(0.0, std * 0.4),
        )

    def timer_cb(self, _event):
        now = rospy.Time.now()
        mode, mode_elapsed = self.mode_at()
        self.mode_pub.publish(String(data=mode))
        if self.latest_source is None or mode == "lost":
            return

        msg = copy.deepcopy(self.latest_source)
        msg.header.stamp = now
        if self.frame_id:
            msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id

        std = self.normal_noise_std if mode == "normal" else self.degraded_noise_std
        noise = self.noise3(std)
        jump = (0.0, 0.0, 0.0)
        if mode == "degraded":
            self.update_jump(mode_elapsed)
            jump = self.current_jump

        p = msg.pose.pose.position
        p.x += noise[0] + jump[0]
        p.y += noise[1] + jump[1]
        p.z += noise[2] + jump[2]

        cov = max(std * std, 1e-6)
        msg.pose.covariance = list(msg.pose.covariance)
        msg.pose.covariance[0] = cov
        msg.pose.covariance[7] = cov
        msg.pose.covariance[14] = cov
        self.odom_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("gnss_fault_injector")
    GnssFaultInjector()
    rospy.spin()
