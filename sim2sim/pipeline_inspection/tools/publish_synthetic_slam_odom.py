#!/usr/bin/env python3
"""Publish a smooth SLAM odometry trajectory for GNSS-mode node tests."""

from __future__ import annotations

import argparse
import math

import rospy
from nav_msgs.msg import Odometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/slam/odom")
    parser.add_argument("--frame_id", default="world")
    parser.add_argument("--child_frame_id", default="base_link")
    parser.add_argument("--duration", type=float, default=27.0)
    parser.add_argument("--rate", type=float, default=30.0)
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    rospy.init_node("publish_synthetic_slam_odom")
    args = parse_args()
    pub = rospy.Publisher(args.topic, Odometry, queue_size=20)
    rate = rospy.Rate(args.rate)
    t0 = rospy.Time.now()

    while not rospy.is_shutdown():
        t = (rospy.Time.now() - t0).to_sec()
        if t > args.duration:
            break

        s = t / max(args.duration, 1e-6)
        x = 12.0 * s
        y = 1.6 * math.sin(2.0 * math.pi * s) + 0.35 * math.sin(6.0 * math.pi * s)
        z = 1.8 + 0.15 * math.sin(4.0 * math.pi * s)

        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = args.frame_id
        msg.child_frame_id = args.child_frame_id
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z
        msg.pose.pose.orientation.w = 1.0
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
