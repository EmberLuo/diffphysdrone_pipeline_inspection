#!/usr/bin/env python3
import os
import subprocess
import sys

import rospy


if __name__ == "__main__":
    rospy.init_node("pointcloud2livox_wrapper", anonymous=False)
    repo_root = os.environ.get("REPO_ROOT")
    if not repo_root:
        this_file = os.path.realpath(__file__)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(this_file), "../../.."))
    script = os.path.join(repo_root, "tools", "pointcloud2livox.py")
    args = [
        sys.executable,
        script,
        "_input_topic:=%s" % rospy.get_param("~input_topic", "/livox/lidar2"),
        "_output_topic:=%s" % rospy.get_param("~output_topic", "/livox/lidar"),
        "_input_type:=%s" % rospy.get_param("~input_type", "pointcloud"),
        "_frame_id:=%s" % rospy.get_param("~frame_id", ""),
        "_lidar_id:=%s" % rospy.get_param("~lidar_id", 1),
        "_scan_period:=%s" % rospy.get_param("~scan_period", 0.1),
    ]
    sys.exit(subprocess.call(args))
