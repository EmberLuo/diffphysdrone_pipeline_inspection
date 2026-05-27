cmake_minimum_required(VERSION 3.0.2)
project(global_path_target_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/global_path_target_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
