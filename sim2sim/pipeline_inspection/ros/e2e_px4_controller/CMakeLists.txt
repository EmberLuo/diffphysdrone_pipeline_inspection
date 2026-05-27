cmake_minimum_required(VERSION 3.0.2)
project(e2e_px4_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  mavros_msgs
  cv_bridge
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/e2e_px4_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
