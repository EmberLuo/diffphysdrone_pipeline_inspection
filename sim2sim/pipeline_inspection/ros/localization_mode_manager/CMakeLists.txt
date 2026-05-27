cmake_minimum_required(VERSION 3.0.2)
project(localization_mode_manager)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  std_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/gnss_fault_injector.py
  scripts/localization_mode_manager.py
  scripts/gnss_mode_metrics_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
