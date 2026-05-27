cmake_minimum_required(VERSION 3.0.2)
project(navigation_bringup)

find_package(catkin REQUIRED COMPONENTS rospy)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/pointcloud2livox_wrapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
