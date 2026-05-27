cmake_minimum_required(VERSION 3.0.2)
project(pcd_localization)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  pcl_ros
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
)
find_package(PCL 1.7 REQUIRED COMPONENTS common io filters registration)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(localization_node src/localization_node.cpp)
target_link_libraries(localization_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS localization_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
