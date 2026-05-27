cmake_minimum_required(VERSION 3.0.2)
project(global_astar_planner)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  pcl_ros
  pcl_conversions
  plan_env
  path_searching
)
find_package(PCL 1.7 REQUIRED COMPONENTS common io filters)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(global_astar_planner_node src/global_astar_planner_node.cpp)
target_link_libraries(global_astar_planner_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS global_astar_planner_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
