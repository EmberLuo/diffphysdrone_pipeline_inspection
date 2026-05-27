cmake_minimum_required(VERSION 3.0.2)
project(map_tools)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  pcl_conversions
)
find_package(PCL 1.7 REQUIRED COMPONENTS common io filters)
find_package(Boost REQUIRED COMPONENTS filesystem system)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable(prepare_pcd_map src/prepare_pcd_map.cpp)
target_link_libraries(prepare_pcd_map
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

install(TARGETS prepare_pcd_map
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
