cmake_minimum_required(VERSION 2.8.3)
project(mrvk_gazebo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  gazebo_msgs
  visualization_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    geometry_msgs
    nav_msgs
    gazebo_msgs
    visualization_msgs
    sensor_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(dynamic_obstacle_predictor src/dynamic_obstacle_predictor.cpp)

target_link_libraries(dynamic_obstacle_predictor
  ${catkin_LIBRARIES}
)

add_dependencies(dynamic_obstacle_predictor
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)