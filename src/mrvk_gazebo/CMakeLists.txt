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
  costmap_2d
  pluginlib
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES predicted_sweep_risk_layer
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
    costmap_2d
    pluginlib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(dynamic_obstacle_predictor
  src/dynamic_obstacle_predictor.cpp
)

add_executable(safe_zone_goal_router
  src/safe_zone_goal_router.cpp
)

target_link_libraries(dynamic_obstacle_predictor
  ${catkin_LIBRARIES}
)

target_link_libraries(safe_zone_goal_router
  ${catkin_LIBRARIES}
)

add_dependencies(dynamic_obstacle_predictor
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

add_dependencies(safe_zone_goal_router
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

add_library(predicted_sweep_risk_layer
  src/predicted_sweep_risk_layer.cpp
)

target_link_libraries(predicted_sweep_risk_layer
  ${catkin_LIBRARIES}
)

add_dependencies(predicted_sweep_risk_layer
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

install(TARGETS
  dynamic_obstacle_predictor
  safe_zone_goal_router
  predicted_sweep_risk_layer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/safe_zone_goal_router.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES predicted_sweep_risk_layer.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
