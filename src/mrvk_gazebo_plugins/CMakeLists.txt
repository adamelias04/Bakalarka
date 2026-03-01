cmake_minimum_required(VERSION 3.0.2)
project(mrvk_gazebo_plugins)

find_package(catkin REQUIRED)
find_package(gazebo REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES oscillate_box_plugin
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(oscillate_box_plugin SHARED
  src/oscillate_box_plugin.cpp
)

target_link_libraries(oscillate_box_plugin
  ${GAZEBO_LIBRARIES}
)

install(TARGETS oscillate_box_plugin
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

