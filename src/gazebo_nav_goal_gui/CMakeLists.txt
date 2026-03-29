cmake_minimum_required(VERSION 3.0.2)
project(gazebo_nav_goal_gui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  gazebo_ros
)

find_package(gazebo REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets Core)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gazebo_goal_tool gazebo_nav_goal_gui
  CATKIN_DEPENDS roscpp geometry_msgs gazebo_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})
add_definitions(${GAZEBO_CXX_FLAGS})

add_library(gazebo_goal_tool SHARED
  src/gazebo_goal_tool.cc
)

add_library(gazebo_nav_goal_gui SHARED
  src/gazebo_nav_goal_gui.cpp
)

target_link_libraries(gazebo_goal_tool
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  Qt5::Widgets
  Qt5::Core
)

target_link_libraries(gazebo_nav_goal_gui
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  Qt5::Widgets
  Qt5::Core
)

add_dependencies(gazebo_goal_tool
  ${catkin_EXPORTED_TARGETS}
)

add_dependencies(gazebo_nav_goal_gui
  ${catkin_EXPORTED_TARGETS}
)
