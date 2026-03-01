# MRVK ROS Workspace

## Install

```sh
$ cd ${ROS_WS}
$ rosdep install --from-paths src --ignore-src -r -y
$ catkin_make
$ source ./devel/setup.bash
$ export GAZEBO_MODEL_DATABASE_URI=http://models.gazebosim.org
```

## Launch simulation

In file [robot.launch](./mrvk_gazebo/launch/robot.launch) choose the world to be launched (uncomment or rewrite) and set initial coordinates.

```sh
$ roslaunch mrvk_gazebo robot.launch
```
### Launch Path Detection

* launch `path_detection_node`
```sh
$ rosrun path_detection path_detection_node.py
```
* set navigation goal by publishing on `/move_base/goal` topic
