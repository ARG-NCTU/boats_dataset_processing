#! /bin/bash

export ROS1_INSTALL_PATH=/opt/ros/noetic

cd ros1_ws
catkin build
cd ..

source ${ROS1_INSTALL_PATH}/setup.bash
source ./ros1_ws/devel/setup.bash