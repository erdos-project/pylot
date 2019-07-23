#!/bin/bash
# $1 path where to save the data.

############ Town 01 ############
${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town01 -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
mkdir $1/signs_town01/
sleep 5
python sign_data_gatherer.py --data_path=$1/signs_town01/ --log_bbox_images
sleep 10
killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4
sleep 10

############ Town 02 ############
${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town02 -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
mkdir $1/signs_town02/
sleep 5
python sign_data_gatherer.py --data_path=$1/signs_town02/ --log_bbox_images
sleep 10
killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4
sleep 10

############ Town 03 ############
${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town03 -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
mkdir $1/signs_town03/
sleep 5
python sign_data_gatherer.py --data_path=$1/signs_town03/ --log_bbox_images
sleep 10
killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4
sleep 10

############ Town 04 ############
${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town04 -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
mkdir $1/signs_town04/
sleep 5
python sign_data_gatherer.py --data_path=$1/signs_town04/ --log_bbox_images
sleep 10
killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4
sleep 10

############ Town 05 ############
${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town05 -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
mkdir $1/signs_town05/
sleep 5
python sign_data_gatherer.py --data_path=$1/signs_town05/ --log_bbox_images
sleep 10
killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4
sleep 10
