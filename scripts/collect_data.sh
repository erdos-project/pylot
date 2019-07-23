#!/bin/bash
# $1 path where to save the data.

cd ../
############ Town 01 ############
${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town01 -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
mkdir $1/town01_start1/
sleep 10
python data_gatherer.py --flagfile=configs/data_gatherer_autopilot.conf --carla_start_player_num=1 --data_path=$1/town01_start1/ & sleep 3600; kill -9 $!
# Collect data for an hour.
# Kill data gathering script and Carla.
killall python data_gatherer.py
sleep 10
echo ${CARLA_ROOT}
killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4
sleep 10

############ Town 02 ############
${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town02 -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
mkdir $1/town02_start1/
sleep 10
python data_gatherer.py --flagfile=configs/data_gatherer_autopilot.conf --carla_start_player_num=1 --data_path=$1/town02_start1/ & sleep 3600 ; kill -9 $!
# Collect data for an hour.
# Kill data gathering script and Carla.
killall python data_gatherer.py
sleep 10
echo ${CARLA_ROOT}
killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4
sleep 10

############ Town 03 ############
${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town03 -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
mkdir $1/town03_start1/
sleep 10
python data_gatherer.py --flagfile=configs/data_gatherer_autopilot.conf --carla_start_player_num=1 --data_path=$1/town03_start1/ & sleep 3600 ; kill -9 $!
# Collect data for an hour.
# Kill data gathering script and Carla.
killall python data_gatherer.py
sleep 10
echo ${CARLA_ROOT}
killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4
sleep 10
