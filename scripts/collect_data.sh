#!/bin/bash
# $1 path where to save the data.

# Change the Carla kill command if you want to collect data with 0.8.4 or 0.9.5.
cd ../

start_player_nums=(1 10 20 30 40)
towns=(1 2 3 4 5)

for pn in ${start_player_nums[@]}; do
    for town in ${towns[@]}; do
        ${CARLA_ROOT}CarlaUE4.sh /Game/Carla/Maps/Town0${town} -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
        mkdir $1/town0${town}_start${pn}/
        sleep 10
        python data_gatherer.py --flagfile=configs/data_gatherer_autopilot.conf --carla_start_player_num=${pn} --data_path=$1/town0${town}_start${pn}/ --carla_town=${town} & sleep 4800; kill -9 $!
        # Collect data for an hour.
        # Kill data gathering script and Carla.
        killall python data_gatherer.py
        sleep 10
        killall -s 9 ${CARLA_ROOT}CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
        sleep 10
    done
done
