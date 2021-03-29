#!/bin/bash
# $1 path where to save the data.

if [ -z "$PYLOT_HOME" ]; then
    echo "Please set \$PYLOT_HOME before sourcing this script"
    exit 1
fi

if [ -z "$CARLA_HOME" ]; then
    echo "Please set \$CARLA_HOME before running this script"
    exit 1
fi

start_player_nums=(1 10 20 30 40)
towns=(1 2 3 4 5)

cd $PYLOT_HOME
for pn in ${start_player_nums[@]}; do
    for town in ${towns[@]}; do
        echo "[x] Driving in town $town from start position $pn"
        echo "[x] Starting the CARLA simulator"
        SDL_VIDEODRIVER=offscreen ${CARLA_HOME}/CarlaUE4.sh -opengl /Game/Carla/Maps/Town0${town} -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
        sleep 10
        mkdir $1/town0${town}_start${pn}/
        python3 data_gatherer.py --flagfile=configs/data_gatherer.conf --carla_spawn_point_index=${pn} --data_path=$1/town0${town}_start${pn}/ --simulator_town=${town} &
        # Collect data for an hour.
        sleep 4800
        # Kill data gathering script and Carla.
        pkill -9 -f -u $USER data_gatherer
        pkill -9 -f -u $USER CarlaUE4
        sleep 5
    done
done
