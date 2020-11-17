#!/bin/bash
# $1 path where to save the data.

if [ -z "$CARLA_HOME" ]; then
    echo "Please set \$CARLA_HOME before running this script"
    exit 1
fi

towns=("Town01" "Town02")

# towns=("Town05")

for town in ${towns[@]}; do
    SDL_VIDEODRIVER=offscreen ${CARLA_HOME}/CarlaUE4.sh -opengl -windowed -ResX=800 -ResY=600 -carla-server -benchmark -fps=20 -quality-level=Epic &
    sleep 10
    # Change the town using the config file.
    # Do not change Town03 because we start in Town03 and running config.py
    # randomizes the identifiers for the traffic lights.
    if [ $town != "Town03" ]; then
        ${CARLA_HOME}/PythonAPI/util/config.py --map $town
    fi
    mkdir -p $1/signs_${town}/
    sleep 5
    python sign_data_gatherer.py --data_path=$1/signs_${town}/ --log_bbox_images
    sleep 10
    killall -s 9 ${CARLA_HOME}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
    sleep 10
done
