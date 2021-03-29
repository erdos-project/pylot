#!/bin/bash
# $1 path where to save the data.

if [ -z "$CARLA_HOME" ]; then
    echo "Please set \$CARLA_HOME before running this script"
    exit 1
fi

towns=("Town01" "Town02")

for town in ${towns[@]}; do
    echo "[x] Starting the CARLA Simulator.."
    SDL_VIDEODRIVER=offscreen ${CARLA_HOME}/CarlaUE4.sh -opengl -windowed -ResX=800 -ResY=600 -carla-server -benchmark -fps=20 -quality-level=Epic &
    sleep 10
    # Change the town using the config file.
    # Do not change Town03 because we start in Town03 and running config.py
    # randomizes the identifiers for the traffic lights.
    if [ $town != "Town03" ]; then
        ${CARLA_HOME}/PythonAPI/util/config.py --map $town
        sleep 5
    fi

    mkdir -p $1/signs_${town}/
    echo "[x] Starting the sign data gatherer..."
    python3 sign_data_gatherer.py --data_path=$1/signs_${town}/ --log_bbox_images

    # Kill the simulator.
    pkill -9 -f -u $USER CarlaUE4
    sleep 5
done
