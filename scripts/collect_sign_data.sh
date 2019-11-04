#!/bin/bash
# $1 path where to save the data.

if [ -z "$CARLA_HOME" ]; then
    echo "Please set \$CARLA_HOME before running this script"
    exit 1
fi

towns=(1 2 3 4 5)

for town in ${towns[@]}; do
    ${CARLA_HOME}/CarlaUE4.sh /Game/Carla/Maps/Town0${town} -windowed -ResX=1920 -ResY=1080 -carla-server -benchmark -fps=10 &
    mkdir $1/signs_town0${town}/
    sleep 5
    python sign_data_gatherer.py --data_path=$1/signs_town0${town}/ --log_bbox_images
    sleep 10
    killall -s 9 ${CARLA_HOME}/CarlaUE4/Binaries/Linux/CarlaUE4
    sleep 10
done
