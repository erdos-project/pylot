#!/bin/bash

if [ -z "$PYLOT_HOME" ]; then
    echo "Please set \$PYLOT_HOME before sourcing this script"
    exit 1
fi
if [ -z "$CARLA_HOME" ]; then
    echo "Please set \$CARLA_HOME before sourcing this script"
    exit 1
fi

CARLA_EGG=$(ls $CARLA_HOME/PythonAPI/carla/dist/carla*py3*egg)
export PYTHONPATH=$PYTHONPATH:$PYLOT_HOME:/$PYLOT_HOME/dependencies/:$CARLA_EGG:$CARLA_HOME/PythonAPI/carla/
