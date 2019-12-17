#!/bin/bash

if [ -z "$PYLOT_HOME" ]; then
    echo "Please set \$PYLOT_HOME before sourcing this script"
    exit 1
fi
if [ -z "$CARLA_HOME" ]; then
    echo "Please set \$CARLA_HOME before sourcing this script"
    exit 1
fi

# Get the carla version.
CARLA_VERSION=$(cat $CARLA_HOME/VERSION)
if [ -z "$CARLA_VERSION" ]; then
    echo "Please set \$CARLA_VERSION before sourcing this script"
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$PYLOT_HOME/dependencies/:$CARLA_HOME/PythonAPI/carla/dist/carla-$CARLA_VERSION-py3.5-linux-x86_64.egg:$CARLA_HOME/PythonAPI/carla/
