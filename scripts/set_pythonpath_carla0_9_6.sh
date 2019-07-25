#!/bin/bash
# Assumes it is called from the scripts directory
cur_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH/:$cur_dir/../dependencies/:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py2.7-linux-x86_64.egg:$CARLA_ROOT/PythonAPI/carla/:$cur_dir/../dependencies/CenterNet/src/lib/
