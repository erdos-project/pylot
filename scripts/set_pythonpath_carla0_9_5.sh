#!/bin/bash
# Assumes it is called from the scripts directory
cur_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH/:$cur_dir/../dependencies/:$cur_dir/../dependencies/CARLA_0.9.5/PythonAPI/carla/dist/carla-0.9.5-py2.7-linux-x86_64.egg:$cur_dir/../dependencies/CARLA_0.9.5/PythonAPI/carla/:$cur_dir/../dependencies/CenterNet/src/lib/
