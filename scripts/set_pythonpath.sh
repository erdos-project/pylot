#!/bin/bash
# Assumes it is called from the scripts directory
cur_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH/:$cur_dir/../dependencies/:$cur_dir/../dependencies/CARLA_0.8.4/PythonClient:$cur_dir/../../../:$cur_dir/../dependencies/CenterNet/src/lib/
