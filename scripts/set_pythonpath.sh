#!/bin/bash
# Assumes it is called from the scripts directory
cur_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH/:$cur_dir/../dependencies/:$CARLA_ROOT/PythonClient:$cur_dir/../dependencies/CenterNet/src/lib/
