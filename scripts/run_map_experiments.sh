#!/bin/bash

# Assumes that $CARLA_HOME AND $PYLOT_HOME are set.
if [ -z "$PYLOT_HOME" ]; then
    echo "Please set \$PYLOT_HOME before sourcing this script"
    exit 1
fi

if [ -z "$CARLA_HOME" ]; then
    echo "Please set \$CARLA_HOME before running this script"
    exit 1
fi

if [ -z "$SCENARIO_RUNNER_HOME" ]; then
    echo "Please set \$SCENARIO_RUNNER_HOME before running this script"
    exit 1
fi

speeds=( 10 15 20 25 30 35 40 )
SAMPLING_RATE=0.005 # The delta between two subsequent frames.

cd $PYLOT_HOME
for speed in ${speeds[@]}; do
    echo "[x] Running the experiment with the speed $speed and the sampling rate of $SAMPLING_RATE"
    # Start the simulator.
    echo "[x] Starting the CARLA Simulator..."
    SDL_VIDEODRIVER=offscreen ${CARLA_HOME}/CarlaUE4.sh -opengl -windowed -ResX=800 -ResY=600 -carla-server -benchmark -fps=20 -quality-level=Epic &
    sleep 10
    
    # Start the scenario runner.
    echo "[x] Starting the scenario runner..."
    cd $SCENARIO_RUNNER_HOME
    python3 scenario_runner.py --scenario ERDOSBenchmarks_2 --reloadWorld &
    sleep 5  

    # Start the mIoU script. 
    cd $PYLOT_HOME
    python3 scripts/map_scenario_runner.py -s $speed -d $SAMPLING_RATE -o results_map_$speed.csv

    # Kill the scenario runner.
    pkill -9 -f -u $USER scenario_runner
    # Kill the simulator.
    pkill -9 -f -u $USER CarlaUE4
done
