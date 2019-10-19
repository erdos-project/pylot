CARLA_PATH=../../CARLA_0.9.6
SCENARIO_RUNNER_PATH=../../scenario_runner

speeds=( 10 15 20 25 30 35 40 )
SAMPLING_RATE=0.005 # The delta between two subsequent frames.

for speed in ${speeds[@]}; do
    echo "[x] Running the experiment with the speed $speed and the sampling rate of $SAMPLING_RATE"
    # Start the simulator.
    echo "[x] Starting the Carla Simulator 0.9.6"
    ./$CARLA_PATH/CarlaUE4.sh & 
    sleep 10
    
    # Start the scenario runner.
    echo "[x] Starting the scenario runner."
    pushd $SCENARIO_RUNNER_PATH > /dev/null
    python scenario_runner.py --scenario ERDOSBenchmarks_2 --reloadWorld &
    SCENARIO_RUNNER_PID=$!
    sleep 15  

    # Start the mIoU script. 
    popd > /dev/null 
    python map_scenario_runner.py -s $speed -d $SAMPLING_RATE -o results_map_$speed.csv

    # Kill the scenario runner.
    kill -9 $SCENARIO_RUNNER_PID
    
    # Kill the simulator.
    `ps aux | grep CarlaUE4 | awk '{print $2}' | head -n -1 | xargs kill -9`
done
