export CARLA_ROOT=/home/efang96/Desktop/pylot/dependencies/CARLA_0.9.6
source /opt/ros/melodic/setup.sh
source ./set_pythonpath_carla0_9_6.sh

CARLA_PATH=/home/efang96/Desktop/pylot/dependencies/CARLA_0.9.6
SCENARIO_RUNNER_PATH=/home/efang96/Desktop/scenario_runner

SAMPLING_RATE=0.1 # The delta between two subsequent frames.

# Start the simulator.
echo "[x] Starting the Carla Simulator 0.9.6"
$CARLA_PATH/CarlaUE4.sh &
sleep 10

# Start the scenario runner.
echo "[x] Starting the scenario runner."
pushd $SCENARIO_RUNNER_PATH > /dev/null
python scenario_runner.py --scenario ERDOSBenchmarks_1 --reloadWorld &
SCENARIO_RUNNER_PID=$!
sleep 15

# Start the mIoU script.
popd > /dev/null
python plan_scenario_runner.py

# Kill the scenario runner.
kill -9 $SCENARIO_RUNNER_PID

# Kill the simulator.
ps aux | grep CarlaUE4 | awk '{print $2}' | head -n -1 | xargs kill -9