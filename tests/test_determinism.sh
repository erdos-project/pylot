#!/bin/bash

hzs=(10)
detection_runtimes=(200)
detectors=(efficientdet-d6)
planning_runtimes=(309)
target_speeds=(10)
file_prefix_name=traffic_jam
scenario=ERDOSTrafficJam


for hz in ${hzs[@]}; do
    for run in `seq 1 5`; do
        for target_speed in ${target_speeds[@]}; do
            for d_index in ${!detection_runtimes[@]}; do
                for planning in ${planning_runtimes[@]}; do
                    file_base=${file_prefix_name}_detection_${detection_runtimes[$d_index]}_planning_${planning}_target_speed_${target_speed}_Hz_${hz}_run_${run}
                    if [ ! -f "${PYLOT_HOME}/${file_base}.csv" ]; then
                        echo "[x] Running the experiment with detection deadline ${detection_runtimes[$d_index]} , planning deadline $planning , target speed $target_speed"
                        cd ${PYLOT_HOME}/scripts ; ./run_simulator.sh &
                        sleep 10
                        cd $PYLOT_HOME ; python3 pylot.py --flagfile=configs/scenarios/traffic_jam_static_deadlines.conf --target_speed=$target_speed --log_file_name=$file_base.log --csv_log_file_name=$file_base.csv --profile_file_name=$file_base.json --deadline_enforcement=static --detection_deadline=${detection_runtimes[$d_index]} --planning_deadline=$planning --obstacle_detection_model_paths=efficientdet/${detectors[$d_index]} --obstacle_detection_model_names=${detectors[$d_index]} --carla_camera_frequency=$hz --carla_imu_frequency=$hz --carla_lidar_frequency=$hz --carla_localization_frequency=$hz &
                        cd $ROOT_SCENARIO_RUNNER ; python3 scenario_runner.py --scenario $scenario --reloadWorld --timeout 600
                        echo "[x] Scenario runner finished. Killing Pylot..."
                        pkill --signal 9 -f -u $USER scenario_runner.py
                        # Kill the simulator
                        pkill --signal 9 -f -u $USER CarlaUE4
                        sleep 5
                    else
                        echo "$file_base exists"
                    fi
                done
            done
        done
        cd ${PYLOT_HOME}
        dir_name=results_${file_prefix_name}
        mkdir -p ${dir_name}
        mv ${file_prefix_name}* ${dir_name}/
    done
done
