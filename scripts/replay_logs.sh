#!/bin/bash
# $1 Directory where the log files are

if [ -z "$PYLOT_HOME" ]; then
    echo "Please set \$PYLOT_HOME before sourcing this script"
    exit 1
fi

cd ${CARLA_HOME}/PythonAPI/examples

function replay_log {
    FILE_NAME=$1
    echo "Replaying ${FILE_NAME}"
    HERO_ID=`python3 show_recorder_file_info.py -f ${FILE_NAME} | grep -B 5 hero | head -n 1 | cut -d' ' -f3 | cut -d':' -f1`
    python3 start_replaying.py -f ${FILE_NAME} -c ${HERO_ID} -x 5
    read -p "Press any key when the replay completes... " -n1 -s
}

if [ -f $1 ]; then
    replay_log $1
else
    for FILE_NAME in `ls -d $1/*.log`; do
        replay_log $FILE_NAME
    done
fi
