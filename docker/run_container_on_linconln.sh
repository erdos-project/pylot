docker run --gpus all --rm --privileged --net host -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/bdd/dbw_ws:/home/erdos/workspace/dbw_ws -itd erdosproject/lincolnmkz:v3 /bin/bash
