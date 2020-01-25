Pylot is an autonomous vehicle platform for developing and testing autonomous
vehicle components (e.g., perception, prediction, planning) using the
Carla simulator.

[![Build Status](https://travis-ci.org/erdos-project/pylot.svg)](https://travis-ci.org/erdos-project/pylot)
[![Documentation Status](https://readthedocs.org/projects/pylot/badge/?version=latest)](https://pylot.readthedocs.io/en/latest/?badge=latest)

# Setup instructions

## Deploy using Docker

The easiest way to get Pylot running is to use our Docker images. Please ensure
you have `nvidia-docker` on your machine before you start installing Pylot.
In case you do not have `nvidia-docker` please
run ```./scripts/install-nvidia-docker.sh```

We provide a Docker image to run the Carla simulator in, and a Docker image with
Pylot and ERDOS already setup.

```console
docker pull erdosproject/pylot
docker pull erdosproject/carla
```

Next, create a Docker network, a Carla container, and a Pylot container:

```console
docker network create carla-net
nvidia-docker run -itd --name carla_v1 --net carla-net erdosproject/carla /bin/bash
nvidia-docker run -itd --name pylot_v1 -p 20022:22 --net carla-net erdosproject/pylot /bin/bash
```

Following, start the simulator in the Carla container:

```console
nvidia-docker exec -i -t carla_v1 /bin/bash
SDL_VIDEODRIVER=offscreen /home/erdos/workspace/CARLA_0.9.6/CarlaUE4.sh -opengl -windowed -ResX=800 -ResY=600 -carla-server -benchmark -fps=10 -quality-level=Epic
```

Finally, start Pylot in the container:

```console
nvidia-docker exec -i -t pylot_v1 /bin/bash
cd workspace/pylot/
python3 pylot.py --flagfile=configs/detection.conf --carla_host=carla_v1
```

In case you desire to visualize outputs of different components (e.g., bounding boxes),
you have to forward X from the pylot container. First, in the container execute the
following steps:
```console
ssh-keygen
# Copy yout public ssh key into .ssh/authorized_keys
sudo sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
sudo service ssh restart
exit
```

Finally, ssh into the container with X forwarding:
```console
ssh -p 20022 -X erdos@localhost
/bin/bash
cd /home/erdos/workspace/pylot/
python3 pylot.py --flagfile=configs/detection.conf --carla_host=carla_v1
```

## Manual installation instructions
Alternatively, you can install Pylot on your base system by executing the
following steps:

```console
./install.sh
pip install -e ./
```

Next, start the simulator:
```console
export CARLA_HOME=$PYLOT_HOME/dependencies/CARLA_0.9.6/
./scripts/run_simulator.sh
```

In a different terminal, setup the paths:
```console
export CARLA_HOME=$PYLOT_HOME/dependencies/CARLA_0.9.6/
cd $PYLOT_HOME/scripts/
source ./set_pythonpath.sh
```

Finally, execute Pylot in a different terminal:
```console
cd  $PYLOT_HOME/
python3 pylot.py --flagfile=configs/detection.conf
```

# Pylot components

Pylot comprises of several components: obstacle detection, traffic light
detection, segmentation, fusion, prediction, planners, and driving agents.
Each component is implemented using one or more ERDOS operators and can be
executed in isolation or with the entire Pylot application.

Run the following command to see a demo of all the components, and the Pylot
driving policy:

```console
python3 pylot.py --flagfile=configs/demo.conf
```

The demo will execute: obstacle detection, traffic light detection,
segmentation, prediction, planning, and the driving policy.

***
You can also run components in isolation:

### Obstacle detection
Pylot supports three object detection models: `frcnn_resnet101`,
`ssd-mobilenet-fpn-640` and `ssdlite-mobilenet-v2`. The following command runs
a detector in isolation:

```console
python3 pylot.py --flagfile=configs/detection.conf
```

In case you want to evaluate the detector (i.e., compute mAP), you can run:
```console
python3 pylot.py --flagfile=configs/detection.conf --evaluate_obstacle_detection
```

In case you are not satisfied with the accuracy of our obstacle detector, you
can run a perfect version of it:

```console
python3 pylot.py --flagfile=configs/perfect_detection.conf
```

If the detector does not run at your desired frequency, or if you want to track
obstacles across frames, you can use a mix of detector plus tracker by running:

```console
python3 pylot.py --flagfile=configs/tracking.conf
```

### Traffic light detection
Pylot has uses a separate component for traffic light detection and
classification. The following command runs the component in isolation:

```console
python3 pylot.py --flagfile=configs/traffic_light.conf
```

In case you require higher accuracy, you can run perfect traffic light detection
by passing the ```--perfect_traffic_light_detection``` flag.

### Segmentation
In order to run Pylot's segmentation component in isolation execute the
following command:

```console
python3 pylot.py --flagfile=configs/segmentation.conf
```

Similarly, pass ```--perfect_segmentation``` if you desire ideal pixel semantic
segmentation.

### Prediction
Pylot offers a simple linear prediction component:

```console
python3 pylot.py --flagfile=configs/prediction.conf
```

### Planning
The planning component provides two planning options, which can be specified
using the ```--planning_type``` flag:

1. `waypoint`: a A-star planner that uses the Carla simulator map.
2. `rrt_star`: a rapidly-explory random tree planner.


### Driving policies
Pylot supports three driving policies, which can be specified using the
```control_agent``` flag:
1. `pylot`: an agent that stops when obstacles are detected. It can use
 outputs from operators that use trained models or operators that output
 perfect obstacles using CARLA data.
2. `mpc`: uses model predictive control for speed and steering.
3. `carla_auto_pilot`: uses the carla auto pilot to drive on predefined routes.

You can run all the components, together with one of the two policies by
executing:

```console
# Runs all components and the Pylot policy
python3 pylot.py --flagfile=configs/pylot_agent_e2e.conf
# Runs the MPC policy
python3 pylot.py --flagfile=configs/mpc_agent.conf
# Runs the carla policy
python3 pylot.py --control_agent=carla_auto_pilot
```

### Debug logs
In case you want to debug the application, you can active additional logging
by passing: `--log_file_name=pylot.log --v=1` to your command.

# Data collection

Pylot also provides a script for collecting Carla data such as: RGB images,
segmented images, obstacle 2D bounding boxes, depth frames, point clouds,
traffic lights, obstacle trajectories, and data in Chauffeur format.

Run ```python3 data_gatherer.py --help``` to see what data you can collect.
Alternatively, you can inspect
[a link](https://github.com/erdos-project/pylot/blob/master/configs/data_gatherer.conf)
for an example of a data collection setup.

# Build your own Docker images

In case you want to build your own images from the latest code, you can execute:

```console
cd docker
./build_Ubuntu18.04_images.sh
```

The script creates two Docker images: one that contains the Carla simulator and
another one that contains ERDOS and Pylot.
