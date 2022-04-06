[![Build Status](https://github.com/erdos-project/pylot/workflows/CI/badge.svg)](https://github.com/erdos-project/pylot/actions)
[![Documentation Status](https://readthedocs.org/projects/pylot/badge/?version=latest)](https://pylot.readthedocs.io/en/latest/?badge=latest)

Pylot is an autonomous vehicle platform for developing and testing autonomous
vehicle components (e.g., perception, prediction, planning) on the
CARLA simulator and real-world cars.

* [**Setup instructions**](#setup-instructions)
* [**Documentation**](https://pylot.readthedocs.io/en/latest/)
* [**Pylot components**](#pylot-components)
* [**Data collection**](#data-collection)
* [**Build Docker image**](#build-your-own-docker-image)
* [**CARLA autonomous driving challenge**](#carla-autonomous-driving-challenge)
* [**Getting involved**](#getting-involved)

# Setup instructions

## Deploy using Docker

The easiest way to get Pylot running is to use our Docker image. Please ensure
you have `nvidia-docker` on your machine before you start installing Pylot.
In case you do not have `nvidia-docker` please
run ```./scripts/install-nvidia-docker.sh```

We provide a Docker image with both Pylot and CARLA already setup.

```console
docker pull erdosproject/pylot
nvidia-docker run -itd --name pylot -p 20022:22 erdosproject/pylot /bin/bash
```

Following, start the simulator in the container:

```console
nvidia-docker exec -i -t pylot /home/erdos/workspace/pylot/scripts/run_simulator.sh
```

Finally, start Pylot in the container:

```console
nvidia-docker exec -i -t pylot /bin/bash
cd ~/workspace/pylot/
python3 pylot.py --flagfile=configs/detection.conf
```

## Visualizing components
In case you desire to visualize outputs of different components (e.g., bounding boxes),
you have to forward X from the container. First, add your public ssh key to the
`~/.ssh/authorized_keys` in the container:

```console
nvidia-docker cp ~/.ssh/id_rsa.pub pylot:/home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo chown erdos /home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo service ssh start
```

Finally, ssh into the container with X forwarding:
```console
ssh -p 20022 -X erdos@localhost
cd /home/erdos/workspace/pylot/
python3 pylot.py --flagfile=configs/detection.conf --visualize_detected_obstacles
```

If everything worked ok, you should be able to see a visualization like
the one below:

![Pylot obstacle detection](/doc/source/images/pylot-obstacle-detection.png)

## Manual installation instructions
Alternatively, you can install Pylot on your base system by executing the
following steps:

```console
./install.sh
pip install -e ./
```

Next, start the simulator:
```console
export CARLA_HOME=$PYLOT_HOME/dependencies/CARLA_0.9.10.1/
./scripts/run_simulator.sh
```

In a different terminal, setup the paths:
```console
export CARLA_HOME=$PYLOT_HOME/dependencies/CARLA_0.9.10.1/
cd $PYLOT_HOME/scripts/
source ./set_pythonpath.sh
```

Finally, run Pylot:
```console
cd  $PYLOT_HOME/
python3 pylot.py --flagfile=configs/detection.conf
```

# Pylot components

Pylot comprises of several components: obstacle detection, traffic light
detection, lane detection, obstacle tracking, localization, segmentation,
fusion, prediction, planners, and control.
Each component is implemented using one or more ERDOS operators and can be
executed in isolation or with the entire Pylot application. Please read
the [**Documentation**](https://pylot.readthedocs.io/en/latest/) for
a more in depth description.

![Pylot pipeline](/doc/source/images/pylot.png)

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

### Lane detection

```console
python3 pylot.py --flagfile=configs/lane_detection.conf
```

### Obstacle tracking

```console
python3 pylot.py --flagfile=configs/tracking.conf
```

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

1. `waypoint`: a simple planner that follows predefined waypoints. These
waypoints can either be either pre-specified or computed using the A-star
planner part of the CARLA simulator map. The planner ensures that the
ego-vehicle respects traffic lights, stops whenever there are obstacles in its
path, but does not implement obstacle avoidance.
2. `frenet_optimal_trajectory`: a Frenet Optimal Trajectory planner.
3. `rrt_star`: a Rapidly-explory Random Tree planner.
4. `hybrid_astar`: a Hybrid A* planner.

```console
# To run the Frenet Optimal Trajectory planner.
python3 pylot.py --flagfile=configs/frenet_optimal_trajectory_planner.conf

# To run the RRT* planner.
python3 pylot.py --flagfile=configs/rrt_star_planner.conf

# To run the Hybrid A* planner.
python3 pylot.py --flagfile=configs/hybrid_astar_planner.conf
```

### Control
Pylot supports three controllers, which can be specified using the
```control``` flag:
1. `pid`: follows the waypoints computed by the planning component using a PID
controller.
2. `mpc`: uses model predictive control for speed and waypoint following.
3. `simulator_auto_pilot`: uses the simulator auto pilot to drive on predefined routes.
This controller drives independent of the output of the other components.

You can run all the components, together with one of the two policies by
executing:

```console
# Runs all components using the algorithms we implemented and the models we trained:
python3 pylot.py --flagfile=configs/e2e.conf
# Runs the MPC
python3 pylot.py --flagfile=configs/mpc.conf
# Runs the simulator auto pilot.
python3 pylot.py --control=simulator_auto_pilot
```

### Debug logs
In case you want to debug the application, you can active additional logging
by passing: `--log_file_name=pylot.log --v=1` to your command.

# Data collection

Pylot also provides a script for collecting CARLA data such as: RGB images,
segmented images, obstacle 2D bounding boxes, depth frames, point clouds,
traffic lights, obstacle trajectories, and data in Chauffeur format.

Run ```python3 data_gatherer.py --help``` to see what data you can collect.
Alternatively, you can inspect this
[configuration](https://github.com/erdos-project/pylot/blob/master/configs/data_gatherer.conf)
for an example of a data collection setup.

# Build your own Docker image

In case you want to build your own images from the latest code, you can execute:

```console
cd docker
./build_images.sh
```

The script creates two Docker images: one that contains the CARLA simulator and
another one that contains ERDOS and Pylot.

# CARLA autonomous driving challenge

Pylot can also be used as a baseline for executing on the CARLA
[**Leaderboard**](https://leaderboard.carla.org/) routes. We provide an  agent
that offers reference implementations for perception (i.e., detection,
tracking), localization (Extended Kalman filter), prediction, planning
(e.g., waypoint follower, Frenet optimal trajectory, RRT*, Hybrid A*),
and control.

To test this agent you can pull our image which has all the necessary software
already installed.

```console
docker pull erdosproject/pylot-carla-challenge
nvidia-docker run -itd --name pylot-challenge -p 20022:22 erdosproject/pylot-carla-challenge /bin/bash
```

Alternatively, you can manually install the dependencies on your machine by
following the instructions provided below:

```console
mkdir challenge
export CHALLENGE_ROOT=`pwd`
# Clone the challenge leaderboard repository.
git clone -b stable --single-branch https://github.com/carla-simulator/leaderboard.git
export LEADERBOARD_ROOT=${CHALLENGE_ROOT}/leaderboard/
cd ${LEADERBOARD_ROOT} ; pip3 install -r requirements.txt ; cd ${CHALLENGE_ROOT}
# Clone the CARLA scenario runner repository. This is used by the leaderboard.
git clone -b leaderboard --single-branch https://github.com/carla-simulator/scenario_runner.git
export SCENARIO_RUNNER_ROOT=${CHALLENGE_ROOT}/scenario_runner/
cd ${SCENARIO_RUNNER_ROOT} ; pip3 install -r requirements.txt ; cd ${CHALLENGE_ROOT}
# Checkout the CARLA challenge branch.
cd ${PYLOT_HOME} ; git checkout -b challenge origin/challenge
export CARLA_ROOT=Path to CARLA 0.9.10.1.
cd ${CHALLENGE_ROOT}
export TEAM_CODE_ROOT=${PYLOT_HOME} ; ${LEADERBOARD_ROOT}/scripts/make_docker.sh
```

## Notes on the Pylot CARLA challenge agent
Similar to regular Pylot, the [Challenge agent](https://github.com/erdos-project/pylot/blob/master/pylot/simulation/challenge/ERDOSAgent.py)
not only connects different reference implementation, but also provides the
option of testing them in different configurations (e.g., test prediction,
planning and control with perfect perception). This can be done by changing the
flags in the [challenge configuration](https://github.com/erdos-project/pylot/blob/master/pylot/simulation/challenge/challenge.conf)
according to the specification from the Pylot documentation.

# More Information

To read more about the ideas behind Pylot, refer to our paper,
*Pylot: A Modular Platform for Exploring Latency-Accuracy Tradeoffs in Autonomous Vehicles* ([IEEE](https://ieeexplore.ieee.org/document/9561747/)) ([arXiv](https://arxiv.org/abs/2104.07830)).
If you find Pylot useful to your work, please cite our paper as follows:
```bibtex
@inproceedings{gog2021pylot,
  title={Pylot: A modular platform for exploring latency-accuracy tradeoffs in autonomous vehicles},
  author={Gog, Ionel and Kalra, Sukrit and Schafhalter, Peter and Wright, Matthew A and Gonzalez, Joseph E and Stoica, Ion},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8806--8813},
  year={2021},
  organization={IEEE}
}
```

# Getting Involved
* [Community on Slack](https://forms.gle/KXwSrjM6ZqRi2MT18): Join our community
on Slack for discussions about development, questions about usage, and feature
requests.
* [Github Issues](https://github.com/erdos-project/pylot/issues): For reporting
bugs.
