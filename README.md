Pylot is an autonomous vehicle platform for developing and test autonomous
vehicle components (e.g., perception, prediction, planning) using the
Carla simulator.

# Setup instructions
Please ensure you have `nvidia-docker` on your machine before you start installing Pylot.
In case you do not have `nvidia-docker` please run ```./scripts/install-nvidia-docker.sh```

To install Pylot, use the Dockerfiles we provide:

```console
$ cd docker
$ ./build_Ubuntu18.04_images.sh
```

The script creates two docker images: one that contains the Carla simulator and
another one that contains ERDOS and Pylot.

Next, create a Docker network, a Carla container, and an ERDOS container:

```console
docker network create carla-net
nvidia-docker run -itd --name carla_v1 --net carla-net carla_ubuntu_18.04 /bin/bash
nvidia-docker run -itd --name erdos_v1 --net carla-net erdos_ubuntu_18.04 /bin/bash
```

Following, start the simulator in the Carla container:

```console
nvidia-docker exec -i -t carla_v1 /bin/bash
SDL_VIDEODRIVER=offscreen ${CARLA_ROOT}/CarlaUE4.sh -windowed -ResX=800 -ResY=600 -carla-server -benchmark -fps=10
```

Finally, start Pylot in the ERDOS container:

```console
nvidia-docker exec -i -t erdos_v1 /bin/bash
cd examples/pylot/scripts; ./install_centernet.sh
cd ../
python pylot.py --flagfile=configs/eval/segmentation_drn.conf --carla_host=carla_v1
```

## Manual installation instructions
Alternatively, you can install Pylot on your base system by executing the
following steps:

```console
$ sudo apt-get install python-rosinstall python-cv-bridge python-pip wget
$ pip install -r requirements.txt
$ pip install tensorflow-gpu
$ cd scripts
$ ./get_dependencies.sh
$ ./install_centernet.sh
```

Next, setup the paths:
```console
$ export CARLA_ROOT=$ERDOS_HOME/examples/pylot/dependencies/CARLA_0.8.4/
$ source ./set_pythonpath.sh
```

Following, execute the simulator:
```console
$ ./scripts/run_simulator.sh
```

Finally, execute Pylot in a different terminal:
```console
python pylot.py --flagfile=configs/eval/segmentation_drn.conf
```


# Pylot components

Pylot comprises of several components: obstacle detection, traffic light
detection, segmentation, fusion, and driving agents. Each component is
implemented using one or more ERDOS operatos and can be executed in
isolation or with the entire Pylot application.

### Obstacle detection
Pylot supports four object detection models: `ssd_mobilenet_v1`, `ssd_resnet50_v1`,
`frcnn_resnet101`, and `Centernet`. The following command runs a detector in isolation:

```console
python pylot.py --flagfile=configs=configs/detection.conf
```

In case you want to evaluate the detector (i.e., compute mAP), you can run:
```console
python pylot.py --flagfile=configs=configs/object_accuracy.conf
```

If the detector cannot run at your desired frequency or if you want to ensure that
objects are reliably detected across frames, you can use a mix of detector plus
tracker by running:

```console
python pylot.py --flagfile=configs=configs/tracking.conf
```

### Traffic light detection
Pylot has uses a separate component for traffic light detection and classification.
The following command runst the component in isolation:

```console
python pylot.py --flagfile=configs=configs/traffic_light.conf
```

### Segmentation
In order to run Pylot's segmentation component in isolation execute the
following command:

```console
python pylot.py --flagfile=configs=configs/segmentation.conf
# Evaluates the segmented frames
python pylot.py --flagfile=configs=configs/segmentation_mIoU_eval_drn.conf
```
### Sensor fusion
In order to run Pylot's naive fusion in isolation execute:

```console
python pylot.py --flagfile=configs=configs/fusion.conf
```

### Driving policies
Pylot supports two driving policies:
 1. Ground agent policy: uses perfect sensor data and information to drive
 without crashing. This policy is useful to test components in long running
 experiments in which you do not want the car the crash.
 2. ERDOS agent policy: uses a mix of data from our components (e.g., detected
 obstacles, traffic lights) and data from Carla (e.g., traffic lanes). This
 policy is currently unreliable (i.e., crashes, runs red traffic lights), but
 we will improve it to be more reliable and we will adapt it to only use data
 from Pylot components.

You can run all the components, together with one of the two policies by
executing:

```console
# Runs all components and the ERDOS policy
python pylot.py --flagfile=configs=configs/erdos_agent_e2e.conf
# Runs all components and the ground policy
python pylot.py --flagfile=configs=configs/ground_agent_e2e.conf
```
