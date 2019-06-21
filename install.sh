#!/bin/bash
# Assumes the script is called from PYLOT_HOME directory

sudo apt-get -y update
sudo apt-get install -y git wget python-pip
pip install --user gdown

sudo apt-get install -y lsb-release
os_version=$(lsb_release -sr)

###############################################################################
# Install ROS
###############################################################################
if [ $os_version == '16.04' ]
then
    # Install and setup ROS.
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
    sudo apt-get -y update
    sudo apt-get install -y --allow-unauthenticated ros-kinetic-ros-base
    echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
    source /opt/ros/kinetic/setup.bash
elif [ $os_version == '18.04' ]
then
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
    sudo apt-get -y update
    sudo apt-get install -y --allow-unauthenticated ros-melodic-ros-base
    echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
    source /opt/ros/melodic/setup.bash
else
    echo "Unsupported OS version"
    exit 1
fi
sudo apt-get install -y --allow-unauthenticated python-cv-bridge python-rosinstall

# Initialize rosdep
sudo rosdep init
rosdep update

###############################################################################
# Get models & code bases we depend on
###############################################################################
mkdir dependencies/models
cd dependencies/models

###### Download object detection models from TensorFlow zoo ######
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
tar -xvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
rm faster_rcnn_resnet101_coco_2018_01_28.tar.gz
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
rm ssd_mobilenet_v1_coco_2018_01_28.tar.gz
wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar -xvf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
rm ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

###### Download the traffic light model ######
wget https://www.dropbox.com/s/fgvvfsjuezbswy2/traffic_light_det_inference_graph.pb

###### Download the DRN segmentation cityscapes models ######
wget https://www.dropbox.com/s/i6v54gng0rao6ff/drn_d_22_cityscapes.pth

###### Download the DASiamRPN object tracker models ######
~/.local/bin/gdown https://doc-08-6g-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/vclpii8js65be25rf8v1dttqkpscs4l8/1555783200000/04094321888119883640/*/1G9GtKpF36-AwjyRXVLH_gHvrfVSCZMa7?e=download --output SiamRPNVOT.model
~/.local/bin/gdown https://doc-0s-6g-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/tomo4jo32befsdhi6qeaapdeep2v18np/1555783200000/04094321888119883640/*/18-LyMHVLhcx6qBWpUJEcPFoay1tSqURI?e=download --output SiamRPNBIG.model
~/.local/bin/gdown https://doc-0k-6g-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/dpfhmlqtdcbn0rfvqhbd0ofcg5aqphps/1555783200000/04094321888119883640/*/1_bIGtHYdAoTMS-hqOPE1j3KU-ON15cVV?e=download --output SiamRPNOTB.model
cd ../

###### Get the CRV Tracker model and dependencies ######
~/.local/bin/gdown http://drive.google.com/uc?id=0B1sg8Yyw1JCDOUNsYkpQTGdLYVU
mkdir -p conv_reg_vot/vgg_model
mv VGG_16_layers_py3.npz conv_reg_vot/vgg_model/
pip install --user matplotlib
sudo apt-get -y install python-tk

###### Download the DaSiamRPN code ######
git clone https://github.com/ICGog/DaSiamRPN.git
pip install --user opencv-python

###### Download the DRN segmentation code ######
git clone https://github.com/ICGog/drn.git

###### Download CenterNet ######
git clone https://github.com/ICGog/CenterNet.git

###### Download the Carla simulator ######
if [ "$1" != 'challenge' ]; then
    mkdir CARLA_0.8.4
    cd CARLA_0.8.4
    ~/.local/bin/gdown https://drive.google.com/uc?id=18OaDbQ2K9Dcs25d-nIxpw3GPRHhG1r_2
    tar xvf CARLA_0.8.4.tar.gz
    rm CARLA_0.8.4.tar.gz
    if [ "$1" == 'docker' ]; then
        rm -r CarlaUE4
    fi

    cd ../
    mkdir CARLA_0.9.5
    cd CARLA_0.9.5
    wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.5.tar.gz
    tar xvf CARLA_0.9.5.tar.gz
    rm CARLA_0.9.5.tar.gz
    if [ "$1" == 'docker' ]; then
        rm -r CarlaUE4; rm -r HDMaps
    fi
fi
