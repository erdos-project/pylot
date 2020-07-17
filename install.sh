#!/bin/bash
# Assumes the script is called from PYLOT_HOME directory

export PYLOT_HOME=$(pwd)

sudo apt-get -y update
sudo apt-get install -y git wget cmake python3-pip unzip clang libpng-dev libgeos-dev
pip3 install gdown --user
# Install opencv separately because pip3 install doesn't install all libraries
# opencv requires.
sudo apt-get install -y python3-opencv

###############################################################################
# Get models & code bases we depend on
###############################################################################

###### Build the FrenetOptimalTrajectory Planner ######
cd $PYLOT_HOME/dependencies/
git clone https://github.com/erdos-project/frenet_optimal_trajectory_planner.git
cd frenet_optimal_trajectory_planner/
bash build.sh

###### Build the RRT* Planner ######
cd $PYLOT_HOME/dependencies/
git clone https://github.com/erdos-project/rrt_star_planner.git
cd rrt_star_planner/
bash build.sh

###### Build the Hybrid A* Planner ######
cd $PYLOT_HOME/dependencies/
git clone https://github.com/erdos-project/hybrid_astar_planner.git
cd hybrid_astar_planner/
bash build.sh

###### Clone the Prediction Repository #####
cd $PYLOT_HOME
cd pylot/prediction/
git clone https://github.com/erdos-project/prediction.git

cd $PYLOT_HOME/dependencies/
mkdir -p dependencies/models

###### Download CARLA-trained object detection models ######
cd $PYLOT_HOME/dependencies/models
mkdir obstacle_detection ; cd obstacle_detection
wget --max-redirect=20 -O download.zip https://www.dropbox.com/sh/v1mex7ykdrak3av/AADFsd8mwFcQnzJsIAQR6oEja
unzip download.zip ; rm download.zip

###### Download the traffic light model ######
cd $PYLOT_HOME/dependencies/models
mkdir -p traffic_light_detection/faster-rcnn ; cd traffic_light_detection/faster-rcnn
wget https://www.dropbox.com/s/63ig79xrkubigpu/frozen_inference_graph.pb

###### Download the Lanenet lane detection model ######
cd $PYLOT_HOME/dependencies/models
mkdir -p lane_detection/lanenet ; cd lane_detection/lanenet
wget --max-redirect=20 -O download.zip https://www.dropbox.com/sh/hdqpoxr0yo7mk05/AABx5b7QdEmkxWX2VDa19xKCa
unzip download.zip ; rm download.zip

###### Download the DRN segmentation cityscapes models ######
cd $PYLOT_HOME/dependencies/models
mkdir -p segmentation/drn ; cd segmentation/drn
wget https://www.dropbox.com/s/i6v54gng0rao6ff/drn_d_22_cityscapes.pth

###### Download the DASiamRPN object tracker models ######
cd $PYLOT_HOME/dependencies/models
mkdir -p tracking/DASiamRPN ; cd tracking/DASiamRPN
# SiamRPNVOT.model
~/.local/bin/gdown https://drive.google.com/uc?id=1G9GtKpF36-AwjyRXVLH_gHvrfVSCZMa7
# SiamRPNBIG.model
~/.local/bin/gdown https://drive.google.com/uc?id=1_bIGtHYdAoTMS-hqOPE1j3KU-ON15cVV
# SiamRPNOTB.model
~/.local/bin/gdown https://drive.google.com/uc?id=18-LyMHVLhcx6qBWpUJEcPFoay1tSqURI

##### Download AnyNet depth estimation models #####
cd $PYLOT_HOME/dependencies/models
mkdir -p depth_estimation/AnyNet ; cd depth_estimation/AnyNet
~/.local/bin/gdown https://drive.google.com/uc?id=18Vi68rQO-vcBn3882vkumIWtGggZQDoU
unzip checkpoint.zip ; rm checkpoint.zip

###### Download DeepSort models ######
# Download the real-world model.
cd $PYLOT_HOME/dependencies/models
mkdir -p tracking/deep-sort ; cd tracking/deep-sort
~/.local/bin/gdown https://drive.google.com/uc?id=1bB66hP9voDXuoBoaCcKYY7a8IYzMMs4P
cd ../
# Download the CARLA model.
mkdir deep-sort-carla ; cd deep-sort-carla
wget https://www.dropbox.com/s/j007vs40f9tlcui/feature_extractor

###### Download the R2P2 prediction model ######
cd $PYLOT_HOME/dependencies/models
mkdir -p prediction/r2p2 ; cd prediction/r2p2
wget https://www.dropbox.com/s/vyrh8tysre1bmwi/r2p2-model.pt

#################### Download the code bases ####################
cd $PYLOT_HOME/dependencies/
###### Get the EfficientDet code ######
git clone https://github.com/ICGog/automl.git

###### Get DeepSORT and SORT tracker code bases ######
git clone https://github.com/ICGog/nanonets_object_tracking.git
git clone https://github.com/ICGog/sort.git

###### Download the DaSiamRPN code ######
git clone https://github.com/ICGog/DaSiamRPN.git

##### Download the Lanenet code #####
git clone https://github.com/ICGog/lanenet-lane-detection.git
mv lanenet-lane-detection lanenet
#pip3 install glog loguru

###### Download the DRN segmentation code ######
git clone https://github.com/ICGog/drn.git

###### Download AnyNet depth estimation code #####
git clone https://github.com/mileyan/AnyNet.git
cd AnyNet/models/spn_t1/ ; python3 setup.py clean ; python3 setup.py build

cd $PYLOT_HOME/dependencies/
###### Download the Carla simulator ######
if [ "$1" != 'challenge' ] && [ ! -d "CARLA_0.9.9" ]; then
    mkdir CARLA_0.9.9
    cd CARLA_0.9.9
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.9.2.tar.gz
    tar -xvf CARLA_0.9.9.2.tar.gz
    rm CARLA_0.9.9.2.tar.gz
    if [ "$1" == 'docker' ]; then
        rm -r CarlaUE4; rm -r HDMaps
    fi
fi
