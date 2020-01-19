#!/bin/bash
# Assumes the script is called from PYLOT_HOME directory

sudo apt-get -y update
sudo apt-get install -y git wget python3-pip
pip3 install gdown
# Install opencv separately because pip3 install doesn't install all libraries
# opencv requires.
sudo apt-get install -y python3-opencv

###############################################################################
# Get models & code bases we depend on
###############################################################################
mkdir -p dependencies/models
cd dependencies/models

###### Download object detection models from TensorFlow zoo ######
mkdir obstacle_detection ; cd obstacle_detection
mkdir faster-rcnn ; cd faster-rcnn
wget https://www.dropbox.com/s/s3pdfw34jd41mna/frozen_inference_graph.pb ; cd ../
mkdir ssd-mobilenet-fpn-640 ; cd ssd-mobilenet-fpn-640
wget https://www.dropbox.com/s/tcom898p9os6kzc/frozen_inference_graph.pb ; cd ../
mkdir ssdlite-mobilenet-v2 ; cd ssdlite-mobilenet-v2
wget https://www.dropbox.com/s/09dmldz7d7zfmrl/frozen_inference_graph.pb ; cd ../
cd ../
###### Download the traffic light model ######
mkdir -p traffic_light_detection/faster-rcnn ; cd traffic_light_detection/faster-rcnn
wget https://www.dropbox.com/s/63ig79xrkubigpu/frozen_inference_graph.pb
cd ../../

###### Download the DRN segmentation cityscapes models ######
mkdir -p segmentation/drn ; cd segmentation/drn
wget https://www.dropbox.com/s/i6v54gng0rao6ff/drn_d_22_cityscapes.pth
cd ../../

###### Download the DASiamRPN object tracker models ######
mkdir -p tracking/DASiamRPN ; cd tracking/DASiamRPN
# SiamRPNVOT.model
gdown https://drive.google.com/uc?id=1G9GtKpF36-AwjyRXVLH_gHvrfVSCZMa7
# SiamRPNBIG.model
gdown https://drive.google.com/uc?id=1_bIGtHYdAoTMS-hqOPE1j3KU-ON15cVV
# SiamRPNOTB.model
gdown https://drive.google.com/uc?id=18-LyMHVLhcx6qBWpUJEcPFoay1tSqURI
cd ../../../

###### Get DeepSORT and SORT tracker code bases
git clone https://github.com/ICGog/nanonets_object_tracking.git
git clone https://github.com/abewley/sort.git

###### Download the DaSiamRPN code ######
git clone https://github.com/ICGog/DaSiamRPN.git

###### Download the DRN segmentation code ######
git clone https://github.com/ICGog/drn.git

###### Download the Carla simulator ######
if [ "$1" != 'challenge' ]; then
    mkdir CARLA_0.9.6
    cd CARLA_0.9.6
    wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
    tar xvf CARLA_0.9.6.tar.gz
    rm CARLA_0.9.6.tar.gz
    if [ "$1" == 'docker' ]; then
        rm -r CarlaUE4; rm -r HDMaps
    fi
fi
