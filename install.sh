#!/bin/bash

if [ -z $PYLOT_HOME ] ; then
    PYLOT_HOME=$(pwd)
    echo "WARNING: \$PYLOT_HOME is not set; Setting it to ${PYLOT_HOME}"
else
    echo "INFO: \$PYLOT_HOME is set to ${PYLOT_HOME}"
fi

sudo apt-get -y update
sudo apt-get install -y git wget cmake python3-pip unzip clang libpng-dev libgeos-dev
# Install opencv separately because pip3 install doesn't install all libraries
# opencv requires.
sudo apt-get install -y python3-opencv
python3 -m pip install user gdown
# Install Pygame if available.
PYGAME_PKG=`apt-cache search python3-pygame`
if [ -n "$PYGAME_PKG" ] ; then
    sudo apt-get install python3-pygame
fi

###############################################################################
# Get models & code bases we depend on
###############################################################################
cd $PYLOT_HOME/dependencies/

###### Download the model weights ######
echo "[x] Downloading all model weights..."
cd $PYLOT_HOME/dependencies/
~/.local/bin/gdown https://drive.google.com/uc?id=1rQKFDxGDFi3rBLsMrJzb7oGZvvtwgyiL
unzip models.zip ; rm models.zip

#################### Download the code bases ####################
echo "[x] Compiling the planners..."
###### Build the FrenetOptimalTrajectory Planner ######
echo "[x] Compiling the Frenet Optimal Trajectory planner..."
cd $PYLOT_HOME/dependencies/
git clone https://github.com/erdos-project/frenet_optimal_trajectory_planner.git
cd frenet_optimal_trajectory_planner/
bash build.sh

###### Build the RRT* Planner ######
echo "[x] Compiling the RRT* planner..."
cd $PYLOT_HOME/dependencies/
git clone https://github.com/erdos-project/rrt_star_planner.git
cd rrt_star_planner/
bash build.sh

###### Build the Hybrid A* Planner ######
echo "[x] Compiling the Hybrid A* planner..."
cd $PYLOT_HOME/dependencies/
git clone https://github.com/erdos-project/hybrid_astar_planner.git
cd hybrid_astar_planner/
bash build.sh

###### Clone the Prediction Repository #####
echo "[x] Cloning the prediction code..."
cd $PYLOT_HOME/pylot/prediction/
git clone https://github.com/erdos-project/prediction.git

###### Get DeepSORT and SORT tracker code bases ######
echo "[x] Cloning the object tracking code..."
cd $PYLOT_HOME/dependencies/
git clone https://github.com/ICGog/nanonets_object_tracking.git
sudo apt-get install python3-tk
git clone https://github.com/ICGog/sort.git
###### Download the DaSiamRPN code ######
cd $PYLOT_HOME/dependencies/
git clone https://github.com/ICGog/DaSiamRPN.git

###### Install CenterTrack ######
echo "[x] Installing the CenterTrack object tracking code..."
cd $PYLOT_HOME/dependencies/
git clone https://github.com/ICGog/CenterTrack
cd CenterTrack/src/lib/model/networks/
git clone https://github.com/CharlesShang/DCNv2/
cd DCNv2
sudo apt-get install llvm-9
export LLVM_CONFIG=/usr/bin/llvm-config-9
python3 setup.py build develop user

###### Install QDTrack ######
cd $PYLOT_HOME/dependencies/
git clone https://github.com/mageofboy/qdtrack.git
cd $PYLOT_HOME/dependencies/qdtrack
python3 -m pip install mmcv==1.3.10 mmdet==2.14.0
python3 -m pip install -e ./

##### Download the Lanenet code #####
echo "[x] Cloning the lanenet lane detection code..."
cd $PYLOT_HOME/dependencies/
git clone https://github.com/ICGog/lanenet-lane-detection.git
mv lanenet-lane-detection lanenet

###### Download the DRN segmentation code ######
echo "[x] Cloning the DRN segmentation code..."
cd $PYLOT_HOME/dependencies/
git clone https://github.com/ICGog/drn.git

###### Download AnyNet depth estimation code #####
echo "[x] Cloning the AnyNet depth estimation code..."
cd $PYLOT_HOME/dependencies/
git clone https://github.com/mileyan/AnyNet.git
cd AnyNet/models/spn_t1/ ; python3 setup.py clean ; python3 setup.py build

###### Download the Carla simulator ######
echo "[x] Downloading the CARLA 0.9.10.1 simulator..."
cd $PYLOT_HOME/dependencies/
if [ "$1" != 'challenge' ] && [ ! -d "CARLA_0.9.10.1" ]; then
    mkdir CARLA_0.9.10.1
    cd CARLA_0.9.10.1
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
    tar -xvf CARLA_0.9.10.1.tar.gz
    rm CARLA_0.9.10.1.tar.gz
fi
