FROM nvidia/cudagl:10.2-devel-ubuntu18.04
MAINTAINER Sukrit Kalra (sukrit.kalra@berkeley.edu)

# Set up a erdos user first.
RUN apt-get -y update && apt-get -y install sudo
ENV uid 1000
ENV gid 1000

RUN mkdir -p /home/erdos
RUN groupadd erdos -g ${gid} 
RUN useradd -r -u ${uid} -g erdos erdos
RUN echo "erdos ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/erdos
RUN chmod 0440 /etc/sudoers.d/erdos
RUN chown ${uid}:${gid} -R /home/erdos
RUN usermod --shell /bin/bash erdos


USER erdos
ENV HOME /home/erdos
ENV SHELL /bin/bash
WORKDIR /home/erdos
SHELL ["/bin/bash", "--login", "-c"]

RUN mkdir -p /home/erdos/workspace
RUN cd /home/erdos/workspace

RUN sudo apt-get -y update && sudo apt-get -y install --reinstall locales && sudo locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL en_US.UTF-8

# Install tzdata without prompt.
RUN sudo apt-get -y --fix-missing update
ENV DEBIAN_FRONTEND=noninteractive
RUN sudo DEBIAN_FRONTEND=noninteractive sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

# Get the erdos directory.
RUN sudo apt-get -y update
RUN sudo apt-get -y install git curl clang python3 python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools setuptools-rust numpy==1.19.5
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/erdos/.cargo/bin:${PATH}"
RUN rustup default nightly
RUN mkdir -p /home/erdos/workspace
RUN cd /home/erdos/workspace && git clone https://github.com/erdos-project/erdos.git && cd erdos && python3 python/setup.py install --user

# Set up Pylot.
RUN sudo apt-get install -y libcudnn7 cuda-toolkit-10-0 ssh libqt5core5a libeigen3-dev cmake qtbase5-dev libpng16-16 libtiff5 python3-tk
# Get the Pylot directory.
RUN cd /home/erdos/workspace && git clone https://github.com/erdos-project/pylot.git
WORKDIR /home/erdos/workspace/pylot/
ENV PYLOT_HOME /home/erdos/workspace/pylot/
# Install all the Python dependencies.
RUN cd /home/erdos/workspace/pylot/ && python3 -m pip install -e ./
# Get the Pylot models and code dependencies.
RUN cd /home/erdos/workspace/pylot/ && ./install.sh
ENV CARLA_HOME /home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1
# Clone scenario_runner.
RUN cd /home/erdos/workspace && git clone https://github.com/erdos-project/scenario_runner.git
# Install scenario_runner's dependencies.
RUN python3 -m pip install py-trees==0.8.3 networkx==2.2 Shapely==1.6.4 psutil==5.7.0 xmlschema==1.0.18 ephem==3.7.6.0 tabulate==0.8.7 six==1.14.0
# Clone leaderboard.
RUN cd /home/erdos/workspace && git clone https://github.com/erdos-project/leaderboard.git
RUN python3 -m pip install dictor requests

RUN echo "export PYTHONPATH=/home/erdos/workspace/pylot/dependencies/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/carla/agents/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1/PythonAPI/:/home/erdos/workspace/scenario_runner" >> ~/.bashrc
RUN echo "export PYLOT_HOME=/home/erdos/workspace/pylot/" >> ~/.bashrc
RUN echo "export CARLA_HOME=/home/erdos/workspace/pylot/dependencies/CARLA_0.9.10.1" >> ~/.bashrc
RUN echo "if [ -f ~/.bashrc ]; then . ~/.bashrc ; fi" >> ~/.bash_profile

# Set up ssh access to the container.
RUN cd ~/ && ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa <<<y 2>&1 >/dev/null
RUN sudo sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config

# WIP (for visualization in Foxglove)
# FOR NOW: install foxglove-studio and ROS manually in the pylot container

# # Download Foxglove
# RUN cd /home/erdos/workspace/pylot
# RUN curl -L https://github.com/foxglove/studio/releases/download/v0.10.2/foxglove-studio-0.10.2-linux-amd64.deb -o foxglove-studio-0.10.2-linux-amd64.deb
# RUN sudo apt install ./foxglove-studio-*.deb

# # Download ROS
# RUN cd /home/erdos/workspace/pylot
# # From, http://wiki.ros.org/melodic/Installation/Ubuntu:
# # Setup your computer to accept software from packages.ros.org: 
# RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# # Set up your keys
# RUN sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# # Make sure your Debian package index is up-to-date
# RUN sudo apt update
# # Desktop-Full Install
# RUN sudo apt install ros-melodic-desktop-full
# # Set up ROS environment variables
# RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# RUN source ~/.bashrc




