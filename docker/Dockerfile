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

# Install tzdata without prompt.
RUN sudo apt-get -y update
ENV DEBIAN_FRONTEND=noninteractive
RUN sudo DEBIAN_FRONTEND=noninteractive sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

# Get the erdos directory.
RUN sudo apt-get -y update
RUN sudo apt-get -y install git curl clang python3 python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install setuptools setuptools-rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/erdos/.cargo/bin:${PATH}"
RUN rustup default nightly
RUN mkdir -p /home/erdos/workspace
RUN cd /home/erdos/workspace && git clone https://github.com/erdos-project/erdos.git && cd erdos && python3 python/setup.py install --user

# Set up Pylot.
RUN sudo apt-get install -y python3-pip libcudnn7 cuda-toolkit-10-0 ssh libqt5core5a libeigen3-dev cmake qtbase5-dev libpng16-16 libtiff5
# Get the Pylot directory.
RUN cd /home/erdos/workspace && git clone https://github.com/erdos-project/pylot.git
WORKDIR /home/erdos/workspace/pylot/
ENV PYLOT_HOME /home/erdos/workspace/pylot/
# Get the Pylot models and code dependencies.
RUN cd /home/erdos/workspace/pylot/ && ./install.sh
# Install all the Python dependencies.
RUN cd /home/erdos/workspace/pylot/ && pip3 install -e ./
ENV CARLA_HOME /home/erdos/workspace/pylot/dependencies/CARLA_0.9.9
# Clone scenario_runner.
RUN cd /home/erdos/workspace &&  git clone https://github.com/erdos-project/scenario_runner.git
# Install scenario_runner's dependencies.
RUN pip3 install py-trees==0.8.3 networkx==2.2 Shapely==1.6.4 psutil xmlschema==1.0.18 ephem tabulate

RUN echo "export PYTHONPATH=/home/erdos/workspace/pylot/dependencies/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.9/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.9/PythonAPI/carla/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.9/PythonAPI/carla/agents/:/home/erdos/workspace/pylot/dependencies/CARLA_0.9.9/PythonAPI/:/home/erdos/workspace/scenario_runner" >> ~/.bashrc
RUN echo "export PYLOT_HOME=/home/erdos/workspace/pylot/" >> ~/.bashrc
RUN echo "export CARLA_HOME=/home/erdos/workspace/pylot/dependencies/CARLA_0.9.9" >> ~/.bashrc
RUN echo "if [ -f ~/.bashrc ]; then . ~/.bashrc ; fi" >> ~/.bash_profile

# Set up ssh access to the container.
RUN cd ~/ && ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa <<<y 2>&1 >/dev/null
RUN sudo sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
