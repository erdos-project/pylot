#!/bin/bash
# Build the Carla image
nvidia-docker build -t carla_ubuntu_18.04 -f Dockerfile_carla_Ubuntu18.04 .
# Build the ERDOS image
nvidia-docker build -t erdos_ubuntu_18.04 -f Dockerfile_erdos_carla_Ubuntu18.04 .
