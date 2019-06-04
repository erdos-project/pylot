#!/bin/bash
# Build the Carla image
nvidia-docker build -t carla_ubuntu_16.04 -f Dockerfile_carla_Ubuntu16.04 .
# Build the ERDOS image
nvidia-docker build -t erdos_ubuntu_16.04 -f Dockerfile_erdos_carla_Ubuntu16.04 .
