#!/bin/bash
# Build the Carla image
nvidia-docker build -t carla_ubuntu_16.04 -f Dockerfile_carla_Ubuntu16.04 .
# Build the Pylot image
nvidia-docker build -t pylot_ubuntu_16.04 -f Dockerfile_pylot_Ubuntu16.04 .
