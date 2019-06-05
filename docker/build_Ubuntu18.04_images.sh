#!/bin/bash
# Build the Carla image
nvidia-docker build -t erdosproject/carla -f Dockerfile_carla_Ubuntu18.04 .
# Build the Pylot image
nvidia-docker build -t erdosproject/pylot -f Dockerfile_pylot_Ubuntu18.04 .
