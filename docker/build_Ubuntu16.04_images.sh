#!/bin/bash
# Build the Carla image
nvidia-docker build -t erdosproject/carla -f Dockerfile_carla_Ubuntu16.04 .
# Build the Pylot image
nvidia-docker build -t erdosproject/pylot -f Dockerfile_pylot_Ubuntu16.04 .
