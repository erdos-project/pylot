#!/bin/bash
# Build the Pylot image
nvidia-docker build -t erdosproject/pylot -f Dockerfile .
