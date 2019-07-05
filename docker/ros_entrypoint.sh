#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/kinetic/setup.bash"
exec "$@"
