#!/bin/bash
# Assumes CARLA_ROOT is set.
SDL_VIDEODRIVER=offscreen ${CARLA_ROOT}/CarlaUE4.sh -windowed -ResX=800 -ResY=600 -carla-server -benchmark -fps=10
