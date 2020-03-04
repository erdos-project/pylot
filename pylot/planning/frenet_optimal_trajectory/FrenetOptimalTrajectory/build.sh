#!/bin/bash
sudo apt-get install -y libeigen3-dev clang cmake
if [ ! -d "build" ]; then
  mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target all -- -j 8
