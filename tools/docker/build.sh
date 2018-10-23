#!/bin/bash

# docker rmi --force rosimage


# Install map-server
apt-get update
apt-get install -y ros-indigo-map-server ros-indigo-amcl
# keyboard driving
sudo apt-get -y install ros-indigo-teleop-twist-keyboard

# scipy
#apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
apt-get install -y python-numpy python-scipy python-matplotlib

apt-get install -y python-tk
apt-get install -y gdb


# Create workspace
mkdir -p /workspace/src

# Copy container run script
cp /build/container.sh /
