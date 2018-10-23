FROM osrf/ros:indigo-desktop-full

# Install map-server
RUN apt-get update && apt-get install -y ros-indigo-map-server python3-tk

# Environment variables
#ENV ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/workspace/src
#ENV PATH=/opt/ros/indigo/bin:${PATH}
#ENV PYTHONPATH=/opt/ros/indigo/lib/python2.7/dist-packages
WORKDIR /workspace

# Create workspace
RUN mkdir -p /workspace/src
VOLUME /workspace/src

ADD container.sh /container.sh
ENTRYPOINT ["/ros_entrypoint.sh", "/workspace/src/tools/docker/container.sh"]
