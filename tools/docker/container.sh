#!/bin/bash

# useful links
# - http://wiki.ros.org/stage/Tutorials/SimulatingOneRobot

map_path="/workspace/src/exercise2/map/lgfloor" # "/workspace/src/socspioneer/lgfloor"
ROS_PACKAGE_PATH=/workspace/src:${ROS_PACKAGE_PATH}
export ROS_PACKAGE_PATH

echo "give the argument 'viz' to start rviz automatically"

# Copied from syncd
function check_shebang {
    FILE="$1"
    export FILE
    if [[ ! "$(head -n 1 "$FILE")" == *"#!"* ]]; then
        echo "does not have shebang $FILE"
    fi
}
export -f check_shebang
# set all python in the workspace as executable
find /workspace/src -type f -name "*.py" -exec chmod +x {} \;
CHECK=$(find /workspace/src -type f -name "*.py" -exec bash -c 'check_shebang "$0"' {} \;)
if [ ! -z "$CHECK" ]; then
    echo "some of the .py files do not have shebangs!!! Exiting... "
    echo "$CHECK"
    exit 1
    # TODO: dont exit here
fi

function background_services {
    # Start roscore
    roscore &>/dev/null &
    echo waiting for roscore to initialise
    sleep 3s # wait to start

    # Start map_server and amcl (localisation)
    rosrun map_server map_server "${map_path}.yaml" &>/dev/null &
    #rosrun amcl amcl scan:=base_scan &>/dev/null &
}

# Setup workspace
function create_workspace {
    cd /workspace/src/
    catkin_init_workspace
    cd /workspace
    catkin_make

    # to add c++ build executables into the path (they are placed in
    # /workspace/devel/lib/...
    source "/workspace/devel/setup.sh"

    ln -s "/workspace/src/tools/load_weights"

    # setup some helpful scripts
    echo "#!/bin/bash" > "/workspace/keyboard-driving"
    echo "xterm -e /workspace/src/tools/docker/keyboard-driving &" >> "/workspace/keyboard-driving"
    chmod +x "/workspace/keyboard-driving"

    echo "#!/bin/bash" >  "/workspace/run_slam"
    # only run if build succeeds
    echo "catkin_make && roslaunch exercise3 SLAM.launch" >> "/workspace/run_slam"
    chmod +x "/workspace/run_slam"

    echo "#!/bin/bash" >  "/workspace/run_stage"
    # need map_path to expand here and be written into the file
    echo "rosrun stage_ros stageros \"${map_path}.world\" &>/dev/null &" >> "/workspace/run_stage"
    echo '[ "$1" == "viz" ] && rosrun rviz rviz -d "/workspace/src/tools/docker/stage.rviz" &>/dev/null &' >> "/workspace/run_stage"
    echo '[ "$1" == "k" ] && ./keyboard-driving' >> "/workspace/run_stage"
    chmod +x "/workspace/run_stage"

    echo "#!/bin/bash" > "/workspace/replay"
    echo 'rosrun exercise3 message_replay.py "$1"' >> "/workspace/replay"
    chmod +x "/workspace/replay"

    echo 'pgrep python | xargs kill -9' >> "/workspace/kill_python"
    chmod +x "/workspace/kill_python"
}

# Only need to be run when the container is made
create_workspace
background_services

# Show visualisations (should be an option)
if [ "$1" == "viz" ]; then
    rosrun stage_ros stageros "${map_path}.world" &>/dev/null &
    rosrun rviz rviz -d "/workspace/src/tools/docker/stage.rviz" &>/dev/null &
fi

# Give a terminal
bash
