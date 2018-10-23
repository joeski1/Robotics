# Running ROS in docker #
- install docker (Ubuntu: `docker.io` package. Arch: `docker` package)
- create a symlink to `run-docker` outside the workspace folder i.e. `ln -s workspace/tools/run-docker`
- Run `./run-docker` to start the container (will take a while the first time)
- To rebuild the image, run `docker rmi rosimage`
