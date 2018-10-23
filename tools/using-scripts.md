# Using the run script #
1. on the robot: run `./syncd`
2. on your laptop, write some code
3. on your laptop run `./run <package> <module>.py`


# Setup #
- create a workspace directory: `git clone https://github.com/mbway/intelligent-robotics.git workspace`
- create a symlink for the run command `ln -s workspace/tools/run run`
- run `./run`
    - enter `y` to create the `sync` directory
    - enter your uni username and password to sync with gitlab
- create a script called "commit_to_svn" for committing to svn:

```sh
#!/bin/bash
cd workspace/tools
./commit_to_svn "../../svn"
```

- run "./commit_to_svn"

# Setup In VM #
- download and run `VM_bootstrap`
