
Note: all these steps must be run either in docker or on the robot because of
the required dependencies

# Building libnabo
- `cd libnabo`
- `mkdir build && cd build`
- `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../`
- `make -j 4`

# Building libpointmatcher
- https://libpointmatcher.readthedocs.io/en/latest/Compilation/
- `cd libpointmatcher`
- `mkdir build && cd build`
- `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ../`
- `make -j 4`

## Building libpointmatcher documentation
- `cd libpointmatcher`
- `mkdir build && cd build`
- `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -D GENERATE_API_DOC=true ../`
- `make -j 4`

