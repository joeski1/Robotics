# Writing a ROS node in C++ #

- edit `package.xml`
    - add `<build_depend>roscpp</build_depend>`
    - add `<run_depend>roscpp</run_depend>`
- edit `CMakeLists.txt`
    - add `find_package(catkin REQUIRED COMPONENTS roscpp std_msgs)`
    - add `include_directories(include ${catkin_INCLUDE_DIRS})`
    - add `add_executable(my_node src/fileA.cpp src/fileB.cpp)`
    - add `target_link_libraries(my_node ${catkin_LIBRARIES})`
- optional in cmake file:
    - add flags with: `set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")`

Run `catkin_make`

You will need to have sourced from `devel/setup.sh` (only exists after running
catkin_make) in order for the executables to be placed in the path, and be
executable using `rosrun`.


Sources
- `http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber(c%2B%2B)`




# Writing Custom Messages #

- all of these above `catkin_package()`
    - write a .msg file under the `msg` directory (sibling of `src`)
    - add `genmsg` to the list in `find_package(catkin REQUIRED COMPONENTS ...)`
    - add `add_message_files(DIRECTORY msg FILES my_msg.msg)`
    - add `generate_messages(DEPENDENCIES std_msgs)`
- below `add_executable(my_node src/my_node.cpp)`
    - add `add_dependencies(my_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})`

Sources
- `http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber(c%2B%2B)`
- didn't follow any of this but somehow got it to work. If something breaks
  then check this:
    - `http://docs.ros.org/hydro/api/catkin/html/howto/format2/building_msgs.html`
