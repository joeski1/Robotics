
# System Testing
- 3 sets of parameters
    - 'best'
        - config.h:54
            - #define STITCHING STITCH_EDGE
        - SLAM.h
            - sufficient_translate = square(0.25)
            - sufficient_rotate    = to_radians(20)
            - stitching_N = 4

    - 'more frequent local maps'
        - config.h:54
            - #define STITCHING STITCH_EDGE
        - SLAM.h
            - sufficient_translate = square(0.10)
            - sufficient_rotate    = to_radians(10)
            - stitching_N = 8

    - brute stitching
        - config.h:54
            - #define STITCHING STITCH_BRUTE
        - SLAM.h
            - sufficient_translate = square(0.25)
            - sufficient_rotate    = to_radians(20)
            - stitching_N = 1

- steps to run system test
    - `rosrun exercise3 message_gather.py system_test_N_`
    - `roslaunch exercise3 SLAM.launch`
    - press enter on the message gather terminal, then on the explore terminal
    - Ctrl+C the message gatherer
    - close the global map viewer and make sure it says 'saving map'
    - close everything else (closing the main SLAM should close everything)
    - zip up the global map and the message recordings with 7zip


# Gathering Message
- slow tour x2 (being kind)
- fast tour x2
- mixture tour x2
- wall hugger
- hoover
- spin on the spot for 30 seconds
    - in corridor
    - in open space
- up and down the middle a few times

- steps to gather data
    - `cd ~/workspace`
    - `./joystick-control`
    - `rosrun exercise3 message_gather.py system_test_N_`
    - press enter on the message gather terminal then drive
    - when finished, Ctrl+C the message gatherer
    - close everything including the global map viewer
    - zip up the recorded messages and global map images

