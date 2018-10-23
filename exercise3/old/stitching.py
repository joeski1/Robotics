#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from global_map import GlobalMap
from update import update_global_map
import rospy
import numpy as np

# messages
from exercise3.msg import local_map

# globals
#   parameters
max_vary_x = 0 # maximum +/- estimated x position in metres
max_vary_y = 0 # maximum +/- estimated y position in metres
max_vary_t = 0 # maximum +/- estimated angle (yaw/theta) in radians

#   variables
# estimated position and orientation of the top left cell of the local map when
# imposed over the global map (in metres and radians)
local_map_pose = (0,0,0)

def stitch(local_map):
    '''
        vary x, y, t estimate of the last local map pose + change in odometry

        Pick the hypothesis which minimises error
    '''

    est_x, est_y, est_t = (0, 0, 0)
    update_global_map((est_x, est_y, est_t), local_map)

    # implicitly determine robot pose
    #GlobalMap.robot_pose = (0, 0, 0) # new best estimate




def local_map_callback(msg):
    # TODO: maybe shouldn't do a lot of processing in the callback?
    # TODO: then again they might be placed sufficiently far apart so a backlog doesn't form
    print('received local map {}'.format(msg.counter))
    #print(msg)
    stitch(msg)


def init():
    rospy.Subscriber('local_map', local_map, local_map_callback)


if __name__ == '__main__':
    rospy.init_node('stitching')
    init()
    rospy.spin()
