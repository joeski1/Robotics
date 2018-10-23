#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from global_map import GlobalMap
import explore
import rospy
import numpy as np

# messages
from exercise3.msg import local_map

# updates the values of the global map with the values of the local map, gx and gy are the x and y coordinates, respectively,
# of the top left corner of the local map in reference to the global map
def update_global_map(est, local_map):
    '''
        given the best estimate for the pose of the local map relative to the
        global map: take the information that the local map gives and update the
        probabilities of the global map.
    '''
    gx, gy, gt = est
    (max_global_Y, max_global_X) = GlobalMap.global_map.shape

    assert(local_map.cols + gy < max_global_Y and (len(local_map.local_map) / local_map.cols) + gx < max_global_X)

    # local_map an object described in exercise3/msg/local_map.msg
    # So local_map.local_map is a float32 array
    '''
    for y, i in enumerate(local_map.local_map):
        for x, j in enumerate(i):
            if(j[y] >= 0):
                P = GlobalMap.global_map[gx + x, gy + y]
                Pn = j[y]
                GlobalMap.global_map[gx + x, gy + y] = 1.0 - (1.0 + ((1.0-P)**-1.0 - 1.0) * Pn)**-1.0
    '''
    for y in range(0,(len(local_map.local_map)/local_map.cols) - 1):
        for x in range(0, local_map.cols - 1):
            i = (y * local_map.cols) + x
            Pn = local_map.local_map[i]
            if Pn >= 0:
                P = GlobalMap.global_map[gx + x, gy + y]
                GlobalMap.global_map[gx + x, gy + y] = 1.0 - (1.0 + ((1.0-P)**-1.0 - 1.0) * Pn)**-1.0


    # probably can use numpy functions to improve performance
    # eg create a numpy array and then a*b will do component-wise multiplication
    # of the two arrays
    #GlobalMap.global_map[0,0] = 0

    explore.handle_map_changed()
