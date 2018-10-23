#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../common"))

import rospy
import roslib
import math

from movement import Movement, AABB, grid_laser_check_for_danger
from viz import Viz

if __name__ == '__main__':

    rospy.init_node('aabb_test', anonymous=True)

    mov = Movement()
    mov.debug = True

    mov.pos = (0, 0)
    mov.orientation = 0

    while not rospy.is_shutdown():
        job = Movement.ForwardJob(mov, 1, 0.1, None, None)
        mov.job = job
        job.start()
        '''
        if (mov.last_laser_data is not None and
            grid_laser_check_for_danger(mov, mov.last_laser_data)):
            mov.viz.draw_AABB(AABB(-0.5, 0, 1, 1), color=(1, 0, 0, 1))
        else:
            mov.viz.draw_AABB(AABB(-0.5, 0, 1, 1), color=(0, 0, 1, 1))
        '''
        mov.job = None
        #mov.forward_1m()
        mov.forward_distance(1, 1, 0.1, grid_laser_check_for_danger)
        mov.rotate_angle(math.radians(90))
        #raw_input()

