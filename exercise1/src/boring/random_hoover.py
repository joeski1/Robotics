#!/usr/bin/env python
#
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../common"))

import rospy
import roslib
import math
import random

from movement import Movement

if __name__ == '__main__':
    rospy.init_node('hoover', anonymous=True)
    turn_counter = 0
    turn_direction = 1
    turn_deg = 25
    turn_amount = math.radians(turn_deg)
    mov = Movement()

    while not rospy.is_shutdown():
        mov.forward_distance(random.randrange(2, 8), 1)

        if random.randrange(0, turn_deg - turn_counter) is 0:
            mov.rotate_angle(turn_amount, 1)
            turn_direction = -turn_direction
            turn_counter = 0
        else:
            turn_counter += 1
        mov.rotate_angle(turn_direction * turn_amount, 1)

