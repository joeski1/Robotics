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

from movement import Movement, AABB, canMoveForward
from viz import Viz

def sonar_in_danger(movement, data):
    return False

def laser_in_danger(movement, data):
    return not canMoveForward(data, 1)


if __name__ == '__main__':
    global mov
    rospy.init_node('hoover', anonymous=True)
    turn_amount = math.radians(30)
    turn_counter = 0
    turn_limit = math.floor(3*math.pi/turn_amount)
    mov = Movement()

    while not rospy.is_shutdown():
        rospy.loginfo("Forward")
        mov.forward_distance(999, 10,
                laser_check_for_danger=laser_in_danger,
                sonar_check_for_danger=sonar_in_danger)
        rospy.loginfo("Turn")
        mov.rotate_angle(turn_amount if turn_counter < turn_limit else
                -turn_amount, 1)
        turn_counter += 1 if turn_counter < 2 * turn_limit else -turn_counter
