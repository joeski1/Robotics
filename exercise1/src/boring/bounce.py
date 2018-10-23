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

from movement import Movement, AABB, getOffendingReading, is_simulated
from viz import Viz

def sonar_in_danger(movement, data):
    return False

offendingReading = None
front_reading = None
if is_simulated():
    threshold = 0.5
else:
    threshold = 1

def log(x):
    rospy.loginfo(x)

def laser_in_danger(movement, data):
    global offendingReading
    global front_reading
    front_reading = data[255]
    offendingReading = getOffendingReading(data, threshold)
    return offendingReading is not None


if __name__ == '__main__':
    rospy.init_node('hoover', anonymous=True)
    mov = Movement()

    side_interval = 250
    side_turn = 25
    hard_turn = 55
    turn_counter = 0
    turn_limit = math.floor(540/hard_turn)

    while not rospy.is_shutdown():
        log("Moving Forward")
        start_pos = mov.pos if mov.pos is not None else (0, 0)
        mov.forward_distance(999, 10, sonar_check_for_danger=sonar_in_danger)

        laser_in_danger(mov, mov.last_laser_data)


        moved = math.sqrt((mov.pos[0]-start_pos[0])**2+(mov.pos[1]-start_pos[1])**2) > 0.1
        log('moved: ' + str(moved))


        if offendingReading is None or not moved:
            if offendingReading is None:
                log('offending reading is None')
            offendingReading = 255 # middle value => hard turn


        log('offending reading: ' + str(offendingReading))
        if offendingReading < side_interval:
            log("on the right")
            mov.rotate_angle(math.radians(side_turn))
        elif offendingReading > 512-side_interval:
            log("on the left")
            mov.rotate_angle(math.radians(-side_turn))
        else:
            log("Hard Turn")
            mov.rotate_angle(hard_turn if turn_counter < turn_limit else
                    -hard_turn, 1)
            turn_counter += 1 if turn_counter < 2 * turn_limit else -turn_counter



